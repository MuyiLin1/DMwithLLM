import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import json
import os
import torch.nn as nn
import numpy as np
import re
import gzip
import pickle
from tqdm import tqdm

# ==========================================
# 1. The Anisotropic Wrapper
# ==========================================
class AnisotropicLLMWrapper:
    def __init__(self, action_features, lam=1.0, L_sq=10000.0):
        self.action_features = action_features
        self.lam = lam
        self.L_sq = L_sq 

    def get_prediction_and_uncertainty(self, chosen_action_idx, context_x):
        z = self.action_features[chosen_action_idx]
        theta_llm = z.copy()
        
        x_norm_sq = np.dot(context_x, context_x)
        z_norm_sq = np.dot(z, z)
        xz_dot = np.dot(context_x, z)
        
        term1 = (1.0 / self.lam) * x_norm_sq
        term2 = (self.L_sq / (self.lam * (self.lam + self.L_sq * z_norm_sq))) * (xz_dot ** 2)
        uncertainty_llm = term1 - term2
        
        return theta_llm, uncertainty_llm

# ==========================================
# 2. Data Structures & Helpers
# ==========================================
class EasyAcc:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sumsq = 0

    def __iadd__(self, other):
        self.n += 1
        self.sum += other
        self.sumsq += other*other
        return self

    def mean(self):
        return self.sum / max(self.n, 1)

# Note: MyDataset definition must remain here so pickle can load it
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, threshold):
        pass # Full generation logic lives in make_data.py
            
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, index):
        return self.Xs[index], self.ys[index], self.pre_texts[index], self.post_texts[index], self.text_entities[index]

def loadMyDataset(threshold):
    with gzip.open(f'mydataset.{threshold}.pickle.gz', 'rb') as handle:
        return pickle.load(handle)

def load_model_and_tokenizer(model_name, device):
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer

def generate_mask_fillings(sentences, model, tokenizer, device):
    input_texts = [f"question: {sent.replace('[MASK]', '<extra_id_0>')}" for sent in sentences]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    # Fixed the warning by explicitly setting max_new_tokens
    output_ids = model.generate(input_ids, max_new_tokens=20)
    answers = [tokenizer.decode(ids, skip_special_tokens=True).split('<extra_id_0>')[-1].strip() for ids in output_ids]
    return answers

def language_model_outputs(pre_texts, post_texts, sentence_model, t5_tokenizer, t5_model, entity_model, entity_embd, num_entities, cos, device):
    sentences = [(pre_tokens + ' [MASK]. ' + post_tokens) for pre_tokens, post_tokens in zip(pre_texts, post_texts)]
    predicted_entities = generate_mask_fillings(sentences, t5_model, t5_tokenizer, device)
    predicted_entities_embd = torch.FloatTensor(entity_model.encode(predicted_entities)).to(device)
    predicted_entities_embd = torch.unsqueeze(predicted_entities_embd, 1).repeat(1, num_entities, 1)
    entity_embd_expanded = torch.unsqueeze(entity_embd, 0).repeat(len(pre_texts), 1, 1).to(device)
    cos_similarities = cos(predicted_entities_embd, entity_embd_expanded)
    predicted_labels = torch.argmax(cos_similarities, dim=-1)
    return predicted_labels.detach()

def get_embd(dataset):
    temp_d = {}
    for k in dataset.labelfeats.keys():
        id, embd = dataset.labelfeats[k]
        temp_d[id] = torch.FloatTensor(embd)
    return torch.stack([temp_d[k] for k in sorted(temp_d.keys())])

# ==========================================
# 3. The Online Learning Loop
# ==========================================
def learnOnline(dataset, rank, batch_size, cuda, seed, llm_type):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    entity_embd = get_embd(dataset)
    num_entities = entity_embd.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    print(f"Running on device: {device}")
    
    t5_model, t5_tokenizer = load_model_and_tokenizer("google/flan-t5-" + llm_type, device)
    # Keep consistent with the embedding model used in make_data.py
    entity_model = SentenceTransformer('bert-base-nli-mean-tokens', device=device)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize our Custom Wrapper
    action_features_np = entity_embd.cpu().numpy()
    llm_wrapper = AnisotropicLLMWrapper(action_features_np, lam=1.0, L_sq=10000.0)
    
    # Initialize LinUCB dynamically based on the exact embedding context size
    sample_X = next(iter(generator))[0]
    d_context = sample_X.shape[1] 
    V_lin = np.eye(d_context)
    b_lin = np.zeros(d_context)
    
    avreward_combined = EasyAcc()
    
    print("Starting Deterministic Routing Experiment...")
    
    for bno, (Xs, ys, pre_texts, post_texts, text_entities) in enumerate(generator):
        Xs, ys = Xs.to(device), ys.to(device)
        
        # 1. Get LLM Predictions using the REAL text contexts
        with torch.no_grad():
            lm_predicted_labels = language_model_outputs(pre_texts, post_texts, entity_model, t5_tokenizer, t5_model, entity_model, entity_embd, num_entities, cos, device)
            
        # 2. Iterate through the batch to route deterministically
        for i in range(len(Xs)):
            # Grab the mathematical vector for LinUCB math, not the string!
            context_x = Xs[i].cpu().numpy() 
            llm_choice = lm_predicted_labels[i].item()
            true_label = ys[i].item()
            
            # --- Our Routing Logic ---
            # Ask the Wrapper for the LLM's uncertainty
            _, u_llm = llm_wrapper.get_prediction_and_uncertainty(llm_choice, context_x)
            
            # Calculate LinUCB's uncertainty
            vx = np.linalg.solve(V_lin, context_x)
            u_lin = float(np.dot(context_x, vx))
            
            # The Deterministic Switchboard
            if u_llm <= u_lin:
                final_action = llm_choice
            else:
                theta_hat = np.linalg.solve(V_lin, b_lin)
                scores = action_features_np @ theta_hat 
                final_action = np.argmax(scores)
                
            # Update metrics and LinUCB background model
            reward = 1.0 if final_action == true_label else 0.0
            avreward_combined += reward
            
            V_lin += np.outer(context_x, context_x)
            b_lin += reward * context_x
            
        if bno % 10 == 0:
            print(f"Batch {bno} | Combined Accuracy: {avreward_combined.mean():.4f}")
            
    print("Experiment Complete!")
    print(f"Final Combined Accuracy: {avreward_combined.mean():.4f}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--llm_type', type=str, default='small')
    args = parser.parse_args()
    
    # Ensure this matches the threshold you used in make_data.py
    mydata = loadMyDataset(10) 
    learnOnline(mydata, rank=50, batch_size=32, cuda=True, seed=args.seed, llm_type=args.llm_type)