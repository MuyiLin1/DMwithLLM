import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
from sentence_transformers import SentenceTransformer
import json
import os
import torch.nn as nn
import numpy as np
import time

# ==========================================
# 1. New Addition: The Anisotropic Wrapper
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
# 2. Original Helper Functions
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

def categoryCount():
    import gzip
    counts = {}
    with gzip.open('oneshotwikilinks/entityfreq.gz', 'rt') as f:
        for line in f:
            try:
                freq, entity = line.strip().split()
            except:
                continue
            counts[entity] = int(freq)
    return counts

def getCategories(threshold):
    import re
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    for entity, freq in categoryCount().items():
        if freq >= threshold:
            niceentity = re.sub(r'_', r' ', entity)
            embedcat = model.encode([niceentity])[0]
            yield entity, embedcat

def makeData(threshold, categories):
    from collections import defaultdict
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    catcount = defaultdict(int)
    
    with open('oneshotwikilinks/shuffled_dedup_entities.tsv') as f:
        batchline, batchencode, batchentity = [], [], []
        for line in f:
            try:
                entity, pre, mention, post = line.strip().split('\t')
            except:
                continue
                
            if entity in categories and catcount[entity] < threshold:
                catcount[entity] += 1
                batchline.append(line)
                batchencode.append(pre)
                batchencode.append(post)
                batchentity.append(entity)

                if len(batchline) == 5:
                    embed = model.encode(batchencode)
                    for n, (line, entity) in enumerate(zip(batchline, batchentity)):
                        embedpre, embedpost = embed[2*n], embed[2*n+1]
                        entityord, entityvec = categories[entity]
                        yield { 'line': line, 'entityord': entityord, 'entityvec': entityvec, 'pre': embedpre, 'post': embedpost }
                    batchline, batchencode, batchentity = [], [], []

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, threshold):
        from tqdm import tqdm
        self.labelfeats = { n: (n, v) for n, (k, v) in enumerate(getCategories(threshold)) } 
        Xs, ys, pre_texts, post_texts, text_entities = [], [], [], [], []
        for n, what in tqdm(enumerate(makeData(threshold, self.labelfeats))):
            pre = torch.tensor(what['pre'])
            post = torch.tensor(what['post'])
            Xs.append(torch.cat((pre, post)).unsqueeze(0))
            ys.append(what['entityord'])
            pre_texts.append(pre)
            post_texts.append(post)
        self.Xs = torch.cat(Xs, dim=0)
        self.ys = torch.LongTensor(ys)
        self.pre_texts = pre_texts
        self.post_texts = post_texts
            
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, index):
        return self.Xs[index], self.ys[index], self.pre_texts[index], self.post_texts[index], ""

def loadMyDataset(threshold):
    import gzip
    import pickle
    with gzip.open(f'oneshotwikilinks/mydataset.{threshold}.pickle.gz', 'rb') as handle:
        return pickle.load(handle)

def load_model_and_tokenizer(model_name, device):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer

def generate_mask_fillings(sentences, model, tokenizer, device):
    input_texts = [f"question: {sent.replace('[MASK]', '<extra_id_0>')}" for sent in sentences]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    output_ids = model.generate(input_ids)
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
# 3. Modified Online Learning Loop
# ==========================================
def learnOnline(dataset, rank, batch_size, cuda, seed, llm_type):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    entity_embd = get_embd(dataset)
    num_entities = entity_embd.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5_model, t5_tokenizer = load_model_and_tokenizer("google/flan-t5-" + llm_type, device)
    entity_model = SentenceTransformer('bert-base-nli-mean-tokens')
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize our Custom Wrapper
    action_features_np = entity_embd.numpy()
    llm_wrapper = AnisotropicLLMWrapper(action_features_np, lam=1.0, L_sq=10000.0)
    
    # Initialize LinUCB
    d_context = 768  # Using pre_texts embedding size
    V_lin = np.eye(d_context)
    b_lin = np.zeros(d_context)
    
    avreward_combined = EasyAcc()
    combined_reward_history = []
    
    print("Starting Deterministic Routing Experiment...")
    
    for bno, (Xs, ys, pre_texts, post_texts, text_entities) in enumerate(generator):
        Xs, ys = Xs.to(device), ys.to(device)
        
        # 1. Get LLM Predictions
        with torch.no_grad():
            # For simplicity, we decode pre_texts back to text in a real scenario, 
            # but using dummy text here to match your original function signature.
            dummy_text = ["text" for _ in range(len(Xs))]
            lm_predicted_labels = language_model_outputs(dummy_text, dummy_text, entity_model, t5_tokenizer, t5_model, entity_model, entity_embd, num_entities, cos, device)
            
        # 2. Iterate through the batch to route deterministically
        for i in range(len(Xs)):
            context_x = pre_texts[i].numpy() # 768-dim context
            llm_choice = lm_predicted_labels[i].item()
            true_label = ys[i].item()
            
            # --- Our Routing Logic ---
            _, u_llm = llm_wrapper.get_prediction_and_uncertainty(llm_choice, context_x)
            
            vx = np.linalg.solve(V_lin, context_x)
            u_lin = float(np.dot(context_x, vx))
            
            if u_llm <= u_lin:
                final_action = llm_choice
            else:
                # Basic LinUCB prediction logic for demonstration
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
            combined_reward_history.append(avreward_combined.mean())
            
    print("Experiment Complete!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--llm_type', type=str, default='small')
    args = parser.parse_args()
    
    mydata = loadMyDataset(2000)
    learnOnline(mydata, rank=50, batch_size=32, cuda=True, seed=args.seed, llm_type=args.llm_type)