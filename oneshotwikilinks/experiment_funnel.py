import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import json
import os
import torch.nn as nn
import numpy as np
import gzip
import pickle
import gc
from tqdm import tqdm

# ==========================================
# 1. Data Structures & Helpers
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

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, threshold):
        pass 
        
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
    attention_mask = inputs.attention_mask.to(device)
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20)
    answers = [tokenizer.decode(ids, skip_special_tokens=True).split('<extra_id_0>')[-1].strip() for ids in output_ids]
    return answers

# =========================================================
# PATH B KEY CHANGE: Returning Top-K instead of Argmax
# =========================================================
def language_model_outputs(pre_texts, post_texts, sentence_model, t5_tokenizer, t5_model, entity_model, entity_embd, num_entities, cos, device, top_k=5):
    sentences = [(pre_tokens + ' [MASK]. ' + post_tokens) for pre_tokens, post_tokens in zip(pre_texts, post_texts)]
    predicted_entities = generate_mask_fillings(sentences, t5_model, t5_tokenizer, device)
    predicted_entities_embd = torch.FloatTensor(entity_model.encode(predicted_entities)).to(device)
    predicted_entities_embd = torch.unsqueeze(predicted_entities_embd, 1).repeat(1, num_entities, 1)
    
    entity_embd_expanded = torch.unsqueeze(entity_embd, 0).repeat(len(pre_texts), 1, 1).to(device)
    cos_similarities = cos(predicted_entities_embd, entity_embd_expanded)
    
    # We now ask the LLM for its Top 5 highest confidence answers
    _, top_k_indices = torch.topk(cos_similarities, k=top_k, dim=-1)
    return top_k_indices.detach()

def get_embd(dataset):
    temp_d = {}
    for k in dataset.labelfeats.keys():
        id, embd = dataset.labelfeats[k]
        temp_d[id] = torch.FloatTensor(embd)
    return torch.stack([temp_d[k] for k in sorted(temp_d.keys())])

# ==========================================
# 2. The Online Learning Loop
# ==========================================
def learnOnline(dataset, rank, batch_size, cuda, seed, llm_type):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    entity_embd = get_embd(dataset)
    num_entities = entity_embd.shape[0]
    
    if cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Running on device: {device}")
    
    t5_model, t5_tokenizer = load_model_and_tokenizer("google/flan-t5-" + llm_type, device)
    entity_model = SentenceTransformer('bert-base-nli-mean-tokens', device=device)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    action_features_np = entity_embd.cpu().numpy()
    d_context = action_features_np.shape[1] 
    
    # ---------------------------------------------------------
    # EXPERIMENT INITIALIZATIONS
    # ---------------------------------------------------------
    # 1. The Funneled LinUCB (Our Method)
    V = 0.1 * np.eye(d_context)
    V_inv = np.linalg.inv(V)
    b_lin = np.zeros(d_context)
    
    # 2. Cold LinUCB Initialization (Searching the entire haystack)
    V_cold = 0.1 * np.eye(d_context)
    V_cold_inv = np.linalg.inv(V_cold)
    b_cold = np.zeros(d_context)
    
    # Trackers
    acc_hybrid = EasyAcc()
    acc_llm_only = EasyAcc()
    acc_linucb_only = EasyAcc()
    
    history_batches = []
    history_hybrid = []
    history_llm = []
    history_linucb = []
    
    # Funnel Size
    TOP_K = 5
    
    print(f"Starting Path B (Funnel) Experiment with LLM: {llm_type}...")
    print(f"LLM filtering {num_entities} entities down to Top-{TOP_K} for LinUCB.")
    print(f"Total batches to process: {len(generator)}")

    for bno, (Xs, ys, pre_texts, post_texts, text_entities) in enumerate(generator):
        current_alpha = max(0.1, 1.0 * (0.9985 ** bno))
        Xs, ys = Xs.to(device), ys.to(device)
        
        with torch.no_grad():
            lm_top_k_indices = language_model_outputs(pre_texts, post_texts, entity_model, t5_tokenizer, t5_model, entity_model, entity_embd, num_entities, cos, device, top_k=TOP_K)
            
        batch_llm_correct = 0.0
        batch_hybrid_correct = 0.0
            
        for i in range(len(Xs)):
            context_x = Xs[i].cpu().numpy() 
            d = context_x.shape[0] // 2
            context_x_llm = context_x[:d]
            
            # The LLM's #1 absolute best guess
            top_k_options = lm_top_k_indices[i].cpu().numpy()
            llm_choice = top_k_options[0] 
            true_label = ys[i].item()
            
            joint_features_all = action_features_np * context_x_llm 
            x_norm_sq = float(np.dot(context_x_llm, context_x_llm))

            # =========================================================
            # TRACK 1: BASELINE LLM ONLY (Its #1 Pick)
            # =========================================================
            reward_llm = 1.0 if llm_choice == true_label else 0.0
            acc_llm_only += reward_llm
            batch_llm_correct += reward_llm

            # =========================================================
            # TRACK 2: BASELINE COLD LINUCB (Searches all 10k entities)
            # =========================================================
            theta_cold = V_cold_inv @ b_cold
            base_scores_cold = joint_features_all @ theta_cold
            
            V_inv_features_cold = joint_features_all @ V_cold_inv
            variances_cold = np.sum(joint_features_all * V_inv_features_cold, axis=1)
            relative_var_cold = variances_cold / x_norm_sq if x_norm_sq > 0 else variances_cold
            
            ucb_cold = base_scores_cold + current_alpha * np.sqrt(relative_var_cold)
            choice_cold = np.argmax(ucb_cold)
            
            reward_cold = 1.0 if choice_cold == true_label else 0.0
            acc_linucb_only += reward_cold
            
            chosen_x_cold = joint_features_all[choice_cold]
            V_cold += np.outer(chosen_x_cold, chosen_x_cold)
            V_cold_inv = np.linalg.inv(V_cold)
            b_cold += reward_cold * chosen_x_cold

            # =========================================================
            # TRACK 3: PATH B HYBRID (LinUCB searches ONLY Top-5)
            # =========================================================
            # Filter the massive feature array down to just the 5 rows the LLM suggested
            subset_features = joint_features_all[top_k_options]
            
            theta_hat = V_inv @ b_lin
            base_scores = subset_features @ theta_hat
            
            V_inv_features = subset_features @ V_inv
            variances = np.sum(subset_features * V_inv_features, axis=1)
            relative_variances = variances / x_norm_sq if x_norm_sq > 0 else variances
                
            # Calculate UCB scores for ONLY those 5 options
            ucb_scores = base_scores + current_alpha * np.sqrt(relative_variances)
            
            # Find the winner among the 5
            best_local_idx = np.argmax(ucb_scores)
            
            # Map that winner back to its true global ID
            final_action = top_k_options[best_local_idx]
            
            reward_hybrid = 1.0 if final_action == true_label else 0.0
            acc_hybrid += reward_hybrid
            batch_hybrid_correct += reward_hybrid
            
            # Standard matrix inversion update based on the hybrid's choice
            chosen_x_hybrid = subset_features[best_local_idx]
            V += np.outer(chosen_x_hybrid, chosen_x_hybrid)
            V_inv = np.linalg.inv(V)
            b_lin += reward_hybrid * chosen_x_hybrid
            
        # =========================================================
        # LOGGING & CONTINUOUS SAVING
        # =========================================================
        if bno % 10 == 0:
            history_batches.append(bno)
            history_hybrid.append(acc_hybrid.mean())
            history_llm.append(acc_llm_only.mean())
            history_linucb.append(acc_linucb_only.mean())
            
            batch_llm_acc = batch_llm_correct / len(Xs)
            batch_hybrid_acc = batch_hybrid_correct / len(Xs)
            
            print(f"Batch {bno:4d} | Global Hybrid (Top-5): {acc_hybrid.mean():.4f} | Global LLM (Top-1): {acc_llm_only.mean():.4f} | Global UCB (Full Space): {acc_linucb_only.mean():.4f}")
            print(f"           > THIS BATCH | LLM Acc: {batch_llm_acc:.4f} | Hybrid Acc: {batch_hybrid_acc:.4f}")
            
            results = {
                "batches": history_batches,
                "hybrid": history_hybrid,
                "llm_only": history_llm,
                "linucb_only": history_linucb
            }
            with open("experiment_results_funnel.json", "w") as f:
                json.dump(results, f)
                
        # =========================================================
        # MEMORY CLEANUP
        # =========================================================
        del Xs, ys, pre_texts, post_texts, text_entities, lm_top_k_indices
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    print("\nExperiment Complete!")
    print(f"Final Hybrid Accuracy: {acc_hybrid.mean():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--llm_type', type=str, default='large') 
    args = parser.parse_args()
    
    mydata = loadMyDataset(2000)
    learnOnline(mydata, rank=50, batch_size=128, cuda=True, seed=args.seed, llm_type=args.llm_type)
