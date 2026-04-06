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
    
    # Disabled multiprocessing (num_workers=0) to prevent memory leaks
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    action_features_np = entity_embd.cpu().numpy()
    llm_wrapper = AnisotropicLLMWrapper(action_features_np, lam=1.0, L_sq=10000.0)
    
    d_context = action_features_np.shape[1] 
    
    # ---------------------------------------------------------
    # EXPERIMENT INITIALIZATIONS
    # ---------------------------------------------------------
    # 1. Hybrid Method Initialization (NEW: 0.1 Ridge Regularization)
    V = 0.1 * np.eye(d_context)
    V_inv = np.linalg.inv(V)
    b_lin = np.zeros(d_context)
    
    # Adapter Initialization
    adapter_theta = np.zeros(d_context)
    adapter_lr = 0.0001 
    
    # 2. Cold LinUCB Initialization (NEW: 0.1 Ridge Regularization)
    V_cold = 0.1 * np.eye(d_context)
    V_cold_inv = np.linalg.inv(V_cold)
    b_cold = np.zeros(d_context)
    
    # 3. Trackers
    acc_hybrid = EasyAcc()
    acc_llm_only = EasyAcc()
    acc_linucb_only = EasyAcc()
    
    # 4. JSON Graph History Lists
    history_batches = []
    history_hybrid = []
    history_llm = []
    history_linucb = []
    
    print(f"Starting 3-Baseline Adaptive Routing Experiment with LLM: {llm_type}...")
    print(f"Total batches to process: {len(generator)}")
    
    total_llm_picks = 0
    total_linucb_picks = 0

    for bno, (Xs, ys, pre_texts, post_texts, text_entities) in enumerate(generator):
        
        # Calculate dynamic alpha at the start of the batch loop
        current_alpha = max(0.1, 1.0 * (0.9985 ** bno))
        
        # --- NEW: Decay the Adapter's grudge so it lets the LLM back in later ---
        adapter_theta *= 0.99 
        
        Xs, ys = Xs.to(device), ys.to(device)
        
        with torch.no_grad():
            lm_predicted_labels = language_model_outputs(pre_texts, post_texts, entity_model, t5_tokenizer, t5_model, entity_model, entity_embd, num_entities, cos, device)
            
        for i in range(len(Xs)):
            context_x = Xs[i].cpu().numpy() 
            d = context_x.shape[0] // 2
            context_x_llm = context_x[:d]
            
            llm_choice = lm_predicted_labels[i].item()
            true_label = ys[i].item()
            
            joint_features_all = action_features_np * context_x_llm 
            x_norm_sq = float(np.dot(context_x_llm, context_x_llm))

            # =========================================================
            # TRACK 1: BASELINE LLM ONLY
            # =========================================================
            reward_llm = 1.0 if llm_choice == true_label else 0.0
            acc_llm_only += reward_llm

            # =========================================================
            # TRACK 2: BASELINE COLD LINUCB ONLY
            # =========================================================
            theta_cold = V_cold_inv @ b_cold
            base_scores_cold = joint_features_all @ theta_cold
            
            V_inv_features_cold = joint_features_all @ V_cold_inv
            variances_cold = np.sum(joint_features_all * V_inv_features_cold, axis=1)
            relative_var_cold = variances_cold / x_norm_sq if x_norm_sq > 0 else variances_cold
            
            # Use dynamic alpha
            ucb_cold = base_scores_cold + current_alpha * np.sqrt(relative_var_cold)
            choice_cold = np.argmax(ucb_cold)
            
            reward_cold = 1.0 if choice_cold == true_label else 0.0
            acc_linucb_only += reward_cold
            
            # Standard matrix inversion update
            chosen_x_cold = joint_features_all[choice_cold]
            V_cold += np.outer(chosen_x_cold, chosen_x_cold)
            V_cold_inv = np.linalg.inv(V_cold)
            b_cold += reward_cold * chosen_x_cold

            # =========================================================
            # TRACK 3: OUR METHOD (HYBRID)
            # =========================================================
            _, u_llm = llm_wrapper.get_prediction_and_uncertainty(llm_choice, context_x_llm)
            
            theta_hat = V_inv @ b_lin
            base_scores = joint_features_all @ theta_hat
            
            V_inv_features = joint_features_all @ V_inv
            variances = np.sum(joint_features_all * V_inv_features, axis=1)
            
            if x_norm_sq > 0:
                relative_variances = variances / x_norm_sq
                relative_u_llm = u_llm / x_norm_sq
            else:
                relative_variances = variances
                relative_u_llm = u_llm
                
            # Adapter (The B.S. Detector) - NEW: max penalty cap is 1.5
            calibration_penalty = float(np.dot(adapter_theta, context_x_llm))
            penalty_multiplier = 1.0 + min(1.5, max(0.0, calibration_penalty))
            calibrated_u_llm = relative_u_llm * penalty_multiplier
            
            # Hybrid LinUCB Proposal - Use dynamic alpha
            ucb_scores = base_scores + current_alpha * np.sqrt(relative_variances)
            linucb_choice = np.argmax(ucb_scores)
            
            # Switchboard
            runway_weight = 5000.0 
            u_lin_proposed = relative_variances[linucb_choice] * runway_weight
            
            if calibrated_u_llm <= u_lin_proposed:
                final_action = llm_choice
                total_llm_picks += 1
            else:
                final_action = linucb_choice
                total_linucb_picks += 1
                
            # Hybrid Updates
            reward_hybrid = 1.0 if final_action == true_label else 0.0
            acc_hybrid += reward_hybrid
            
            if final_action == llm_choice:
                llm_error = 1.0 - reward_hybrid
                adapter_theta += adapter_lr * llm_error * context_x_llm
            
            # Standard matrix inversion update
            chosen_x_hybrid = joint_features_all[final_action]
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
            
            print(f"Batch {bno:4d} | Hybrid: {acc_hybrid.mean():.4f} | LLM Only: {acc_llm_only.mean():.4f} | Cold UCB: {acc_linucb_only.mean():.4f} || Hybrid Picks -> LLM: {total_llm_picks}, UCB: {total_linucb_picks}")
            
            results = {
                "batches": history_batches,
                "hybrid": history_hybrid,
                "llm_only": history_llm,
                "linucb_only": history_linucb
            }
            with open("experiment_results.json", "w") as f:
                json.dump(results, f)
                
        # =========================================================
        # MEMORY CLEANUP (Run at the end of EVERY batch)
        # =========================================================
        del Xs, ys, pre_texts, post_texts, text_entities, lm_predicted_labels
        
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
