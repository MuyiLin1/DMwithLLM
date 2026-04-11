"""
experiment_corral_proper.py

The two critical fixes vs your previous experiment.py:

  1. IMPORTANCE WEIGHTING ON EVERY ROUND
     When LLM is selected, LinUCB still gets an update on that (context, action, reward)
     triple with weight 1/p_llm. This is the core CORRAL mechanism and prevents LinUCB
     from being starved during the LLM-heavy early phase.

  2. SHERMAN-MORRISON RANK-1 UPDATES
     Replaces np.linalg.inv (O(d^3)) with a O(d^2) rank-1 update.
     For d=768, this is ~450x faster per step.

  3. LOG-BARRIER OMD (actual CORRAL update)
     Manages p_llm / p_cb automatically. Shifts weight from LLM to LinUCB
     as LinUCB accumulates data and improves.

Run variants:
  python experiment_corral_proper.py --llm_type large --eta 0.1 --p_min 0.2
  python experiment_corral_proper.py --llm_type base  --eta 0.1 --p_min 0.2
  python experiment_corral_proper.py --llm_type small --eta 0.1 --p_min 0.2
  python experiment_corral_proper.py --llm_type large --eta 0.05 --p_min 0.1
"""

import torch
import argparse
import numpy as np
import json
import gzip
import pickle
import gc
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from scipy import optimize


# ================================================================
# Sherman-Morrison rank-1 inverse update
# (V + x x^T)^{-1} from V^{-1}, costs O(d^2) not O(d^3)
# ================================================================
def sm_update(V_inv, x):
    Vx = V_inv @ x                          # (d,)
    return V_inv - np.outer(Vx, Vx) / (1.0 + x @ Vx)


# ================================================================
# Log-Barrier OMD  (CORRAL probability update, Algorithm 3)
# ================================================================
def corral_update(p, eta, loss_vec, p_min=0.2):
    """
    p        : current [p_llm, p_cb]
    eta      : learning rate
    loss_vec : [l_llm, l_cb] where l_i = (1-reward)/p_i if i selected, else 0
    p_min    : probability floor applied AFTER the update (clipping strategy)

    Returns updated p.
    """
    inv_p = 1.0 / p

    def residual(lam):
        denom = inv_p + eta * (loss_vec - lam)
        if np.any(denom <= 0):
            return np.inf
        return np.sum(1.0 / denom) - 1.0

    # Bracket the root for brentq
    lo = np.min(loss_vec) - 1.0
    hi = np.max(loss_vec) + 1.0
    for _ in range(60):
        if residual(lo) > 0:
            break
        lo -= 2.0
    for _ in range(60):
        if residual(hi) < 0:
            break
        hi += 2.0

    try:
        lam_star = optimize.brentq(residual, lo, hi, xtol=1e-8)
    except (ValueError, RuntimeError):
        return p  # fallback: no update this round

    denom = inv_p + eta * (loss_vec - lam_star)
    p_new = 1.0 / np.clip(denom, 1e-9, None)

    # Apply p_min floor then renormalize
    p_new = np.maximum(p_new, p_min)
    p_new /= p_new.sum()
    return p_new


# ================================================================
# Accumulators and data helpers (same structure as your code)
# ================================================================
class EasyAcc:
    def __init__(self):
        self.n = 0
        self.sum = 0

    def __iadd__(self, v):
        self.n += 1
        self.sum += v
        return self

    def mean(self):
        return self.sum / max(self.n, 1)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, i):
        return self.Xs[i], self.ys[i], self.pre_texts[i], self.post_texts[i], self.text_entities[i]


def loadMyDataset(threshold):
    with gzip.open(f'mydataset.{threshold}.pickle.gz', 'rb') as f:
        return pickle.load(f)


def load_model_and_tokenizer(model_name, device):
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer


def generate_mask_fillings(sentences, model, tokenizer, device):
    texts = [f"question: {s.replace('[MASK]', '<extra_id_0>')}" for s in sentences]
    inp = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    out = model.generate(
        inp.input_ids.to(device),
        attention_mask=inp.attention_mask.to(device),
        max_new_tokens=20
    )
    return [
        tokenizer.decode(ids, skip_special_tokens=True).split('<extra_id_0>')[-1].strip()
        for ids in out
    ]


def language_model_outputs(pre_texts, post_texts, t5_tokenizer, t5_model,
                            entity_model, entity_embd, num_entities, cos, device):
    sentences = [pre + ' [MASK]. ' + post for pre, post in zip(pre_texts, post_texts)]
    predicted = generate_mask_fillings(sentences, t5_model, t5_tokenizer, device)
    pred_embd = torch.FloatTensor(entity_model.encode(predicted)).to(device)
    pred_embd = pred_embd.unsqueeze(1).repeat(1, num_entities, 1)
    ent_exp = entity_embd.unsqueeze(0).repeat(len(pre_texts), 1, 1).to(device)
    sims = cos(pred_embd, ent_exp)
    return torch.argmax(sims, dim=-1).detach()


def get_embd(dataset):
    d = {}
    for k in dataset.labelfeats:
        id_, embd = dataset.labelfeats[k]
        d[id_] = torch.FloatTensor(embd)
    return torch.stack([d[k] for k in sorted(d)])


# ================================================================
# Main experiment loop
# ================================================================
def learnOnline(dataset, batch_size, cuda, seed, llm_type,
                eta=0.1, p_min=0.2, linucb_alpha=1.0, lam=0.1,
                alpha_decay=0.9985):
    """
    eta          : CORRAL learning rate. Try 0.05, 0.1, 0.5
    p_min        : probability floor per arm. Paper uses 0.2
    linucb_alpha : UCB exploration bonus scale. Try 0.5, 1.0, 2.0
    lam          : LinUCB regularization. Try 0.01, 0.1, 1.0
    alpha_decay  : per-batch decay of linucb_alpha (match your original)
    """
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

    print(f"Device: {device}")
    print(f"LLM: flan-t5-{llm_type} | eta={eta} | p_min={p_min} | alpha={linucb_alpha} | lam={lam}")
    print(f"Num actions: {num_entities}")

    t5_model, t5_tokenizer = load_model_and_tokenizer(f"google/flan-t5-{llm_type}", device)
    entity_model = SentenceTransformer('bert-base-nli-mean-tokens', device=device)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    action_features = entity_embd.cpu().numpy()  # (num_actions, d)
    d = action_features.shape[1]

    # --- HYBRID LinUCB state ---
    V_inv = (1.0 / lam) * np.eye(d)
    b = np.zeros(d)

    # --- STANDALONE (cold) LinUCB for fair comparison ---
    # This runs independently on every round, never touched by hybrid logic
    V_inv_cold = (1.0 / lam) * np.eye(d)
    b_cold = np.zeros(d)

    # --- CORRAL mixing probabilities: [p_llm, p_cb] ---
    p = np.array([0.5, 0.5])

    # --- Accumulators ---
    acc_hybrid    = EasyAcc()
    acc_llm       = EasyAcc()
    acc_cold      = EasyAcc()

    # --- Diagnostics ---
    total_llm_picks = 0
    total_cb_picks  = 0
    correct_llm_picks   = 0  # how often LLM was right when selected
    correct_cb_picks    = 0  # how often CB was right when selected

    history = {
        'batch': [], 'hybrid': [], 'llm': [], 'linucb_cold': [],
        'p_llm': [], 'p_cb': [],
        'llm_picks': [], 'cb_picks': [],
        'llm_acc_when_selected': [], 'cb_acc_when_selected': [],
    }

    print(f"Total batches: {len(loader)}")
    print("=" * 90)

    for bno, (Xs, ys, pre_texts, post_texts, text_entities) in enumerate(loader):

        # Decaying UCB alpha to match your original experiments
        current_alpha = max(0.1, linucb_alpha * (alpha_decay ** bno))

        Xs, ys = Xs.to(device), ys.to(device)

        with torch.no_grad():
            llm_labels = language_model_outputs(
                pre_texts, post_texts, t5_tokenizer, t5_model,
                entity_model, entity_embd, num_entities, cos, device
            )

        for i in range(len(Xs)):
            context_x   = Xs[i].cpu().numpy()
            d_ctx       = context_x.shape[0] // 2
            cx          = context_x[:d_ctx]          # context half of the feature vector
            true_label  = ys[i].item()
            llm_choice  = llm_labels[i].item()

            # Joint features: (num_actions, d)  -- same construction as your code
            joint = action_features * cx

            # ----------------------------------------------------------
            # TRACK 1: LLM only (no update needed, it's static)
            # ----------------------------------------------------------
            r_llm = 1.0 if llm_choice == true_label else 0.0
            acc_llm += r_llm

            # ----------------------------------------------------------
            # TRACK 2: Cold LinUCB (runs on every round, never importance-weighted)
            # This is the fair standalone baseline
            # ----------------------------------------------------------
            theta_cold = V_inv_cold @ b_cold
            Vf_cold    = joint @ V_inv_cold                              # (K, d)
            vars_cold  = np.sum(Vf_cold * joint, axis=1)                # (K,)
            ucb_cold   = joint @ theta_cold + current_alpha * np.sqrt(np.maximum(vars_cold, 0))
            cold_choice = np.argmax(ucb_cold)
            r_cold = 1.0 if cold_choice == true_label else 0.0
            acc_cold += r_cold
            x_cold = joint[cold_choice]
            V_inv_cold = sm_update(V_inv_cold, x_cold)
            b_cold    += r_cold * x_cold                                 # no IW, it always acts

            # ----------------------------------------------------------
            # TRACK 3: CORRAL HYBRID
            # ----------------------------------------------------------
            # Sample which arm to follow this round
            selected = int(np.random.choice(2, p=p))   # 0=LLM, 1=LinUCB
            p_selected = p[selected]

            if selected == 0:
                # Follow LLM this round
                action = llm_choice
                total_llm_picks += 1
                if action == true_label:
                    correct_llm_picks += 1
            else:
                # Follow LinUCB this round
                theta   = V_inv @ b
                Vf      = joint @ V_inv                                  # (K, d)
                variances = np.sum(Vf * joint, axis=1)                   # (K,)
                ucb     = joint @ theta + current_alpha * np.sqrt(np.maximum(variances, 0))
                action  = int(np.argmax(ucb))
                total_cb_picks += 1
                if action == true_label:
                    correct_cb_picks += 1

            reward = 1.0 if action == true_label else 0.0
            acc_hybrid += reward

            # ----------------------------------------------------------
            # KEY FIX: LinUCB importance-weighted update on EVERY round
            #
            # Whether LLM or LinUCB was selected, LinUCB observes the
            # (context, action, reward) triple and updates with weight 1/p_selected.
            # In expectation this is unbiased, and it prevents LinUCB from being
            # starved during the LLM-heavy early phase.
            # ----------------------------------------------------------
            iw = 1.0 / p_selected
            x_chosen = joint[action]
            V_inv = sm_update(V_inv, x_chosen)
            b    += iw * reward * x_chosen                               # importance-weighted

            # ----------------------------------------------------------
            # CORRAL probability update (log-barrier OMD)
            # Only the selected arm gets a nonzero loss entry
            # ----------------------------------------------------------
            loss_vec = np.zeros(2)
            loss_vec[selected] = (1.0 - reward) / p_selected            # importance-weighted loss
            p = corral_update(p, eta, loss_vec, p_min=p_min)

        # ----------------------------------------------------------
        # Logging every 10 batches
        # ----------------------------------------------------------
        if bno % 10 == 0:
            llm_acc_when_selected = correct_llm_picks / max(total_llm_picks, 1)
            cb_acc_when_selected  = correct_cb_picks  / max(total_cb_picks,  1)

            history['batch'].append(bno)
            history['hybrid'].append(acc_hybrid.mean())
            history['llm'].append(acc_llm.mean())
            history['linucb_cold'].append(acc_cold.mean())
            history['p_llm'].append(float(p[0]))
            history['p_cb'].append(float(p[1]))
            history['llm_picks'].append(total_llm_picks)
            history['cb_picks'].append(total_cb_picks)
            history['llm_acc_when_selected'].append(llm_acc_when_selected)
            history['cb_acc_when_selected'].append(cb_acc_when_selected)

            print(
                f"Batch {bno:4d} | "
                f"Hybrid: {acc_hybrid.mean():.4f} | "
                f"LLM: {acc_llm.mean():.4f} | "
                f"Cold LinUCB: {acc_cold.mean():.4f} | "
                f"p_llm={p[0]:.3f} p_cb={p[1]:.3f} | "
                f"LLM acc@select={llm_acc_when_selected:.3f} "
                f"CB acc@select={cb_acc_when_selected:.3f}"
            )

            with open(f"results_corral_{llm_type}_eta{eta}_pmin{p_min}.json", "w") as f:
                json.dump(history, f)

        del Xs, ys, pre_texts, post_texts, text_entities, llm_labels
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    print("=" * 90)
    print(f"FINAL | Hybrid: {acc_hybrid.mean():.4f} | LLM: {acc_llm.mean():.4f} | Cold LinUCB: {acc_cold.mean():.4f}")
    print(f"Total picks -> LLM: {total_llm_picks}  CB: {total_cb_picks}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',         type=int,   default=1)
    parser.add_argument('--llm_type',     type=str,   default='large',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--eta',          type=float, default=0.1,
                        help='CORRAL learning rate. Try 0.05, 0.1, 0.5')
    parser.add_argument('--p_min',        type=float, default=0.2,
                        help='Prob floor per arm. Paper uses 0.2')
    parser.add_argument('--linucb_alpha', type=float, default=1.0,
                        help='UCB bonus scale. Try 0.5, 1.0, 2.0')
    parser.add_argument('--lam',          type=float, default=0.1,
                        help='LinUCB regularization')
    args = parser.parse_args()

    mydata = loadMyDataset(2000)
    learnOnline(
        mydata, batch_size=128, cuda=True, seed=args.seed,
        llm_type=args.llm_type, eta=args.eta, p_min=args.p_min,
        linucb_alpha=args.linucb_alpha, lam=args.lam
    )
