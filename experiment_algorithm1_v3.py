"""
experiment_algorithm1_v3.py

Modernized version of the confidence-gated routing experiment.

Changes from v2:
  1. SentenceTransformer: BAAI/bge-large-en-v1.5 (1024-dim) instead of
     all-mpnet-base-v2 (768-dim).
  2. LLM: Qwen2.5-Instruct (decoder-only, instruction-tuned) instead of
     Flan-T5 (encoder-decoder, mask-filling).
     Sizes: small=0.5B, base=3B, large=7B.
  3. Loads mydataset.2000.v3.pickle.gz (built with bge-large-en-v1.5).
  4. Includes all Theorem 2 theory-validation logging (eigenvalues, tau,
     epsilon_tail, max feature norm).
  5. Output files tagged with 'algo1v3_'.

Usage:
  python experiment_algorithm1_v3.py --llm_type small --llm_confidence 100
  python experiment_algorithm1_v3.py --llm_type base  --llm_confidence 100
  python experiment_algorithm1_v3.py --llm_type large --llm_confidence 100
"""

import warnings
import torch
import argparse
import numpy as np
import json
import gzip
import pickle
import gc
import torch.nn as nn

# Suppress spurious numpy RuntimeWarnings from Apple Accelerate BLAS
# on ARM64 Macs. The matmul results are correct; Accelerate triggers
# intermediate FP exceptions that don't affect final values.
warnings.filterwarnings('ignore', message='.*encountered in matmul.*', category=RuntimeWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from scipy.linalg import eigvalsh


# ================================================================
# Model configuration
# ================================================================
ST_MODEL = 'BAAI/bge-large-en-v1.5'  # 1024-dim

QWEN_MODELS = {
    'small': 'Qwen/Qwen2.5-0.5B-Instruct',   # 0.5B params
    'base':  'Qwen/Qwen2.5-3B-Instruct',      # 3B params
    'large': 'Qwen/Qwen2.5-7B-Instruct',      # 7B params
}


# ================================================================
# Sherman-Morrison rank-1 inverse update: O(d^2)
# ================================================================
def sm_update(V_inv, x):
    Vx = V_inv @ x
    return V_inv - np.outer(Vx, Vx) / (1.0 + x @ Vx)


# ================================================================
# Helpers
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
        return self.Xs[i], self.ys[i], self.pre_texts[i], self.post_texts[i], self.text_entities[i], i


def loadMyDataset(threshold):
    fname = f'mydataset.{threshold}.v3.pickle.gz'
    print(f'Loading dataset: {fname}')
    with gzip.open(fname, 'rb') as f:
        return pickle.load(f)


# ================================================================
# Qwen2.5 LLM: load and generate entity predictions
# ================================================================
def load_llm(llm_type, device):
    """Load a Qwen2.5-Instruct model and tokenizer."""
    model_name = QWEN_MODELS[llm_type]
    print(f"Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def generate_entity_predictions(pre_texts, post_texts, model, tokenizer, device):
    """
    Use Qwen2.5-Instruct to predict what entity fills the blank.

    Batches all samples into a single left-padded generate() call for speed.
    Returns a list of predicted entity name strings.
    """
    # Build chat-formatted prompts for each sample
    all_texts = []
    for pre, post in zip(pre_texts, post_texts):
        prompt = (
            f"What entity or person is being referred to in the blank? "
            f"Context: '{pre} ____ {post}' "
            f"Reply with ONLY the entity name, nothing else."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer with only the entity name, no explanation."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_texts.append(text)

    # Left-pad for batched decoder-only generation
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        all_texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    prompt_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    predictions = []
    for i in range(output_ids.shape[0]):
        new_tokens = output_ids[i, prompt_len:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Clean up: take first line, strip quotes/periods
        prediction = prediction.split('\n')[0].strip().strip('"').strip("'").rstrip('.')
        predictions.append(prediction)

    return predictions


def language_model_scores(pre_texts, post_texts, llm_model, llm_tokenizer,
                           entity_model, entity_embd, num_entities, cos, device):
    """
    Returns:
      labels : (batch_size,) — LLM's argmax predictions
      sims   : (batch_size, num_entities) — cosine similarities
    """
    predicted = generate_entity_predictions(
        pre_texts, post_texts, llm_model, llm_tokenizer, device
    )
    pred_embd = torch.FloatTensor(entity_model.encode(predicted)).to(device)
    pred_embd = pred_embd.unsqueeze(1).repeat(1, num_entities, 1)
    ent_exp = entity_embd.unsqueeze(0).repeat(len(pre_texts), 1, 1).to(device)
    sims = cos(pred_embd, ent_exp)                           # (batch, num_entities)
    labels = torch.argmax(sims, dim=-1).detach()             # (batch,)
    return labels, sims.detach().cpu().numpy()


def get_embd(dataset):
    """Extract entity embeddings from the pickle, sorted by id."""
    d = {}
    for k in dataset.labelfeats:
        id_, embd = dataset.labelfeats[k]
        d[id_] = torch.FloatTensor(embd)
    return torch.stack([d[k] for k in sorted(d)])


# ================================================================
# Main experiment loop
# ================================================================
def learnOnline(dataset, batch_size, cuda, seed, llm_type,
                linucb_alpha=1.0, lam=0.1, alpha_decay=0.9985,
                llm_confidence=100.0, passive_full=False,
                llm_cache_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Entity embeddings from the v3 pickle (bge-large-en-v1.5, 1024-dim)
    entity_embd = get_embd(dataset)
    num_entities = entity_embd.shape[0]

    if cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"LLM: {QWEN_MODELS[llm_type]} | alpha={linucb_alpha} | lam={lam}")
    print(f"LLM confidence (c): {llm_confidence}")
    print(f"Passive update mode: {'full (V + b)' if passive_full else 'covariance only (V)'}")
    print(f"Num actions: {num_entities}")
    print(f"SentenceTransformer: {ST_MODEL}")

    # --- Load LLM cache or live models ---
    cached_labels = None
    if llm_cache_path:
        print(f"Loading LLM cache: {llm_cache_path}")
        with gzip.open(llm_cache_path, 'rb') as f:
            cache = pickle.load(f)
        cached_labels = cache['labels']  # (N,) int64
        print(f"  Cache: {cached_labels.shape[0]} samples, config: {cache['config']}")
        llm_model = llm_tokenizer = entity_model = cos = None
    else:
        llm_model, llm_tokenizer = load_llm(llm_type, device)
        entity_model = SentenceTransformer(ST_MODEL, device=device)
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    print(f"Entity embeddings shape: {entity_embd.shape}")

    # Dedicated generator so shuffle order is identical across LLM sizes
    data_gen = torch.Generator()
    data_gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
        generator=data_gen
    )

    action_features = entity_embd.cpu().numpy()  # (num_actions, 1024)
    d = action_features.shape[1]                 # 1024

    # bge-large produces L2-normalized embeddings, so element-wise
    # products have small norm. Scale so ||phi|| ≈ 1.
    feat_scale = np.sqrt(d)  # sqrt(1024) = 32.0
    action_features = action_features * feat_scale
    print(f"Feature dim: {d}  (scaled by sqrt(d) = {feat_scale:.1f})")

    # --- Warm LinUCB (receives passive covariance updates from LLM rounds) ---
    V_inv_warm = (1.0 / lam) * np.eye(d)
    b_warm = np.zeros(d)

    # --- Cold LinUCB (independent baseline, no help from LLM) ---
    V_inv_cold = (1.0 / lam) * np.eye(d)
    b_cold = np.zeros(d)

    inv_sqrt_c = 1.0 / np.sqrt(llm_confidence)

    # --- Accumulators ---
    acc_router = EasyAcc()
    acc_llm    = EasyAcc()
    acc_cold   = EasyAcc()

    # --- Routing diagnostics ---
    llm_rounds   = 0
    total_rounds = 0

    # --- Theory validation accumulators (Theorem 2) ---
    TAU_THRESHOLD = 0.05
    tau_detected = False
    tau_round = 0
    tau_batch = 0
    tail_gap_sum = 0.0
    tail_gap_count = 0
    llm_correct_post_tau = 0
    llm_total_post_tau = 0
    max_feat_norm = 0.0

    history = {
        'batch': [], 'hybrid': [], 'llm': [], 'linucb_cold': [],
        'llm_fraction': [],
        'eigenvalues_warm_bottom10': [],
        'eigenvalues_cold_bottom10': [],
        'tail_gap_cumulative': [],
        'tau_round': None,
        'tau_batch': None,
        'max_feat_norm': [],
        'config': {
            'algorithm': 'confidence_gated_routing_v3',
            'features': 'elementwise_1024d',
            'sentence_transformer': ST_MODEL,
            'llm_model': QWEN_MODELS[llm_type],
            'linucb_alpha': linucb_alpha, 'lam': lam,
            'llm_type': llm_type, 'alpha_decay': alpha_decay,
            'llm_confidence': llm_confidence,
            'passive_full': passive_full,
        }
    }

    tag = f"algo1v3_{llm_type}_c{llm_confidence}"
    if passive_full:
        tag += "_pfull"
    if seed != 1:
        tag += f"_s{seed}"
    fname = f"results_{tag}.json"
    print(f"Results file: {fname}")
    print(f"Total batches: {len(loader)}")
    print("=" * 90)

    for bno, (Xs, ys, pre_texts, post_texts, text_entities, indices) in enumerate(loader):

        current_alpha = max(0.1, linucb_alpha * (alpha_decay ** bno))
        Xs, ys = Xs.to(device), ys.to(device)

        if cached_labels is not None:
            batch_idx = indices.numpy()
            llm_labels = torch.tensor(cached_labels[batch_idx])
            llm_sims = None  # not needed for bandit loop
        else:
            with torch.no_grad():
                llm_labels, llm_sims = language_model_scores(
                    pre_texts, post_texts, llm_model, llm_tokenizer,
                    entity_model, entity_embd, num_entities, cos, device
                )

        for i in range(len(Xs)):
            context_x   = Xs[i].cpu().numpy()
            d_ctx       = context_x.shape[0] // 2
            cx          = context_x[:d_ctx]            # pre-text embedding (1024-dim)
            true_label  = ys[i].item()
            llm_choice  = llm_labels[i].item()

            # Base features: element-wise product (K, 1024)
            joint_base = action_features * cx

            total_rounds += 1

            # Track max feature norm for L estimate
            feat_norms = np.linalg.norm(joint_base, axis=1)
            round_max_norm = float(np.max(feat_norms))
            if round_max_norm > max_feat_norm:
                max_feat_norm = round_max_norm

            # ===========================================================
            # TRACK 1: LLM only (static baseline)
            # ===========================================================
            r_llm = 1.0 if llm_choice == true_label else 0.0
            acc_llm += r_llm

            # --- Tail gap tracking (for Theorem 2) ---
            if tau_detected:
                tail_gap_sum += (1.0 - r_llm)
                tail_gap_count += 1
                llm_total_post_tau += 1
                llm_correct_post_tau += int(r_llm)

            # ===========================================================
            # TRACK 2: Cold LinUCB (no warm-start, acts every round)
            # ===========================================================
            theta_cold = V_inv_cold @ b_cold
            Vf_cold    = joint_base @ V_inv_cold
            vars_cold  = np.sum(Vf_cold * joint_base, axis=1)
            ucb_cold   = joint_base @ theta_cold + current_alpha * np.sqrt(np.maximum(vars_cold, 0))
            cold_choice = int(np.argmax(ucb_cold))
            r_cold = 1.0 if cold_choice == true_label else 0.0
            acc_cold += r_cold

            x_cold = joint_base[cold_choice]
            V_inv_cold = sm_update(V_inv_cold, x_cold)
            b_cold += r_cold * x_cold

            # ===========================================================
            # TRACK 3: Confidence-Gated Router (Algorithm 1)
            # ===========================================================
            theta_warm = V_inv_warm @ b_warm
            Vf_warm    = joint_base @ V_inv_warm
            vars_warm  = np.sum(Vf_warm * joint_base, axis=1)
            ucb_warm   = joint_base @ theta_warm + current_alpha * np.sqrt(np.maximum(vars_warm, 0))
            linucb_choice = int(np.argmax(ucb_warm))

            x_llm    = joint_base[llm_choice]
            x_linucb = joint_base[linucb_choice]

            u_llm = inv_sqrt_c * np.linalg.norm(x_llm)
            u_linucb = np.sqrt(max(0.0, x_linucb @ V_inv_warm @ x_linucb))

            if u_llm < u_linucb:
                router_choice = llm_choice
                r_router = 1.0 if router_choice == true_label else 0.0
                llm_rounds += 1

                V_inv_warm = sm_update(V_inv_warm, x_llm)
                if passive_full:
                    b_warm += r_router * x_llm
            else:
                router_choice = linucb_choice
                r_router = 1.0 if router_choice == true_label else 0.0

                V_inv_warm = sm_update(V_inv_warm, x_linucb)
                b_warm += r_router * x_linucb

            acc_router += r_router

        # ===========================================================
        # Logging every 10 batches
        # ===========================================================
        # --- Detect tau (handoff point) ---
        if not tau_detected:
            llm_frac_now = llm_rounds / max(total_rounds, 1)
            if total_rounds > batch_size * 10 and llm_frac_now < TAU_THRESHOLD:
                tau_detected = True
                tau_round = total_rounds
                tau_batch = bno
                history['tau_round'] = tau_round
                history['tau_batch'] = tau_batch
                print(f"  >> TAU DETECTED at batch {bno}, round {total_rounds} (LLM frac = {llm_frac_now:.3f})")

        if bno % 10 == 0:
            llm_frac = llm_rounds / max(total_rounds, 1)
            history['batch'].append(bno)
            history['hybrid'].append(acc_router.mean())
            history['llm'].append(acc_llm.mean())
            history['linucb_cold'].append(acc_cold.mean())
            history['llm_fraction'].append(llm_frac)

            try:
                eigs_warm_vinv = eigvalsh(V_inv_warm)
                eigs_warm_v = 1.0 / eigs_warm_vinv[-10:][::-1]
                history['eigenvalues_warm_bottom10'].append(eigs_warm_v.tolist())

                eigs_cold_vinv = eigvalsh(V_inv_cold)
                eigs_cold_v = 1.0 / eigs_cold_vinv[-10:][::-1]
                history['eigenvalues_cold_bottom10'].append(eigs_cold_v.tolist())
            except Exception as e:
                print(f"  eigenvalue computation failed: {e}")
                history['eigenvalues_warm_bottom10'].append([])
                history['eigenvalues_cold_bottom10'].append([])

            eps_tail = (tail_gap_sum / max(tail_gap_count, 1)) if tau_detected else None
            history['tail_gap_cumulative'].append(eps_tail)
            history['max_feat_norm'].append(max_feat_norm)

            print(
                f"Batch {bno:4d} | "
                f"Router: {acc_router.mean():.4f} | "
                f"LLM only: {acc_llm.mean():.4f} | "
                f"Cold LinUCB: {acc_cold.mean():.4f} | "
                f"LLM routing: {llm_frac:.1%}"
            )

            with open(fname, "w") as f:
                json.dump(history, f)

        del Xs, ys, pre_texts, post_texts, text_entities, indices, llm_labels, llm_sims
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    print("=" * 90)
    print(
        f"FINAL | Router: {acc_router.mean():.4f} | "
        f"LLM only: {acc_llm.mean():.4f} | "
        f"Cold LinUCB: {acc_cold.mean():.4f}"
    )
    print(f"LLM routing: {llm_rounds}/{total_rounds} = {llm_rounds/max(total_rounds,1):.1%}")
    if tau_detected:
        eps_tail_final = tail_gap_sum / max(tail_gap_count, 1)
        print(f"Tau detected at round {tau_round} (batch {tau_batch})")
        print(f"Epsilon_tail (avg LLM gap post-tau): {eps_tail_final:.4f}")
        print(f"LLM accuracy post-tau: {llm_correct_post_tau}/{llm_total_post_tau} = {llm_correct_post_tau/max(llm_total_post_tau,1):.4f}")
    else:
        print("Tau NOT detected (LLM routing never dropped below threshold)")
    print(f"Max feature norm L: {max_feat_norm:.4f}")

    with open(fname, "w") as f:
        json.dump(history, f)
    print(f"Results saved to: {fname}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm 1 v3: Qwen2.5 + bge-large-en-v1.5'
    )
    parser.add_argument('--seed',           type=int,   default=1)
    parser.add_argument('--llm_type',       type=str,   default='small',
                        choices=['small', 'base', 'large'],
                        help='LLM size: small=Qwen2.5-0.5B, base=Qwen2.5-3B, large=Qwen2.5-7B')
    parser.add_argument('--linucb_alpha',   type=float, default=1.0)
    parser.add_argument('--lam',            type=float, default=0.1)
    parser.add_argument('--llm_confidence', type=float, default=100.0)
    parser.add_argument('--passive_full',   action='store_true',
                        help='If set, passive updates include reward in b. '
                             'Default: covariance only (paper\'s theoretical version).')
    parser.add_argument('--llm_cache',      type=str, default=None,
                        help='Path to cached LLM predictions (from cache_llm_predictions.py). '
                             'If set, skips LLM loading entirely.')
    args = parser.parse_args()

    mydata = loadMyDataset(2000)
    learnOnline(
        mydata, batch_size=128, cuda=True, seed=args.seed,
        llm_type=args.llm_type,
        linucb_alpha=args.linucb_alpha, lam=args.lam,
        llm_confidence=args.llm_confidence,
        passive_full=args.passive_full,
        llm_cache_path=args.llm_cache,
    )
