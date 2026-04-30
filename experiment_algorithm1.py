"""
experiment_algorithm1.py

Confidence-Gated Meta-Algorithm (Algorithm 1 from the paper)
─────────────────────────────────────────────────────────────

Implements the paper's theoretical framework: a deterministic routing
switchboard between a frozen LLM estimator and an adaptive LinUCB learner,
with passive covariance warm-start.

Architecture
────────────
  The master algorithm computes geometric uncertainty scores each round:

    LLM uncertainty:     u_LLM    = ‖φ(x, a_LLM)‖ / √c
        Models the LLM's "offline" confidence from pretraining.
        c = effective offline sample count (higher → lower uncertainty).

    LinUCB uncertainty:  u_LinUCB = √(φ(x, a_LinUCB)ᵀ V⁻¹ φ(x, a_LinUCB))
        Standard ridge regression variance from the online design matrix.

  Deterministic routing:
    if u_LLM < u_LinUCB  →  defer to frozen LLM
    else                  →  use adaptive LinUCB

  Passive covariance update (the warm-start mechanism):
    When the LLM is routed, LinUCB STILL updates its design matrix V
    with the feature vector of the LLM's chosen action:
        V ← V + φ(x, a_LLM) φ(x, a_LLM)ᵀ
    This shrinks LinUCB's confidence ellipsoid WITHOUT requiring the
    reward signal, allowing faster hand-off.

  "Reward Multiplication" variant (--passive_full):
    Optionally update both V and b during passive rounds (stronger
    empirically, but deviates from the strict proof in the paper).

Why this matters
────────────────
  - Early phase: LLM uncertainty is low (c is large), LinUCB is uninformed
    → router defers to LLM for good early performance.
  - Over time: passive V updates + active LinUCB rounds shrink LinUCB's
    uncertainty → deterministic hand-off to LinUCB.
  - Theoretical regret bound is maintained because routing is deterministic
    (no importance weighting needed).

Recommended usage (proving the hand-off with a weaker LLM):
  python experiment_algorithm1.py --llm_type small --llm_confidence 100
  python experiment_algorithm1.py --llm_type base  --llm_confidence 100
  python experiment_algorithm1.py --llm_type large --llm_confidence 100

The --llm_confidence parameter controls how long the LLM phase lasts:
  Higher c → LLM has lower uncertainty → longer LLM phase before hand-off.
  Lower  c → earlier switch to LinUCB.

Baselines tracked (all on the same data stream):
  Router     — the deterministic confidence-gated switchboard
  LLM only   — frozen LLM (argmax cosine similarity)
  Cold LinUCB — LinUCB without any warm-start (no passive updates)
"""

import torch
import argparse
import warnings
import numpy as np
import json
import gzip
import pickle
import gc
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from scipy.linalg import eigvalsh

# Suppress spurious numpy RuntimeWarnings from Apple Accelerate BLAS
# on ARM64 Macs. The matmul results are correct; Accelerate triggers
# intermediate FP exceptions that don't affect final values.
warnings.filterwarnings('ignore', message='.*encountered in matmul.*', category=RuntimeWarning)


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


def language_model_scores(pre_texts, post_texts, t5_tokenizer, t5_model,
                           entity_model, entity_embd, num_entities, cos, device):
    """
    Returns:
      labels : (batch_size,) — LLM's argmax predictions
      sims   : (batch_size, num_entities) — cosine similarities
    """
    sentences = [pre + ' [MASK]. ' + post for pre, post in zip(pre_texts, post_texts)]
    predicted = generate_mask_fillings(sentences, t5_model, t5_tokenizer, device)
    pred_embd = torch.FloatTensor(entity_model.encode(predicted)).to(device)
    pred_embd = pred_embd.unsqueeze(1).repeat(1, num_entities, 1)
    ent_exp = entity_embd.unsqueeze(0).repeat(len(pre_texts), 1, 1).to(device)
    sims = cos(pred_embd, ent_exp)                           # (batch, num_entities)
    labels = torch.argmax(sims, dim=-1).detach()             # (batch,)
    return labels, sims.detach().cpu().numpy()


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
                linucb_alpha=1.0, lam=0.1, alpha_decay=0.9985,
                llm_confidence=100.0, passive_full=False,
                llm_cache_path=None):
    """
    Parameters
    ----------
    llm_confidence : float
        Effective "offline sample count" for the LLM's design matrix.
        The LLM's geometric uncertainty is  u = ‖φ‖ / √c.
        Higher c → longer LLM phase before hand-off.
    passive_full : bool
        If False (default): passive rounds update only V (covariance).
        If True: passive rounds update both V and b (reward vector).
        The paper's proofs use V-only; b-too is an empirical variant.
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
    print(f"LLM: flan-t5-{llm_type} | alpha={linucb_alpha} | lam={lam}")
    print(f"LLM confidence (c): {llm_confidence}")
    print(f"Passive update mode: {'full (V + b)' if passive_full else 'covariance only (V)'}")
    print(f"Num actions: {num_entities}")

    # --- Load LLM cache or live models ---
    cached_labels = None
    if llm_cache_path:
        print(f"Loading LLM cache: {llm_cache_path}")
        with gzip.open(llm_cache_path, 'rb') as f:
            cache = pickle.load(f)
        cached_labels = cache['labels']  # (N,) int64
        print(f"  Cache: {cached_labels.shape[0]} samples, config: {cache['config']}")
        t5_model = t5_tokenizer = entity_model = cos = None
    else:
        t5_model, t5_tokenizer = load_model_and_tokenizer(f"google/flan-t5-{llm_type}", device)
        entity_model = SentenceTransformer('bert-base-nli-mean-tokens', device=device)
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    # Dedicated generator so shuffle order is identical across LLM sizes
    data_gen = torch.Generator()
    data_gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
        generator=data_gen
    )

    action_features = entity_embd.cpu().numpy()  # (num_actions, 768)
    d = action_features.shape[1]                 # 768

    print(f"Feature dim: {d}")

    # --- Warm LinUCB (receives passive covariance updates from LLM rounds) ---
    V_inv_warm = (1.0 / lam) * np.eye(d)
    b_warm = np.zeros(d)

    # --- Cold LinUCB (independent baseline, no help from LLM) ---
    V_inv_cold = (1.0 / lam) * np.eye(d)
    b_cold = np.zeros(d)

    # LLM's geometric uncertainty uses a fixed "offline" design matrix V_LLM = c * I
    # u_LLM(phi) = sqrt(phi^T (c*I)^{-1} phi) = ||phi|| / sqrt(c)
    inv_sqrt_c = 1.0 / np.sqrt(llm_confidence)

    # --- Accumulators ---
    acc_router = EasyAcc()    # Confidence-gated router
    acc_llm    = EasyAcc()    # LLM only (frozen baseline)
    acc_cold   = EasyAcc()    # Cold LinUCB (no warm-start)

    # --- Routing diagnostics ---
    llm_rounds   = 0          # cumulative rounds LLM was chosen by router
    total_rounds = 0

    # --- Theory validation accumulators (Theorem 2) ---
    # tau: detected when cumulative llm_fraction drops below TAU_THRESHOLD
    TAU_THRESHOLD = 0.05  # 5% LLM routing => handoff effectively done
    tau_detected = False
    tau_round = 0         # round number where tau is detected
    tau_batch = 0         # batch number where tau is detected
    # tail gap: avg(optimal_reward - llm_reward) for rounds after tau
    tail_gap_sum = 0.0
    tail_gap_count = 0
    # per-round LLM accuracy after tau (for epsilon_tail)
    llm_correct_post_tau = 0
    llm_total_post_tau = 0
    # feature norm tracking for L estimate
    max_feat_norm = 0.0

    history = {
        'batch': [], 'hybrid': [], 'llm': [], 'linucb_cold': [],
        'llm_fraction': [],    # fraction of rounds LLM was chosen (for hand-off plot)
        # Theory validation (Theorem 2) logged at checkpoints:
        'eigenvalues_warm_bottom10': [],   # bottom 10 eigenvalues of V_warm
        'eigenvalues_cold_bottom10': [],   # bottom 10 eigenvalues of V_cold
        'tail_gap_cumulative': [],         # running epsilon_tail after tau
        'tau_round': None,                 # round where handoff detected
        'tau_batch': None,                 # batch where handoff detected
        'max_feat_norm': [],               # running max ||phi||_2
        'config': {
            'algorithm': 'confidence_gated_routing',
            'features': 'elementwise_768d',
            'linucb_alpha': linucb_alpha, 'lam': lam,
            'llm_type': llm_type, 'alpha_decay': alpha_decay,
            'llm_confidence': llm_confidence,
            'passive_full': passive_full,
        }
    }

    tag = f"algo1_{llm_type}_c{llm_confidence}"
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
                    pre_texts, post_texts, t5_tokenizer, t5_model,
                    entity_model, entity_embd, num_entities, cos, device
                )

        for i in range(len(Xs)):
            context_x   = Xs[i].cpu().numpy()
            d_ctx       = context_x.shape[0] // 2
            cx          = context_x[:d_ctx]            # pre-text embedding (768-dim)
            true_label  = ys[i].item()
            llm_choice  = llm_labels[i].item()

            # Base features: element-wise product (K, 768)
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
                # optimal reward is always 1 (Precision@1 with binary reward)
                # LLM gap = 1 - r_llm (= 1 if wrong, 0 if correct)
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

            # --- LinUCB's proposal (UCB action selection) ---
            theta_warm = V_inv_warm @ b_warm
            Vf_warm    = joint_base @ V_inv_warm
            vars_warm  = np.sum(Vf_warm * joint_base, axis=1)
            ucb_warm   = joint_base @ theta_warm + current_alpha * np.sqrt(np.maximum(vars_warm, 0))
            linucb_choice = int(np.argmax(ucb_warm))

            # --- Feature vectors for each proposal ---
            x_llm    = joint_base[llm_choice]
            x_linucb = joint_base[linucb_choice]

            # --- Geometric uncertainty scores ---
            # LLM: u = sqrt(φᵀ (c·I)⁻¹ φ) = ‖φ‖ / √c
            u_llm = inv_sqrt_c * np.linalg.norm(x_llm)

            # LinUCB: u = sqrt(φᵀ V⁻¹ φ)  (standard ridge variance)
            u_linucb = np.sqrt(max(0.0, x_linucb @ V_inv_warm @ x_linucb))

            # --- Deterministic routing rule ---
            if u_llm < u_linucb:
                # ─── LLM is more confident → defer to LLM ───
                router_choice = llm_choice
                r_router = 1.0 if router_choice == true_label else 0.0
                llm_rounds += 1

                # PASSIVE COVARIANCE UPDATE:
                # Update LinUCB's design matrix with the LLM's chosen feature.
                # This is the "reward-relevant warm-start coverage" mechanism.
                # LinUCB's confidence ellipsoid shrinks even though it didn't act.
                V_inv_warm = sm_update(V_inv_warm, x_llm)

                if passive_full:
                    # Empirical variant: also update b with observed reward
                    b_warm += r_router * x_llm
                # Else: b is NOT updated (paper's theoretical version).
                # θ = V⁻¹ b stays near 0 during LLM phase, but V becomes
                # well-conditioned. When LinUCB takes over, it refines θ
                # efficiently because V is already well-conditioned.

            else:
                # ─── LinUCB is confident enough → route to LinUCB ───
                router_choice = linucb_choice
                r_router = 1.0 if router_choice == true_label else 0.0

                # FULL UPDATE: both V and b
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

            # --- Eigenvalue logging (bottom 10 of V_warm and V_cold) ---
            # V = inv(V_inv), eigenvalues of V = 1/eigenvalues of V_inv
            # Use eigvalsh on V_inv (symmetric), take bottom 10 of V = top 10 of 1/eig(V_inv)
            # But we want bottom eigenvalues of V, so we want the largest eigenvalues of V_inv
            # and invert them. Instead, just invert V_inv and compute directly.
            # For 768x768, eigvalsh takes ~0.1s — acceptable every 10 batches.
            try:
                eigs_warm_vinv = eigvalsh(V_inv_warm)  # ascending order
                # eigenvalues of V = 1/eigenvalues of V_inv
                # bottom eigs of V = 1/(top eigs of V_inv)
                eigs_warm_v = 1.0 / eigs_warm_vinv[-10:][::-1]  # bottom 10 of V, ascending
                history['eigenvalues_warm_bottom10'].append(eigs_warm_v.tolist())

                eigs_cold_vinv = eigvalsh(V_inv_cold)
                eigs_cold_v = 1.0 / eigs_cold_vinv[-10:][::-1]
                history['eigenvalues_cold_bottom10'].append(eigs_cold_v.tolist())
            except Exception as e:
                print(f"  eigenvalue computation failed: {e}")
                history['eigenvalues_warm_bottom10'].append([])
                history['eigenvalues_cold_bottom10'].append([])

            # --- Tail gap logging ---
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
        description='Algorithm 1: Confidence-Gated Routing with Passive Warm-Start'
    )
    parser.add_argument('--seed',           type=int,   default=1)
    parser.add_argument('--llm_type',       type=str,   default='small',
                        choices=['small', 'base', 'large'],
                        help='LLM size. Use "small" to clearly show the hand-off.')
    parser.add_argument('--linucb_alpha',   type=float, default=1.0,
                        help='UCB exploration bonus scale')
    parser.add_argument('--lam',            type=float, default=0.1,
                        help='LinUCB ridge regularization')
    parser.add_argument('--llm_confidence', type=float, default=100.0,
                        help='Effective offline sample count c for LLM design matrix. '
                             'Higher → longer LLM phase before hand-off. '
                             'Try 50, 100, 500.')
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
