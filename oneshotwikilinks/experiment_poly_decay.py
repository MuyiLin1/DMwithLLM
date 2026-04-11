"""
experiment_poly_decay.py

Simpler sanity check before trusting CORRAL.

Instead of CORRAL managing p_llm adaptively, we use a pre-determined
polynomial schedule: p_llm(t) = min(p_max, max(p_min, C / t^alpha)).

LinUCB still gets importance-weighted updates on EVERY round (same as
experiment_corral_proper.py). The only difference is that we don't use
log-barrier OMD -- the LLM probability is just a fixed schedule.

WHY THIS IS USEFUL:
  - If this works (hybrid > both baselines), the importance-weighting fix alone
    is what matters, not the CORRAL update.
  - If this ALSO fails (hybrid <= cold LinUCB), then LinUCB is fundamentally
    not strong enough on this task regardless of how we route. That's when
    you need to think about replacing LinUCB with SpannerGreedy.

From the paper's ablation (Table 5, OneShotWikiLinks, Flan-T5 large):
  Polynomial decay:   reward=0.174  with ~19K LLM calls
  Log-barrier OMD:    reward=0.179  with ~89K LLM calls

So poly decay is very competitive and much cheaper.

Run variants:
  python experiment_poly_decay.py --llm_type large --C_poly 10 --alpha_decay 1.0
  python experiment_poly_decay.py --llm_type large --C_poly 100 --alpha_decay 1.0
  python experiment_poly_decay.py --llm_type large --C_poly 1 --alpha_decay 0.5
  python experiment_poly_decay.py --llm_type base  --C_poly 10 --alpha_decay 1.0
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


def sm_update(V_inv, x):
    """Sherman-Morrison O(d^2) rank-1 inverse update."""
    Vx = V_inv @ x
    return V_inv - np.outer(Vx, Vx) / (1.0 + x @ Vx)


class EasyAcc:
    def __init__(self): self.n = 0; self.sum = 0
    def __iadd__(self, v): self.n += 1; self.sum += v; return self
    def mean(self): return self.sum / max(self.n, 1)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self): pass
    def __len__(self): return self.Xs.shape[0]
    def __getitem__(self, i):
        return self.Xs[i], self.ys[i], self.pre_texts[i], self.post_texts[i], self.text_entities[i]


def loadMyDataset(threshold):
    with gzip.open(f'mydataset.{threshold}.pickle.gz', 'rb') as f:
        return pickle.load(f)


def load_model_and_tokenizer(model_name, device):
    from transformers import T5Tokenizer, T5ForConditionalGeneration
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


def learnOnline(dataset, batch_size, cuda, seed, llm_type,
                C_poly=10.0, alpha_decay_sched=1.0,
                p_min=0.0, p_max=0.8,
                linucb_alpha=1.0, lam=0.1,
                ucb_alpha_decay=0.9985):
    """
    C_poly          : scale of polynomial decay. Larger = more LLM early on
    alpha_decay_sched : exponent of decay. 1.0 = 1/t, 0.5 = 1/sqrt(t)
    p_min, p_max    : bounds on p_llm at any step
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
    print(f"LLM: flan-t5-{llm_type} | C={C_poly} | alpha_sched={alpha_decay_sched}")
    print(f"p_min={p_min} p_max={p_max} | linucb_alpha={linucb_alpha} | lam={lam}")

    t5_model, t5_tokenizer = load_model_and_tokenizer(f"google/flan-t5-{llm_type}", device)
    entity_model = SentenceTransformer('bert-base-nli-mean-tokens', device=device)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    action_features = entity_embd.cpu().numpy()
    d = action_features.shape[1]

    # Hybrid LinUCB
    V_inv  = (1.0 / lam) * np.eye(d)
    b      = np.zeros(d)

    # Cold LinUCB (standalone baseline)
    V_inv_cold = (1.0 / lam) * np.eye(d)
    b_cold     = np.zeros(d)

    acc_hybrid = EasyAcc()
    acc_llm    = EasyAcc()
    acc_cold   = EasyAcc()

    total_steps      = 0
    total_llm_picks  = 0
    total_cb_picks   = 0
    correct_llm      = 0
    correct_cb       = 0

    history = {
        'batch': [], 'hybrid': [], 'llm': [], 'linucb_cold': [],
        'p_llm_actual': [], 'llm_picks': [], 'cb_picks': [],
        'llm_acc_when_selected': [], 'cb_acc_when_selected': [],
        'config': {
            'C_poly': C_poly, 'alpha_decay_sched': alpha_decay_sched,
            'p_min': p_min, 'p_max': p_max, 'llm_type': llm_type
        }
    }

    print(f"Total batches: {len(loader)}")
    print("=" * 90)

    for bno, (Xs, ys, pre_texts, post_texts, text_entities) in enumerate(loader):

        current_alpha = max(0.1, linucb_alpha * (ucb_alpha_decay ** bno))

        Xs, ys = Xs.to(device), ys.to(device)
        with torch.no_grad():
            llm_labels = language_model_outputs(
                pre_texts, post_texts, t5_tokenizer, t5_model,
                entity_model, entity_embd, num_entities, cos, device
            )

        for i in range(len(Xs)):
            total_steps += 1
            context_x  = Xs[i].cpu().numpy()
            d_ctx      = context_x.shape[0] // 2
            cx         = context_x[:d_ctx]
            true_label = ys[i].item()
            llm_choice = llm_labels[i].item()
            joint      = action_features * cx  # (num_actions, d)

            # Polynomial schedule for p_llm at this step
            p_llm = float(np.clip(C_poly / (total_steps ** alpha_decay_sched), p_min, p_max))
            p_cb  = 1.0 - p_llm

            # -- LLM only baseline --
            acc_llm += 1.0 if llm_choice == true_label else 0.0

            # -- Cold LinUCB (always acts, no IW) --
            theta_cold = V_inv_cold @ b_cold
            Vf_cold    = joint @ V_inv_cold
            ucb_cold   = joint @ theta_cold + current_alpha * np.sqrt(np.maximum(np.sum(Vf_cold * joint, axis=1), 0))
            cold_choice = int(np.argmax(ucb_cold))
            r_cold = 1.0 if cold_choice == true_label else 0.0
            acc_cold += r_cold
            x_cold = joint[cold_choice]
            V_inv_cold = sm_update(V_inv_cold, x_cold)
            b_cold    += r_cold * x_cold

            # -- Poly-decay hybrid --
            use_llm = np.random.rand() < p_llm

            if use_llm:
                action = llm_choice
                iw = 1.0 / p_llm
                total_llm_picks += 1
                if action == true_label:
                    correct_llm += 1
            else:
                theta = V_inv @ b
                Vf    = joint @ V_inv
                ucb   = joint @ theta + current_alpha * np.sqrt(np.maximum(np.sum(Vf * joint, axis=1), 0))
                action = int(np.argmax(ucb))
                iw = 1.0 / p_cb
                total_cb_picks += 1
                if action == true_label:
                    correct_cb += 1

            reward = 1.0 if action == true_label else 0.0
            acc_hybrid += reward

            # Importance-weighted update to LinUCB on EVERY round
            x_chosen = joint[action]
            V_inv = sm_update(V_inv, x_chosen)
            b    += iw * reward * x_chosen

        if bno % 10 == 0:
            p_llm_now = float(np.clip(C_poly / (total_steps ** alpha_decay_sched), p_min, p_max))
            llm_acc_sel = correct_llm / max(total_llm_picks, 1)
            cb_acc_sel  = correct_cb  / max(total_cb_picks,  1)

            history['batch'].append(bno)
            history['hybrid'].append(acc_hybrid.mean())
            history['llm'].append(acc_llm.mean())
            history['linucb_cold'].append(acc_cold.mean())
            history['p_llm_actual'].append(p_llm_now)
            history['llm_picks'].append(total_llm_picks)
            history['cb_picks'].append(total_cb_picks)
            history['llm_acc_when_selected'].append(llm_acc_sel)
            history['cb_acc_when_selected'].append(cb_acc_sel)

            print(
                f"Batch {bno:4d} | "
                f"Hybrid: {acc_hybrid.mean():.4f} | "
                f"LLM: {acc_llm.mean():.4f} | "
                f"Cold LinUCB: {acc_cold.mean():.4f} | "
                f"p_llm={p_llm_now:.3f} | "
                f"LLM acc@sel={llm_acc_sel:.3f}  CB acc@sel={cb_acc_sel:.3f}"
            )

            tag = f"{llm_type}_C{C_poly}_a{alpha_decay_sched}"
            with open(f"results_poly_{tag}.json", "w") as f:
                json.dump(history, f)

        del Xs, ys, pre_texts, post_texts, text_entities, llm_labels
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    print("=" * 90)
    print(f"FINAL | Hybrid: {acc_hybrid.mean():.4f} | LLM: {acc_llm.mean():.4f} | Cold LinUCB: {acc_cold.mean():.4f}")
    print(f"Total steps={total_steps} | LLM picks={total_llm_picks} | CB picks={total_cb_picks}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',              type=int,   default=1)
    parser.add_argument('--llm_type',          type=str,   default='large',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--C_poly',            type=float, default=10.0,
                        help='p_llm = C / t^alpha. Try 1, 10, 100')
    parser.add_argument('--alpha_decay_sched', type=float, default=1.0,
                        help='Decay exponent. 1.0=1/t, 0.5=1/sqrt(t)')
    parser.add_argument('--p_min',             type=float, default=0.0,
                        help='Floor on p_llm. 0 means LLM can go to 0')
    parser.add_argument('--p_max',             type=float, default=0.8,
                        help='Cap on p_llm. 0.8 means CB always gets >= 20%')
    parser.add_argument('--linucb_alpha',      type=float, default=1.0)
    parser.add_argument('--lam',               type=float, default=0.1)
    args = parser.parse_args()

    mydata = loadMyDataset(2000)
    learnOnline(
        mydata, batch_size=128, cuda=True, seed=args.seed,
        llm_type=args.llm_type, C_poly=args.C_poly,
        alpha_decay_sched=args.alpha_decay_sched,
        p_min=args.p_min, p_max=args.p_max,
        linucb_alpha=args.linucb_alpha, lam=args.lam
    )
