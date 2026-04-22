"""Train a small amortized policy: SigLIP prompt embeddings → 14 Bernoulli logits.

Collects (prompts, learned_probs) pairs from all completed 14-bit REINFORCE runs,
embeds prompts with SigLIP, trains an MLP with leave-one-out CV, and saves
predicted probs per test experiment so eval_amortized.py can generate images.

Also trains a Ridge linear baseline for comparison.
"""
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
ANALYSIS = os.path.join(PROJECT_ROOT, "analysis/reinforce_analysis")
OUT_DIR = os.path.join(ANALYSIS, "amortized")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_prompts(p):
    out = {}
    with open(p) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def collect_14bit_experiments():
    """Find all 14-bit REINFORCE runs, dedup by experiment name, prefer sweep's
    alpha_high config → newer v-series → older."""
    found = {}  # name → (priority, pt_path, prompts_path)

    # Priority 1: sweep alpha_high configs (best known hyperparams)
    for sweep in sorted(os.listdir(ANALYSIS)):
        if not sweep.startswith("sweep_"):
            continue
        name = sweep[len("sweep_"):]
        cfg_dir = os.path.join(ANALYSIS, sweep, "configs", "alpha_high")
        pt = os.path.join(cfg_dir, "reinforce_result.pt")
        pp = os.path.join(cfg_dir, "prompts.txt")
        if os.path.isfile(pt) and os.path.isfile(pp):
            found[name] = (1, pt, pp)

    # Priority 2: new_bgrich/<name>/
    new_bgrich_root = os.path.join(ANALYSIS, "new_bgrich")
    if os.path.isdir(new_bgrich_root):
        for name in sorted(os.listdir(new_bgrich_root)):
            if name in found:
                continue
            pt = os.path.join(new_bgrich_root, name, "reinforce_result.pt")
            pp = os.path.join(new_bgrich_root, name, "prompts.txt")
            if os.path.isfile(pt) and os.path.isfile(pp):
                found[name] = (2, pt, pp)

    # Priority 3: latest v-series (v6 > v5 > v4 > v3)
    for version in sorted(os.listdir(ANALYSIS), reverse=True):
        if not re.match(r"v\d+[a-z]*$", version):
            continue
        exp_root = os.path.join(ANALYSIS, version, "experiments")
        if not os.path.isdir(exp_root):
            continue
        for exp_name in sorted(os.listdir(exp_root)):
            name = re.sub(r"^reinforce_", "", exp_name)
            name = re.sub(r"_v\d+[a-z]*$", "", name)
            if name in found:
                continue
            pt = os.path.join(exp_root, exp_name, "reinforce_result.pt")
            pp = os.path.join(exp_root, exp_name, "prompts.txt")
            if os.path.isfile(pt) and os.path.isfile(pp):
                found[name] = (3, pt, pp)

    # Load + filter 14-bit
    records = []
    for name, (_, pt, pp) in sorted(found.items()):
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
        probs = ckpt["probs"]
        if torch.is_tensor(probs):
            probs = probs.float().numpy()
        else:
            probs = np.asarray(probs, dtype=np.float32)
        if probs.shape[0] != 14:
            print(f"  skip {name}: n_bits={probs.shape[0]}")
            continue
        prompts = parse_prompts(pp)
        records.append({
            "name": name,
            "probs": probs,
            "source": prompts["source"],
            "target": prompts["target"],
            "seg": prompts.get("seg", ""),
            "best_reward": float(ckpt.get("best_reward", np.nan)),
        })
    return records


def embed_prompts(texts, tokenizer, text_model, device, batch_size=8):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding="max_length", truncation=True,
                           max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            out = text_model(**inputs)
        pooled = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None \
                 else out.last_hidden_state.mean(dim=1)
        embs.append(pooled.cpu().float().numpy())
    return np.concatenate(embs, axis=0)


def collect_from_dir(root, n_bits_expected):
    """Collect (name, probs, prompts) from each subdir containing reinforce_result.pt."""
    records = []
    if not os.path.isdir(root):
        return records
    for name in sorted(os.listdir(root)):
        pt = os.path.join(root, name, "reinforce_result.pt")
        pp = os.path.join(root, name, "prompts.txt")
        if not (os.path.isfile(pt) and os.path.isfile(pp)):
            continue
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
        probs = ckpt["probs"]
        if torch.is_tensor(probs):
            probs = probs.float().numpy()
        else:
            probs = np.asarray(probs, dtype=np.float32)
        if probs.shape[0] != n_bits_expected:
            continue
        prompts = parse_prompts(pp)
        records.append({
            "name": name,
            "probs": probs,
            "source": prompts["source"],
            "target": prompts["target"],
            "seg": prompts.get("seg", ""),
            "best_reward": float(ckpt.get("best_reward", np.nan)),
        })
    return records


class PolicyMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, n_bits=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_bits),
        )

    def forward(self, x):
        return self.net(x)  # logits


def train_mlp(X_train, y_train, epochs=300, lr=1e-3, device="cuda", track_loss=False,
              X_val=None, y_val=None):
    X = torch.from_numpy(X_train).float().to(device)
    y = torch.from_numpy(y_train).float().to(device)
    model = PolicyMLP(X.shape[1], n_bits=y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    history = {"train": [], "val": []} if track_loss else None
    if X_val is not None:
        Xv = torch.from_numpy(X_val).float().to(device)
        yv = torch.from_numpy(y_val).float().to(device)
    model.train()
    for ep in range(epochs):
        logits = model(X)
        loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if track_loss:
            history["train"].append(loss.item())
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    vloss = loss_fn(model(Xv), yv).item()
                history["val"].append(vloss)
                model.train()
    model.eval()
    return model, loss.item(), history


def predict_mlp(model, X_test, device="cuda"):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float().to(device))
    return torch.sigmoid(logits).cpu().numpy()


def ridge_regression(X_train, y_train, alpha=1.0):
    """Closed-form ridge: w = (X^T X + αI)^-1 X^T y. Fits sigmoid-space logits."""
    # Transform probs to logits (with clipping)
    eps = 1e-3
    y_clipped = np.clip(y_train, eps, 1 - eps)
    y_logits = np.log(y_clipped / (1 - y_clipped))
    # Add bias column
    X_b = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)
    n_feat = X_b.shape[1]
    I = np.eye(n_feat)
    I[-1, -1] = 0  # don't regularize bias
    W = np.linalg.solve(X_b.T @ X_b + alpha * I, X_b.T @ y_logits)
    return W


def predict_ridge(W, X_test):
    X_b = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)
    logits = X_b @ W
    return 1.0 / (1.0 + np.exp(-logits))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", default=None,
                    help="Single directory with REINFORCE results (homogeneous α, n_bits). "
                         "If not set, uses legacy multi-source (mixed α).")
    ap.add_argument("--n_bits", type=int, default=14)
    ap.add_argument("--out_suffix", default="",
                    help="Suffix for output dir, e.g. '_alpha05_14bit'")
    ap.add_argument("--gpu", type=int, default=6)
    args = ap.parse_args()

    global OUT_DIR
    OUT_DIR = os.path.join(ANALYSIS, f"amortized{args.out_suffix}")
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    if args.source_dir:
        print(f"=== Collecting {args.n_bits}-bit from {args.source_dir} ===")
        records = collect_from_dir(args.source_dir, args.n_bits)
    else:
        print("=== Collecting 14-bit experiments (legacy mixed-α) ===")
        records = collect_14bit_experiments()
    print(f"Found {len(records)} experiments")
    for r in records:
        print(f"  {r['name']:<28} best_R={r['best_reward']:.4f}")

    if len(records) < 4:
        print("Not enough data for training"); return

    # Load SigLIP text encoder
    print("\n=== Loading SigLIP text encoder ===")
    from transformers import AutoTokenizer, AutoModel
    model_id = "google/siglip2-so400m-patch14-384"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text_model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32).to(device).eval()
    # Only keep text encoder
    if hasattr(text_model, "text_model"):
        text_model = text_model.text_model

    # Embed prompts
    print("\n=== Embedding prompts ===")
    src_texts = [r["source"] for r in records]
    tgt_texts = [r["target"] for r in records]
    seg_texts = [r["seg"] if r["seg"] else r["source"][:50] for r in records]

    src_emb = embed_prompts(src_texts, tokenizer, text_model, device)
    tgt_emb = embed_prompts(tgt_texts, tokenizer, text_model, device)
    seg_emb = embed_prompts(seg_texts, tokenizer, text_model, device)

    # Features: concat [src, tgt, seg, tgt-src]
    delta = tgt_emb - src_emb
    X = np.concatenate([src_emb, tgt_emb, seg_emb, delta], axis=1)
    y = np.stack([r["probs"] for r in records])
    print(f"X shape: {X.shape}  y shape: {y.shape}")

    # Leave-one-out
    N = len(records)
    preds_mlp = np.zeros_like(y)
    preds_ridge = np.zeros_like(y)
    all_histories = []

    for i in range(N):
        train_idx = [j for j in range(N) if j != i]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[i:i + 1]
        y_te = y[i:i + 1]

        model, loss, history = train_mlp(X_tr, y_tr, epochs=300, device=device,
                                         track_loss=True, X_val=X_te, y_val=y_te)
        all_histories.append(history)
        preds_mlp[i] = predict_mlp(model, X_te, device=device)[0]

        W = ridge_regression(X_tr, y_tr, alpha=10.0)
        preds_ridge[i] = predict_ridge(W, X_te)[0]

        diff_mlp = np.abs(preds_mlp[i] - y[i]).mean()
        diff_ridge = np.abs(preds_ridge[i] - y[i]).mean()
        greedy_mlp = (preds_mlp[i] > 0.5).astype(int)
        greedy_true = (y[i] > 0.5).astype(int)
        bit_agree = (greedy_mlp == greedy_true).sum()
        print(f"  LOO[{i:2d}] {records[i]['name']:<28}  "
              f"MAE_mlp={diff_mlp:.3f}  MAE_ridge={diff_ridge:.3f}  "
              f"bits_agree={bit_agree}/14  train_loss={loss:.4f}")

    # Save predictions + metadata
    out = {}
    for i, r in enumerate(records):
        out[r["name"]] = {
            "probs_true": r["probs"].tolist(),
            "probs_mlp": preds_mlp[i].tolist(),
            "probs_ridge": preds_ridge[i].tolist(),
            "probs_popmean": y.mean(axis=0).tolist(),
            "best_reward_true": r["best_reward"],
        }
    with open(os.path.join(OUT_DIR, "predictions.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Training data for reuse
    np.savez(
        os.path.join(OUT_DIR, "training_data.npz"),
        X=X, y=y, names=np.array([r["name"] for r in records]),
    )
    print(f"\nSaved: {OUT_DIR}/predictions.json")
    print(f"Saved: {OUT_DIR}/training_data.npz")

    # Summary
    mlp_mae = np.abs(preds_mlp - y).mean()
    ridge_mae = np.abs(preds_ridge - y).mean()
    popmean = y.mean(axis=0, keepdims=True)
    popmean_mae = np.abs(popmean - y).mean()
    print("\n=== MAE on learned probs (lower = better) ===")
    print(f"  Population mean (trivial): {popmean_mae:.3f}")
    print(f"  Ridge regression:          {ridge_mae:.3f}")
    print(f"  MLP:                       {mlp_mae:.3f}")

    # Plot MLP training curves (avg train + val loss across LOO folds)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    T = np.array([h["train"] for h in all_histories])  # (N, epochs)
    V = np.array([h["val"] for h in all_histories])
    epochs = np.arange(T.shape[1])
    ax.plot(epochs, T.mean(axis=0), color="tab:blue", label=f"train loss (avg over {N} folds)", linewidth=2)
    ax.fill_between(epochs, T.mean(axis=0) - T.std(axis=0), T.mean(axis=0) + T.std(axis=0),
                    color="tab:blue", alpha=0.15)
    ax.plot(epochs, V.mean(axis=0), color="tab:red", label=f"held-out loss (avg over {N} folds)",
            linewidth=2)
    ax.fill_between(epochs, V.mean(axis=0) - V.std(axis=0), V.mean(axis=0) + V.std(axis=0),
                    color="tab:red", alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title(f"MLP policy — Leave-one-out training (N={N} experiments)")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mlp_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR}/mlp_training_curves.png")


if __name__ == "__main__":
    main()
