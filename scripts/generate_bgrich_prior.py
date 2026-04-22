"""Build bgrich prior from clean_v1 exhaustive data: average mask-bit frequency
in top-K masks across all 8 bgrich experiments.

Output: a single (n_bits,) float32 array of per-bit P(bit=1).
"""
import argparse
import numpy as np
import os

ROOT = "/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1/exhaustive"


def int_to_bits(n, n_bits):
    bits = np.zeros(n_bits, dtype=np.int32)
    for i in range(n_bits):
        if n & (1 << (n_bits - 1 - i)):
            bits[i] = 1
    return bits


def reward_at(bg, fg, alpha=0.5):
    fg_sig = 1.0 / (1.0 + np.exp(-fg * 10.0))
    return np.maximum(bg, 1e-6) ** alpha * np.maximum(fg_sig, 1e-6) ** (1 - alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1/bgrich_prior.npy")
    ap.add_argument("--top_k", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n_bits", type=int, default=14)
    args = ap.parse_args()

    priors = []
    for name in sorted(os.listdir(ROOT)):
        bg_p = os.path.join(ROOT, name, "bg_ssim.npy")
        fg_p = os.path.join(ROOT, name, "fg_clip.npy")
        if not (os.path.isfile(bg_p) and os.path.isfile(fg_p)):
            continue
        bg = np.load(bg_p); fg = np.load(fg_p)
        if not np.isfinite(bg).all():
            continue
        R = reward_at(bg, fg, args.alpha)
        top = np.argsort(-R)[:args.top_k]
        bits = np.stack([int_to_bits(int(i), args.n_bits) for i in top])
        per_exp_prob = bits.mean(axis=0)
        priors.append(per_exp_prob)
        print(f"  {name:<32} top-{args.top_k} mean: [{per_exp_prob.min():.2f}, {per_exp_prob.max():.2f}]")

    final_prior = np.mean(priors, axis=0).astype(np.float32)
    print(f"\nFinal bgrich prior (mean over {len(priors)} experiments):")
    for i, p in enumerate(final_prior):
        print(f"  bit {i:2d}: {p:.3f}")

    np.save(args.out, final_prior)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
