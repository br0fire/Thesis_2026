"""Reward landscape histograms at α=0.5: shape of exhaustive reward distributions.

For each of 15 new_bgrich experiments, shows:
  - Histogram of 16384 exhaustive rewards (raw float, α=0.5)
  - REINFORCE best vs random-640 best vs all-ones marked as vertical lines
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"


def main():
    exh_dir = os.path.join(ANALYSIS, "exhaustive")
    rein_dir = os.path.join(ANALYSIS, "new_bgrich_alpha05")
    exps = sorted(os.listdir(exh_dir))
    exps = [e for e in exps if os.path.isfile(os.path.join(exh_dir, e, "exhaustive_rewards.npy"))]
    print(f"Using {len(exps)} exhaustive α=0.5 experiments")

    # Load all-ones
    ao_path = os.path.join(rein_dir, "_all_ones_rewards.json")
    all_ones = {}
    if os.path.isfile(ao_path):
        d = json.load(open(ao_path))
        for k, v in d.items():
            bg = v["bg_ssim"]; fg = v["fg_clip"]
            fg_sig = 1.0 / (1.0 + np.exp(-fg * 10.0))
            all_ones[k] = float(np.clip(bg, 1e-6, None)**0.5 * np.clip(fg_sig, 1e-6, None)**0.5)

    # Per-experiment panel grid
    ncols = 4
    nrows = (len(exps) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), squeeze=False)
    rng = np.random.default_rng(42)

    for i, name in enumerate(exps):
        ax = axes[i // ncols, i % ncols]
        rewards = np.load(os.path.join(exh_dir, name, "exhaustive_rewards.npy"))
        ax.hist(rewards, bins=60, color="#888", alpha=0.7, density=True)
        ax.axvline(rewards.max(), color="#2ca02c", linewidth=2,
                   label=f"max={rewards.max():.3f}")
        ax.axvline(rewards.mean(), color="gray", linewidth=1, linestyle=":",
                   label=f"mean={rewards.mean():.3f}")

        # Random-640 best
        rand_idx = rng.choice(len(rewards), size=640, replace=False)
        rand_best = rewards[rand_idx].max()
        ax.axvline(rand_best, color="#1f77b4", linewidth=1.5, linestyle="--",
                   label=f"random-640 best={rand_best:.3f}")

        # REINFORCE best
        csv = os.path.join(rein_dir, name, "reinforce_log.csv")
        if os.path.isfile(csv):
            df = pd.read_csv(csv)
            r_best = float(df["best_reward_ever"].iloc[-1])
            ax.axvline(r_best, color="black", linewidth=1.8,
                       label=f"REINFORCE best={r_best:.3f}")

        if name in all_ones:
            ax.axvline(all_ones[name], color="#e41a1c", linewidth=1.5, linestyle=":",
                       label=f"all-ones={all_ones[name]:.3f}")

        ax.set_title(f"{name}\nσ={rewards.std():.3f}  range=[{rewards.min():.2f}, {rewards.max():.2f}]",
                     fontsize=9)
        ax.set_xlabel("reward"); ax.set_ylabel("density")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

    # Turn off unused axes
    for j in range(len(exps), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    plt.suptitle(f"Reward landscape α=0.5 (exhaustive 2^14 = 16384 masks, raw float)",
                 fontsize=14, weight="bold")
    plt.tight_layout()
    out = os.path.join(ANALYSIS, "alpha05_histograms.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Print summary statistics
    print("\n=== Landscape shape stats ===")
    print(f"{'experiment':<32} {'max':>6} {'mean':>6} {'σ':>6} {'p99':>6} {'p95':>6} {'p90':>6}")
    for name in exps:
        r = np.load(os.path.join(exh_dir, name, "exhaustive_rewards.npy"))
        p99 = np.percentile(r, 99)
        p95 = np.percentile(r, 95)
        p90 = np.percentile(r, 90)
        print(f"{name:<32} {r.max():>6.3f} {r.mean():>6.3f} {r.std():>6.3f} "
              f"{p99:>6.3f} {p95:>6.3f} {p90:>6.3f}")


if __name__ == "__main__":
    main()
