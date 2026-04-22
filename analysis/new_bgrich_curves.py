"""Training curves for new_bgrich REINFORCE experiments (generic: any n_bits × α).

All baselines (random, all-ones, exhaustive) are recomputed analytically at the
regime's α from saved bg_ssim + fg_clip components, ensuring consistent scale.

Usage:
  python analysis/new_bgrich_curves.py                      # 14-bit α=0.7 (default)
  python analysis/new_bgrich_curves.py --regime 14bit_a05
  python analysis/new_bgrich_curves.py --regime 28bit_a07
  python analysis/new_bgrich_curves.py --regime 28bit_a05
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"

REGIMES = {
    "14bit_a07": {"subdir": "new_bgrich",             "alpha": 0.7, "n_bits": 14,
                  "title": "14-bit, α=0.7",  "exh_dir": "exhaustive"},
    "14bit_a05": {"subdir": "new_bgrich_alpha05",     "alpha": 0.5, "n_bits": 14,
                  "title": "14-bit, α=0.5",  "exh_dir": "exhaustive"},
    "28bit_a07": {"subdir": "new_bgrich_28bit",       "alpha": 0.7, "n_bits": 28,
                  "title": "28-bit, α=0.7",  "exh_dir": None},
    "28bit_a05": {"subdir": "new_bgrich_28bit_alpha05", "alpha": 0.5, "n_bits": 28,
                  "title": "28-bit, α=0.5",  "exh_dir": None},
}


def _reward_from_components(bg_ssim, fg_clip, alpha):
    fg_sig = 1.0 / (1.0 + np.exp(-fg_clip * 10.0))
    return np.clip(bg_ssim, 1e-6, None) ** alpha * np.clip(fg_sig, 1e-6, None) ** (1 - alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="14bit_a07", choices=list(REGIMES.keys()))
    args = ap.parse_args()
    cfg = REGIMES[args.regime]
    alpha = cfg["alpha"]
    n_bits = cfg["n_bits"]
    new_dir = os.path.join(ANALYSIS, cfg["subdir"])
    exh_dir = os.path.join(ANALYSIS, cfg["exh_dir"]) if cfg["exh_dir"] else None
    out_path = os.path.join(ANALYSIS, f"{cfg['subdir']}_training_curves.png")

    runs = []
    for name in sorted(os.listdir(new_dir)):
        csv = os.path.join(new_dir, name, "reinforce_log.csv")
        if os.path.isfile(csv):
            runs.append((name, pd.read_csv(csv)))
    print(f"Regime {args.regime}: {len(runs)} runs from {cfg['subdir']}")
    if not runs:
        return

    # Random baseline @α. Prefer cleanest source:
    #  1) α=0.5: use original exhaustive_rewards.npy (raw float, no JPG quantization)
    #  2) otherwise: recompute from exhaustive_bg_ssim + fg_clip (JPG-decoded)
    random_rewards = {}
    random_src = "?"
    rng = np.random.default_rng(42)
    if exh_dir is not None:
        for name, _ in runs:
            orig_p   = os.path.join(exh_dir, name, "exhaustive_rewards.npy")  # raw-float, α=0.5
            bg_p = os.path.join(exh_dir, name, "exhaustive_bg_ssim.npy")
            fg_p = os.path.join(exh_dir, name, "exhaustive_fg_clip.npy")
            if abs(alpha - 0.5) < 1e-9 and os.path.isfile(orig_p):
                rewards_all = np.load(orig_p)
                random_src  = "from exhaustive raw-float α=0.5"
            elif os.path.isfile(bg_p) and os.path.isfile(fg_p):
                bg = np.load(bg_p); fg = np.load(fg_p)
                rewards_all = _reward_from_components(bg, fg, alpha)
                random_src  = f"from exhaustive bg/fg (JPG) at α={alpha}"
            else:
                continue
            idx = rng.choice(len(rewards_all), size=min(640, len(rewards_all)), replace=False)
            random_rewards[name] = rewards_all[idx]
    if not random_rewards:
        # Fall back to stored random
        for name, _ in runs:
            # Check if per-sample bg/fg saved
            bg_p = os.path.join(new_dir, name, "random_bg_ssim.npy")
            fg_p = os.path.join(new_dir, name, "random_fg_clip.npy")
            rr_p = os.path.join(new_dir, name, "random_rewards.npy")
            if os.path.isfile(bg_p) and os.path.isfile(fg_p):
                random_rewards[name] = _reward_from_components(np.load(bg_p), np.load(fg_p), alpha)
                random_src = f"from saved random bg/fg at α={alpha}"
            elif os.path.isfile(rr_p):
                random_rewards[name] = np.load(rr_p)
                random_src = "from random_rewards.npy (α=0.5)"
    print(f"Random baseline: {len(random_rewards)} exps ({random_src})")

    # All-ones @α: from stored bg/fg
    all_ones = {}
    aop = os.path.join(new_dir, "_all_ones_rewards.json")
    if os.path.isfile(aop):
        d = json.load(open(aop))
        for k, v in d.items():
            all_ones[k] = float(_reward_from_components(np.array([v["bg_ssim"]]),
                                                         np.array([v["fg_clip"]]), alpha)[0])
        print(f"All-ones baseline (α={alpha}): {len(all_ones)} exps")

    # Exhaustive ceiling @α. For α=0.5 prefer raw-float original; else derive from bg/fg.
    exh_max = None
    exh_max_label_suffix = ""
    exh_src = ""
    exh14_dir = os.path.join(ANALYSIS, "exhaustive")
    maxes = []
    for name, _ in runs:
        orig_p = os.path.join(exh14_dir, name, "exhaustive_rewards.npy")
        bg_p = os.path.join(exh14_dir, name, "exhaustive_bg_ssim.npy")
        fg_p = os.path.join(exh14_dir, name, "exhaustive_fg_clip.npy")
        if abs(alpha - 0.5) < 1e-9 and os.path.isfile(orig_p):
            maxes.append(float(np.load(orig_p).max()))
            exh_src = "raw float"
        elif os.path.isfile(bg_p) and os.path.isfile(fg_p):
            r = _reward_from_components(np.load(bg_p), np.load(fg_p), alpha)
            maxes.append(float(r.max()))
            exh_src = "derived from JPG bg/fg"
    if maxes:
        exh_max = float(np.mean(maxes))
        exh_max_label_suffix = "" if n_bits == 14 else " [14-bit reference]"
        print(f"Exhaustive α={alpha} ceiling: {exh_max:.4f} (N={len(maxes)}, {exh_src}){exh_max_label_suffix}")

    max_ep = max(len(df) for _, df in runs)
    N = len(runs)

    def stack(col):
        M = np.full((N, max_ep), np.nan)
        for i, (_, df) in enumerate(runs):
            M[i, :len(df)] = df[col].values
        return M

    mean_r = stack("mean_reward")
    best_r = stack("best_reward_ever")
    bg_stk = stack("mean_bg_ssim")
    fg_stk = stack("mean_fg_clip")
    ent_stk = stack("entropy")

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    colors = plt.cm.tab20(np.linspace(0, 1, N))

    def plot_panel(ax, M, title, ylabel, hline=None, hlabel=None,
                   random_curve=None, random_label=None, exh_line=None, exh_label=None):
        x = np.arange(M.shape[1])
        for i, (name, _) in enumerate(runs):
            vals = M[i]; m = ~np.isnan(vals)
            ax.plot(x[m], vals[m], color=colors[i], alpha=0.25, linewidth=0.8, label=name)
        mean = np.nanmean(M, axis=0); std = np.nanstd(M, axis=0)
        ax.plot(x, mean, color="black", linewidth=2.8, label=f"REINFORCE mean ± std (N={N})")
        ax.fill_between(x, mean - std, mean + std, color="black", alpha=0.15)
        if random_curve is not None:
            rc_x = np.arange(len(random_curve))
            ax.plot(rc_x, random_curve, color="#1f77b4", linewidth=3.0, linestyle="--",
                    label=random_label)
        if hline is not None:
            ax.axhline(hline, color="#e41a1c", linestyle=":", linewidth=2.0, label=hlabel)
        if exh_line is not None:
            ax.axhline(exh_line, color="#2ca02c", linestyle="-", linewidth=2.0, label=exh_label)
        ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        keep = [i for i, l in enumerate(labels)
                if "REINFORCE mean" in l or "random" in l or "baseline" in l
                   or "all-ones" in l.lower() or "max entropy" in l or "exhaustive" in l.lower()]
        if keep:
            ax.legend([handles[i] for i in keep], [labels[i] for i in keep],
                      fontsize=9, loc="lower right" if "entropy" not in ylabel else "upper right")

    # Build random curves (mean + running_max per batch)
    rand_mean_curve = rand_max_curve = None
    if random_rewards:
        batch_size = 8
        A = np.full((len(random_rewards), max_ep), np.nan)
        B = np.full((len(random_rewards), max_ep), np.nan)
        for i, (name, rew) in enumerate(sorted(random_rewards.items())):
            n_full = len(rew) // batch_size
            for ep in range(min(n_full, max_ep)):
                A[i, ep] = rew[ep * batch_size : (ep + 1) * batch_size].mean()
                B[i, ep] = rew[: (ep + 1) * batch_size].max()
        rand_mean_curve = np.nanmean(A, axis=0)
        rand_max_curve = np.nanmean(B, axis=0)

    ao_mean = np.mean(list(all_ones.values())) if all_ones else None
    ao_label = f"all-ones α={alpha} ({ao_mean:.3f}, N={len(all_ones)})" if all_ones else None
    exh_label = (f"exhaustive α={alpha} ceiling ({exh_max:.3f}){exh_max_label_suffix}"
                 if exh_max is not None else None)

    plot_panel(axes[0, 0], mean_r, "mean_reward per episode", "mean_reward",
               hline=ao_mean, hlabel=ao_label,
               random_curve=rand_mean_curve,
               random_label=f"random mean α={alpha} ({random_src})" if rand_mean_curve is not None else None)
    plot_panel(axes[0, 1], best_r, "best_reward_ever", "best reward",
               hline=ao_mean, hlabel=ao_label,
               random_curve=rand_max_curve,
               random_label=f"random running-max α={alpha} ({random_src})" if rand_max_curve is not None else None,
               exh_line=exh_max, exh_label=exh_label)
    plot_panel(axes[1, 0], bg_stk, "bg_ssim (background preservation)", "bg_ssim")
    plot_panel(axes[1, 1], fg_stk, "fg_clip (foreground edit direction)", "fg_clip")
    plot_panel(axes[2, 0], ent_stk, "policy entropy", "H(π)",
               hline=n_bits * np.log(2), hlabel=f"max entropy ({n_bits} bits)")
    axes[2, 1].axis("off")

    plt.suptitle(f"new_bgrich REINFORCE curves ({cfg['title']}, N={N})",
                 fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
