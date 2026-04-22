"""Training curves for the 15 new bgrich REINFORCE experiments.

All 15 were run with the same config (α=0.7, lr=0.10, ent=0.05, 80 episodes).
Shows per-experiment curves + averaged mean/std across all 15.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"
NEW_DIR = os.path.join(ANALYSIS, "new_bgrich_28bit")


def main():
    import json as _json
    runs = []
    random_rewards = {}  # name → np.ndarray of rewards
    for name in sorted(os.listdir(NEW_DIR)):
        csv = os.path.join(NEW_DIR, name, "reinforce_log.csv")
        if os.path.isfile(csv):
            runs.append((name, pd.read_csv(csv)))
        rr = os.path.join(NEW_DIR, name, "random_rewards.npy")
        if os.path.isfile(rr):
            random_rewards[name] = np.load(rr)
    print(f"Loaded {len(runs)} new_bgrich runs")
    print(f"Loaded random baselines for {len(random_rewards)}/{len(runs)} experiments")

    if not runs:
        return

    # Optional: all-ones baseline rewards per experiment
    all_ones_path = os.path.join(NEW_DIR, "_all_ones_rewards.json")
    all_ones = {}
    if os.path.isfile(all_ones_path):
        d = _json.load(open(all_ones_path))
        all_ones = {k: v["all_ones_reward"] for k, v in d.items()}
        print(f"Loaded all-ones baselines for {len(all_ones)}/{len(runs)} experiments")

    max_ep = max(len(df) for _, df in runs)
    N = len(runs)

    def stack(col):
        M = np.full((N, max_ep), np.nan)
        for i, (_, df) in enumerate(runs):
            M[i, :len(df)] = df[col].values
        return M

    mean_r = stack("mean_reward")
    best_r = stack("best_reward_ever")
    bg = stack("mean_bg_ssim")
    fg = stack("mean_fg_clip")
    ent = stack("entropy")

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    colors = plt.cm.tab20(np.linspace(0, 1, N))

    def plot_panel(ax, M, title, ylabel, hline=None, hlabel=None,
                   random_curve=None, random_label=None):
        x = np.arange(M.shape[1])
        for i, (name, _) in enumerate(runs):
            vals = M[i]
            m = ~np.isnan(vals)
            ax.plot(x[m], vals[m], color=colors[i], alpha=0.25, linewidth=0.8, label=name)
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        ax.plot(x, mean, color="black", linewidth=2.8, label=f"REINFORCE mean ± std (N={N})")
        ax.fill_between(x, mean - std, mean + std, color="black", alpha=0.15)
        if random_curve is not None:
            rc_x = np.arange(len(random_curve))
            ax.plot(rc_x, random_curve, color="#1f77b4", linewidth=3.0, linestyle="--",
                    label=random_label)
        if hline is not None:
            ax.axhline(hline, color="#e41a1c", linestyle=":", linewidth=2.0, label=hlabel)
        ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(alpha=0.3)

        # Compact legend: only show the main signal lines (mean, random, baseline),
        # skip per-experiment noise lines.
        handles, labels = ax.get_legend_handles_labels()
        keep = [i for i, l in enumerate(labels)
                if "REINFORCE mean" in l or "random" in l or "baseline" in l or "max entropy" in l]
        if keep:
            ax.legend([handles[i] for i in keep], [labels[i] for i in keep],
                      fontsize=9,
                      loc="lower right" if "entropy" not in ylabel else "upper right")

    # Build optional all-ones line overlay
    ao_mean = np.mean([all_ones[n] for n, _ in runs if n in all_ones]) if all_ones else None
    ao_label = (f"all-ones baseline (avg {ao_mean:.3f}, "
                f"N={len([1 for n, _ in runs if n in all_ones])})") if all_ones else None

    # Compute random baseline curves: avg mean_reward per batch + avg running_max per batch
    rand_mean_curve = rand_maxes_curve = None
    if random_rewards:
        batch_size = 8
        max_ep = mean_r.shape[1]
        n_exp_rand = len(random_rewards)
        rand_batch_means = np.full((n_exp_rand, max_ep), np.nan)
        rand_running_max = np.full((n_exp_rand, max_ep), np.nan)
        for i, (name, rewards) in enumerate(sorted(random_rewards.items())):
            # Reshape into (n_batches, batch_size)
            n_full_batches = len(rewards) // batch_size
            for ep in range(min(n_full_batches, max_ep)):
                batch = rewards[ep * batch_size : (ep + 1) * batch_size]
                rand_batch_means[i, ep] = batch.mean()
                rand_running_max[i, ep] = rewards[: (ep + 1) * batch_size].max()
        rand_mean_curve = np.nanmean(rand_batch_means, axis=0)
        rand_maxes_curve = np.nanmean(rand_running_max, axis=0)

    plot_panel(axes[0, 0], mean_r, "mean_reward per episode", "mean_reward",
               hline=ao_mean, hlabel=ao_label,
               random_curve=rand_mean_curve, random_label=(
                   f"random mean (N={len(random_rewards)})" if random_rewards else None))
    plot_panel(axes[0, 1], best_r, "best_reward_ever", "best reward",
               hline=ao_mean, hlabel=ao_label,
               random_curve=rand_maxes_curve, random_label=(
                   f"random running-max (N={len(random_rewards)})" if random_rewards else None))
    plot_panel(axes[1, 0], bg,     "bg_ssim (background preservation)", "bg_ssim")
    plot_panel(axes[1, 1], fg,     "fg_clip (foreground edit direction)", "fg_clip")
    plot_panel(axes[2, 0], ent,    "policy entropy", "H(π)",
               hline=14 * np.log(2), hlabel="max entropy (14 bits)")
    axes[2, 1].axis("off")

    plt.suptitle(f"new_bgrich 28-bit REINFORCE curves (α=0.7, N={N})",
                 fontsize=14, weight="bold")
    plt.tight_layout()
    out = os.path.join(ANALYSIS, "new_bgrich_28bit_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
