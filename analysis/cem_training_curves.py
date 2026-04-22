"""Aggregated CEM training curves across all experiments.

Reads cem_log.csv from every cem/<name>_budget{40,80,160}/ dir and builds a
grouped plot: best_reward_ever vs iteration, averaged across experiments with
std shading, one line per budget.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"
CEM_DIR = os.path.join(ANALYSIS, "cem")


def load_cem_runs():
    """Return dict: budget → list of (name, df) with per-iteration log."""
    by_budget = {40: [], 80: [], 160: []}
    for sub in sorted(os.listdir(CEM_DIR)):
        for b in by_budget:
            suffix = f"_budget{b}"
            if sub.endswith(suffix):
                name = sub[:-len(suffix)]
                csv = os.path.join(CEM_DIR, sub, "cem_log.csv")
                if os.path.isfile(csv):
                    df = pd.read_csv(csv)
                    by_budget[b].append((name, df))
    return by_budget


def plot_curves(runs, out_path):
    budgets = [40, 80, 160]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {40: "tab:blue", 80: "tab:orange", 160: "tab:green"}

    # Panel 1: per-iteration mean_reward (averaged across experiments)
    ax = axes[0, 0]
    for b in budgets:
        if not runs[b]:
            continue
        iters = max(len(df) for _, df in runs[b])
        M = np.full((len(runs[b]), iters), np.nan)
        for i, (_, df) in enumerate(runs[b]):
            M[i, :len(df)] = df["mean_reward"].values
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        x = np.arange(iters)
        ax.plot(x, mean, color=colors[b], linewidth=2, label=f"budget={b} (N={len(runs[b])})")
        ax.fill_between(x, mean - std, mean + std, color=colors[b], alpha=0.15)
    ax.set_xlabel("Iteration"); ax.set_ylabel("mean_reward per batch")
    ax.set_title("CEM: mean reward per iteration (avg over experiments)")
    ax.grid(alpha=0.3); ax.legend()

    # Panel 2: best_reward_ever per iteration
    ax = axes[0, 1]
    for b in budgets:
        if not runs[b]:
            continue
        iters = max(len(df) for _, df in runs[b])
        M = np.full((len(runs[b]), iters), np.nan)
        for i, (_, df) in enumerate(runs[b]):
            M[i, :len(df)] = df["best_reward_ever"].values
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        x = np.arange(iters)
        ax.plot(x, mean, color=colors[b], linewidth=2, label=f"budget={b}")
        ax.fill_between(x, mean - std, mean + std, color=colors[b], alpha=0.15)
    # Exhaustive ceiling (α=0.5, raw float from original run)
    exh_dir = os.path.join(os.path.dirname(CEM_DIR), "exhaustive")
    if os.path.isdir(exh_dir):
        maxes = []
        for name in os.listdir(exh_dir):
            orig_p = os.path.join(exh_dir, name, "exhaustive_rewards.npy")
            if os.path.isfile(orig_p):
                maxes.append(float(np.load(orig_p).max()))
        if maxes:
            exh_mean = np.mean(maxes)
            ax.axhline(exh_mean, color="#2ca02c", linewidth=2.2, linestyle="-",
                       label=f"Exhaustive α=0.5 (raw float, {exh_mean:.3f}, N={len(maxes)})")

    ax.set_xlabel("Iteration"); ax.set_ylabel("best_reward_ever")
    ax.set_title("CEM: best-so-far reward (avg over experiments)")
    ax.grid(alpha=0.3); ax.legend()

    # Panel 3: elite_mean_reward
    ax = axes[1, 0]
    for b in budgets:
        if not runs[b]:
            continue
        iters = max(len(df) for _, df in runs[b])
        M = np.full((len(runs[b]), iters), np.nan)
        for i, (_, df) in enumerate(runs[b]):
            M[i, :len(df)] = df["elite_mean_reward"].values
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        x = np.arange(iters)
        ax.plot(x, mean, color=colors[b], linewidth=2, label=f"budget={b}")
        ax.fill_between(x, mean - std, mean + std, color=colors[b], alpha=0.15)
    ax.set_xlabel("Iteration"); ax.set_ylabel("elite_mean_reward")
    ax.set_title("CEM: elite (top-25%) mean reward")
    ax.grid(alpha=0.3); ax.legend()

    # Panel 4: entropy of probs (collapse rate)
    ax = axes[1, 1]
    for b in budgets:
        if not runs[b]:
            continue
        iters = max(len(df) for _, df in runs[b])
        M = np.full((len(runs[b]), iters), np.nan)
        for i, (_, df) in enumerate(runs[b]):
            M[i, :len(df)] = df["entropy"].values
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        x = np.arange(iters)
        ax.plot(x, mean, color=colors[b], linewidth=2, label=f"budget={b}")
        ax.fill_between(x, mean - std, mean + std, color=colors[b], alpha=0.15)
    ax.axhline(14 * np.log(2), color="red", linestyle=":", linewidth=1, label="max (14 bits)")
    ax.set_xlabel("Iteration"); ax.set_ylabel("policy entropy H(p)")
    ax.set_title("CEM: policy entropy (lower = more committed)")
    ax.grid(alpha=0.3); ax.legend()

    plt.suptitle(f"CEM training curves (avg over experiments)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    runs = load_cem_runs()
    for b, rs in runs.items():
        print(f"  budget={b}: {len(rs)} experiments")
    out = os.path.join(ANALYSIS, "cem_training_curves.png")
    plot_curves(runs, out)


if __name__ == "__main__":
    main()
