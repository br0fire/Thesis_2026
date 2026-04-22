"""Plot reward histograms from saved random baselines + REINFORCE per-episode rewards.

Pure disk-IO, no GPU. Shows the shape of the reward landscape per experiment —
if the histogram is narrow/flat, REINFORCE has little to exploit over random.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"

# Collect data sources
# 1) n_bits=14 sweeps: random rewards in sweep_results.npz, REINFORCE mean_reward in configs/*/reinforce_log.csv
# 2) n_bits=20 experiments: random_rewards.npy, reinforce_log.csv at top level


def gather_nbits14():
    out = {}
    for sweep in sorted(os.listdir(ANALYSIS)):
        if not sweep.startswith("sweep_"):
            continue
        sweep_dir = os.path.join(ANALYSIS, sweep)
        npz_path = os.path.join(sweep_dir, "sweep_results.npz")
        summary_path = os.path.join(sweep_dir, "summary.json")
        if not (os.path.isfile(npz_path) and os.path.isfile(summary_path)):
            continue
        npz = np.load(npz_path)
        summary = json.load(open(summary_path))
        # Collect per-episode means from all 8 configs
        reinforce_rewards = []
        configs_dir = os.path.join(sweep_dir, "configs")
        for cfg in sorted(os.listdir(configs_dir)):
            csv_path = os.path.join(configs_dir, cfg, "reinforce_log.csv")
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                reinforce_rewards.extend(df["mean_reward"].values.tolist())
        out[sweep.replace("sweep_", "")] = {
            "random": np.asarray(npz["random_rewards"]),
            "reinforce_per_episode_mean": np.asarray(reinforce_rewards),
            "all_ones": summary["all_ones_reward"],
        }
    return out


def gather_nbits20():
    out = {}
    root = os.path.join(ANALYSIS, "nbits20")
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        exp_dir = os.path.join(root, name)
        rand_path = os.path.join(exp_dir, "random_rewards.npy")
        csv_path = os.path.join(exp_dir, "reinforce_log.csv")
        if not os.path.isfile(rand_path):
            continue
        data = {"random": np.load(rand_path)}
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            data["reinforce_per_episode_mean"] = df["mean_reward"].values
        out[name] = data
    return out


def plot_set(data_map, out_path, title_suffix):
    n = len(data_map)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows), squeeze=False)

    for i, (name, d) in enumerate(sorted(data_map.items())):
        ax = axes[i // ncols, i % ncols]
        rand = d["random"]
        ax.hist(rand, bins=40, alpha=0.6, color="gray", label=f"random N={len(rand)}", density=True)
        if "reinforce_per_episode_mean" in d:
            rein = d["reinforce_per_episode_mean"]
            ax.hist(rein, bins=40, alpha=0.5, color="steelblue",
                    label=f"REINFORCE (n={len(rein)})", density=True)
        if "all_ones" in d:
            ax.axvline(d["all_ones"], color="black", linestyle="--", linewidth=1,
                       label=f"all-ones ({d['all_ones']:.3f})")
        ax.axvline(rand.max(), color="red", linestyle=":", linewidth=1,
                   label=f"random max ({rand.max():.3f})")
        ax.axvline(rand.mean(), color="gray", linestyle=":", linewidth=1,
                   label=f"random mean ({rand.mean():.3f})")
        ax.set_title(f"{name}\nrand σ={rand.std():.4f}  range=[{rand.min():.3f},{rand.max():.3f}]",
                     fontsize=10)
        ax.set_xlabel("reward"); ax.set_ylabel("density")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    plt.suptitle(f"Reward distributions — {title_suffix}", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    d14 = gather_nbits14()
    d20 = gather_nbits20()

    if d14:
        plot_set(d14, os.path.join(ANALYSIS, "reward_histograms_nbits14.png"), "n_bits=14 sweeps")
    if d20:
        plot_set(d20, os.path.join(ANALYSIS, "reward_histograms_nbits20.png"), "n_bits=20 experiments")

    # Quick stats summary
    print("\n=== SUMMARY ===")
    print(f"{'experiment':<32} {'n_bits':>7} {'rand_σ':>8} {'rand_range':>15} {'rand_max':>8} {'reinforce_max':>14}")
    for name, d in sorted(d14.items()):
        rand = d["random"]; rein = d.get("reinforce_per_episode_mean", np.array([]))
        rein_max = rein.max() if rein.size else float("nan")
        print(f"{name:<32} {'14':>7} {rand.std():>8.4f} [{rand.min():.3f},{rand.max():.3f}] "
              f"{rand.max():>8.4f} {rein_max:>14.4f}")
    for name, d in sorted(d20.items()):
        rand = d["random"]; rein = d.get("reinforce_per_episode_mean", np.array([]))
        rein_max = rein.max() if rein.size else float("nan")
        print(f"{name:<32} {'20':>7} {rand.std():>8.4f} [{rand.min():.3f},{rand.max():.3f}] "
              f"{rand.max():>8.4f} {rein_max:>14.4f}")


if __name__ == "__main__":
    main()
