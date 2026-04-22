"""Regenerate sweep_curves.png for all completed sweeps from saved data (no GPU)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_sweep_aggregate import CONFIG_ORDER, plot_sweep

ANALYSIS = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"

for sweep_name in sorted(os.listdir(ANALYSIS)):
    if not sweep_name.startswith("sweep_"):
        continue
    sweep_dir = os.path.join(ANALYSIS, sweep_name)
    summary_path = os.path.join(sweep_dir, "summary.json")
    npz_path = os.path.join(sweep_dir, "sweep_results.npz")
    if not os.path.isfile(summary_path) or not os.path.isfile(npz_path):
        continue

    print(f"\n=== {sweep_name} ===")
    summary = json.load(open(summary_path))
    npz = np.load(npz_path)

    results = []
    for name in CONFIG_ORDER:
        csv_path = os.path.join(sweep_dir, "configs", name, "reinforce_log.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        results.append({
            "name": name,
            "mean_rewards": df["mean_reward"].values,
            "best_rewards_ever": df["best_reward_ever"].values,
            "fg_clips": df["mean_fg_clip"].values,
            "bg_ssims": df["mean_bg_ssim"].values,
            "entropies": df["entropy"].values,
        })

    ones_res = {
        "reward": summary["all_ones_reward"],
        "bg_ssim": summary.get("all_ones_bg", 0),
        "fg_clip": summary.get("all_ones_fg", 0),
    }
    random_res = {
        "rewards": npz["random_rewards"],
        "running_max": npz["random_running_max"],
    }

    experiment = sweep_name.replace("sweep_", "")
    title = f"Parallel sweep: {experiment}"
    out_path = os.path.join(sweep_dir, "sweep_curves.png")
    plot_sweep(results, random_res, ones_res, out_path, title, batch_size=8)
    print(f"  → {out_path}")
