"""Build the one-chart summary: reward vs image budget for each method.

Collects data per experiment, interpolates to common budget points, averages
across experiments with std shading. Shows the sample-efficiency story:
REINFORCE vs CEM vs random vs amortized single-shot.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ANALYSIS = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"
BUDGETS = [1, 8, 40, 80, 160, 320, 640]


def reinforce_best_at_budget(csv_path, batch_size, budget):
    """Return best_reward_ever after `budget` images have been generated."""
    df = pd.read_csv(csv_path)
    ep_needed = budget // batch_size
    if ep_needed == 0:
        return np.nan
    if ep_needed > len(df):
        ep_needed = len(df)
    return float(df["best_reward_ever"].iloc[ep_needed - 1])


def cem_best(cem_dir):
    """Return best reward from CEM run."""
    pt = os.path.join(cem_dir, "cem_result.pt")
    if not os.path.isfile(pt):
        csv = os.path.join(cem_dir, "cem_log.csv")
        if os.path.isfile(csv):
            df = pd.read_csv(csv)
            return float(df["best_reward_ever"].iloc[-1])
        return np.nan
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    return float(ck.get("best_reward", np.nan))


def random_best_at_budget(rewards, budget):
    if len(rewards) < budget:
        return np.nan
    return float(np.max(rewards[:budget]))


def collect_experiment_data():
    """Return dict: exp_name → {method → {budget → reward}}."""
    data = {}

    # 1) Sweep experiments (6) — REINFORCE alpha_high + random-640 + all-ones
    for sweep in sorted(os.listdir(ANALYSIS)):
        if not sweep.startswith("sweep_"):
            continue
        name = sweep[len("sweep_"):]
        sweep_dir = os.path.join(ANALYSIS, sweep)
        csv = os.path.join(sweep_dir, "configs", "alpha_high", "reinforce_log.csv")
        if not os.path.isfile(csv):
            continue
        rec = {"reinforce": {}, "random": {}, "cem": {}, "amortized": {}, "all_ones": None}
        # REINFORCE at each budget
        for b in BUDGETS:
            if b == 1:
                continue
            rec["reinforce"][b] = reinforce_best_at_budget(csv, 8, b)
        # Random from sweep_results.npz
        npz_path = os.path.join(sweep_dir, "sweep_results.npz")
        if os.path.isfile(npz_path):
            npz = np.load(npz_path)
            rand = np.asarray(npz["random_rewards"])
            for b in BUDGETS:
                if b == 1:
                    continue
                rec["random"][b] = random_best_at_budget(rand, b)
        # All-ones
        sj = os.path.join(sweep_dir, "summary.json")
        if os.path.isfile(sj):
            rec["all_ones"] = json.load(open(sj))["all_ones_reward"]
        data[name] = rec

    # 2) new_bgrich experiments — REINFORCE only (no random baseline for these)
    new_root = os.path.join(ANALYSIS, "new_bgrich")
    if os.path.isdir(new_root):
        for name in sorted(os.listdir(new_root)):
            csv = os.path.join(new_root, name, "reinforce_log.csv")
            if not os.path.isfile(csv):
                continue
            rec = data.get(name, {"reinforce": {}, "random": {}, "cem": {}, "amortized": {}, "all_ones": None})
            for b in BUDGETS:
                if b == 1:
                    continue
                rec["reinforce"][b] = reinforce_best_at_budget(csv, 8, b)
            data[name] = rec

    # 3) CEM results at budgets 40, 80, 160
    cem_root = os.path.join(ANALYSIS, "cem")
    if os.path.isdir(cem_root):
        for sub in sorted(os.listdir(cem_root)):
            # sub: <name>_budget{40,80,160}
            for b in (40, 80, 160):
                suffix = f"_budget{b}"
                if sub.endswith(suffix):
                    name = sub[:-len(suffix)]
                    r = cem_best(os.path.join(cem_root, sub))
                    if name not in data:
                        data[name] = {"reinforce": {}, "random": {}, "cem": {}, "amortized": {}, "all_ones": None}
                    data[name]["cem"][b] = r

    # 4) Amortized eval results (single-image)
    amort_root = os.path.join(ANALYSIS, "amortized")
    if os.path.isdir(amort_root):
        for name in sorted(os.listdir(amort_root)):
            ej = os.path.join(amort_root, name, "eval.json")
            if not os.path.isfile(ej):
                continue
            d = json.load(open(ej))
            if name not in data:
                data[name] = {"reinforce": {}, "random": {}, "cem": {}, "amortized": {}, "all_ones": None}
            data[name]["amortized"]["mlp"] = d["mlp"]["reward"]
            data[name]["amortized"]["ridge"] = d["ridge"]["reward"]
            data[name]["amortized"]["popmean"] = d["popmean"]["reward"]
            data[name]["amortized"]["oracle"] = d["oracle"]["reward"]

    return data


def aggregate_method(data, method_key, budgets, amortized_key=None):
    """Return (means, stds, n) arrays indexed by budget."""
    means, stds, ns = [], [], []
    for b in budgets:
        vals = []
        for name, rec in data.items():
            if method_key == "amortized":
                v = rec["amortized"].get(amortized_key, np.nan)
            else:
                v = rec[method_key].get(b, np.nan)
            if np.isfinite(v):
                vals.append(v)
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            ns.append(len(vals))
        else:
            means.append(np.nan); stds.append(np.nan); ns.append(0)
    return np.array(means), np.array(stds), np.array(ns)


def plot_sample_efficiency(data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # LEFT: aggregate across experiments
    ax = axes[0]
    budgets = BUDGETS

    methods = [
        ("reinforce", None,      "REINFORCE",     "tab:blue",   "o", "-"),
        ("cem",       None,      "CEM",           "tab:orange", "s", "-"),
        ("random",    None,      "Random search", "tab:gray",   "^", "--"),
    ]
    for key, amort, label, color, marker, style in methods:
        means, stds, ns = aggregate_method(data, key, budgets)
        valid = np.isfinite(means)
        xs = np.array(budgets)[valid]
        ys = means[valid]
        es = stds[valid]
        n = ns[valid][0] if len(ns[valid]) else 0
        ax.plot(xs, ys, label=f"{label} (N={n})", color=color, marker=marker,
                linestyle=style, linewidth=2, markersize=7)
        ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.15)

    # Amortized: single-shot
    for amort_key, color, marker in [("oracle", "tab:green", "*"),
                                       ("mlp", "tab:red", "D"),
                                       ("ridge", "tab:purple", "P"),
                                       ("popmean", "tab:brown", "X")]:
        means, stds, ns = aggregate_method(data, "amortized", [1], amortized_key=amort_key)
        if ns[0] > 0:
            ax.errorbar([1], means, yerr=stds, fmt=marker, color=color, markersize=10,
                        label=f"Amortized {amort_key} (1-shot, N={ns[0]})", capsize=5)

    # All-ones baseline as horizontal line
    all_ones_vals = [r["all_ones"] for r in data.values() if r["all_ones"] is not None]
    if all_ones_vals:
        aomean = np.mean(all_ones_vals)
        ax.axhline(aomean, color="black", linewidth=1.3, linestyle=":",
                   label=f"All-ones target ({aomean:.3f}, N={len(all_ones_vals)})")

    # Exhaustive 14-bit ceiling (α=0.5, raw float from original run — no JPG quantization)
    exh_dir = os.path.join(ANALYSIS, "exhaustive")
    if os.path.isdir(exh_dir):
        exh_maxes = []
        for name in os.listdir(exh_dir):
            orig_p = os.path.join(exh_dir, name, "exhaustive_rewards.npy")
            if os.path.isfile(orig_p):
                exh_maxes.append(float(np.load(orig_p).max()))
        if exh_maxes:
            exh_mean = np.mean(exh_maxes)
            ax.axhline(exh_mean, color="#2ca02c", linewidth=2.5, linestyle="-",
                       label=f"Exhaustive 14-bit α=0.5 (raw float, {exh_mean:.3f}, N={len(exh_maxes)})")

    ax.set_xscale("log")
    ax.set_xlabel("Images generated (log scale)")
    ax.set_ylabel("Best reward (averaged over experiments ± std)")
    ax.set_title("Sample efficiency: reward vs image budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # RIGHT: per-experiment table-style dots
    ax = axes[1]
    exp_names = sorted(data.keys())
    ys = np.arange(len(exp_names))
    for i, name in enumerate(exp_names):
        rec = data[name]
        # REINFORCE max
        rvals = [v for v in rec["reinforce"].values() if np.isfinite(v)]
        r_max = max(rvals) if rvals else np.nan
        # CEM@160
        c160 = rec["cem"].get(160, np.nan)
        # amortized MLP
        a_mlp = rec["amortized"].get("mlp", np.nan)
        # random@640
        rand640 = rec["random"].get(640, np.nan)

        if np.isfinite(r_max):
            ax.plot(r_max, i, 'o', color="tab:blue", markersize=8, alpha=0.8)
        if np.isfinite(c160):
            ax.plot(c160, i, 's', color="tab:orange", markersize=8, alpha=0.8)
        if np.isfinite(rand640):
            ax.plot(rand640, i, '^', color="tab:gray", markersize=6, alpha=0.6)
        if np.isfinite(a_mlp):
            ax.plot(a_mlp, i, 'D', color="tab:red", markersize=6, alpha=0.8)

    # Legend for right panel
    ax.plot([], [], 'o', color="tab:blue", label="REINFORCE best")
    ax.plot([], [], 's', color="tab:orange", label="CEM@160")
    ax.plot([], [], '^', color="tab:gray", label="Random@640")
    ax.plot([], [], 'D', color="tab:red", label="Amortized MLP")

    ax.set_yticks(ys)
    ax.set_yticklabels(exp_names, fontsize=8)
    ax.set_xlabel("Best reward")
    ax.set_title(f"Per-experiment comparison (N={len(exp_names)})")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=9, loc="lower right")

    plt.suptitle("Fast training-free binary path search — summary", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    data = collect_experiment_data()
    print(f"Collected {len(data)} experiments")
    for name, rec in sorted(data.items()):
        rein = len(rec["reinforce"])
        cem = len(rec["cem"])
        amort = len(rec["amortized"])
        rand = len(rec["random"])
        print(f"  {name:<28}  R={rein} C={cem} A={amort} rand={rand}  "
              f"all1={'Y' if rec['all_ones'] is not None else 'n'}")
    out = os.path.join(ANALYSIS, "sample_efficiency.png")
    plot_sample_efficiency(data, out)


if __name__ == "__main__":
    main()
