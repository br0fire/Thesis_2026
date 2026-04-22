"""
Prior-generalization analysis on 8 diverse (non-bgrich) prompts.

Compares: random-120 vs REINFORCE-15ep vs REINFORCE_prior-15ep (bgrich prior as init).
Emits a CSV, a grouped-bar PNG, and prints a terminal report.

Usage:
    python analysis/diverse_prior_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1/diverse"
OUT_DIR = "/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1"


def best_from_reinforce_log(path):
    if not os.path.isfile(path):
        return np.nan, np.nan
    df = pd.read_csv(path)
    col_max = "reward_max" if "reward_max" in df else "mean_reward"
    best = float(df[col_max].cummax().iloc[-1])
    init_mean = float(df["mean_reward"].iloc[0]) if "mean_reward" in df else np.nan
    return best, init_mean


def collect():
    rows = []
    for exp in sorted(os.listdir(ROOT)):
        r = np.load(f"{ROOT}/{exp}/random/random_rewards.npy")
        rf_best, rf_init = best_from_reinforce_log(f"{ROOT}/{exp}/reinforce/reinforce_log.csv")
        rp_best, rp_init = best_from_reinforce_log(f"{ROOT}/{exp}/reinforce_prior/reinforce_log.csv")
        rows.append(dict(
            exp=exp,
            random_best=float(r.max()),
            random_mean=float(r.mean()),
            reinforce_best=rf_best,
            reinforce_init=rf_init,
            prior_best=rp_best,
            prior_init=rp_init,
        ))
    return pd.DataFrame(rows)


def plot(df, out_path):
    exps = df["exp"].tolist()
    x = np.arange(len(exps))
    w = 0.27

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - w, df["random_best"],    w, label="random-120",        color="#888")
    ax.bar(x,     df["reinforce_best"], w, label="REINFORCE (no prior)", color="#1f6feb")
    ax.bar(x + w, df["prior_best"],     w, label="REINFORCE + bgrich prior", color="#a23b3b")

    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=30, ha="right")
    ax.set_ylabel("best reward (α=0.5)")
    ax.set_title("Does the bgrich prior generalize? — 8 diverse scenes, budget = 120 each")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(1.0, df[["random_best","reinforce_best","prior_best"]].values.max() * 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    df = collect()
    csv_path = os.path.join(OUT_DIR, "diverse_prior_summary.csv")
    df.to_csv(csv_path, index=False)

    png_path = os.path.join(OUT_DIR, "diverse_prior_comparison.png")
    plot(df, png_path)

    print("Per-experiment:")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(df.round(3).to_string(index=False))

    print("\n=== Mean best reward over 8 diverse scenes ===")
    print(df[["random_best", "reinforce_best", "prior_best"]].mean().round(3).to_string())

    wins = {
        "prior > random":     int((df["prior_best"]     > df["random_best"]).sum()),
        "prior > no-prior":   int((df["prior_best"]     > df["reinforce_best"]).sum()),
        "no-prior > random":  int((df["reinforce_best"] > df["random_best"]).sum()),
    }
    print("\n=== Head-to-head wins (out of 8) ===")
    for k, v in wins.items():
        print(f"  {k:<22} {v}/8")

    print("\n=== Warm-start effect (prior_init vs reinforce_init) ===")
    delta = (df["prior_init"] - df["reinforce_init"])
    print(f"  mean Δ = {delta.mean():+.3f}   (+ means prior helps first episode)")
    print(f"  per-exp: {[round(v, 3) for v in delta.tolist()]}")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
