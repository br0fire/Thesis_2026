"""Compare 4 regimes of REINFORCE on new_bgrich: {14-bit, 28-bit} × {α=0.7, α=0.5}.

Shows best_reward_ever curves (averaged over 15 experiments each) with:
  - REINFORCE mean (per regime, solid)
  - Random running-max baseline (per regime, dashed)
  - All-ones baseline (per regime, horizontal)
  - Exhaustive upper bound (2^14 masks, horizontal — only for n_bits=14 regimes)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"

REGIMES = {
    "14-bit, α=0.7": ("new_bgrich",              "tab:blue",   True),
    "14-bit, α=0.5": ("new_bgrich_alpha05",      "tab:cyan",   True),
    "28-bit, α=0.7": ("new_bgrich_28bit",        "tab:red",    False),
    "28-bit, α=0.5": ("new_bgrich_28bit_alpha05","tab:orange", False),
}
BATCH = 8


def collect_regime(subdir):
    root = os.path.join(ANALYSIS, subdir)
    runs = []
    random_rewards = {}
    for name in sorted(os.listdir(root)):
        csv = os.path.join(root, name, "reinforce_log.csv")
        if os.path.isfile(csv):
            runs.append((name, pd.read_csv(csv)))
        rr = os.path.join(root, name, "random_rewards.npy")
        if os.path.isfile(rr):
            random_rewards[name] = np.load(rr)

    all_ones = {}
    aop = os.path.join(root, "_all_ones_rewards.json")
    if os.path.isfile(aop):
        d = json.load(open(aop))
        all_ones = {k: v["all_ones_reward"] for k, v in d.items()}

    return runs, random_rewards, all_ones


def aggregate_best(runs):
    if not runs:
        return None, None, 0
    max_ep = max(len(df) for _, df in runs)
    M = np.full((len(runs), max_ep), np.nan)
    for i, (_, df) in enumerate(runs):
        M[i, :len(df)] = df["best_reward_ever"].values
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0), len(runs)


def aggregate_random_max(random_rewards, max_ep):
    if not random_rewards:
        return None, 0
    arr = []
    for name, r in random_rewards.items():
        n_full = len(r) // BATCH
        row = np.full(max_ep, np.nan)
        for ep in range(min(n_full, max_ep)):
            row[ep] = r[:(ep + 1) * BATCH].max()
        arr.append(row)
    A = np.stack(arr)
    return np.nanmean(A, axis=0), len(random_rewards)


def load_exhaustive_max(alpha=0.5):
    """Per-experiment exhaustive rewards at given alpha.
    For α=0.5 uses raw-float original; otherwise derives from JPG bg/fg."""
    base = os.path.join(ANALYSIS, "exhaustive")
    if not os.path.isdir(base):
        return None, 0, {}
    per_exp = {}
    for name in sorted(os.listdir(base)):
        exp_dir = os.path.join(base, name)
        orig_p = os.path.join(exp_dir, "exhaustive_rewards.npy")
        bg_path = os.path.join(exp_dir, "exhaustive_bg_ssim.npy")
        fg_path = os.path.join(exp_dir, "exhaustive_fg_clip.npy")
        if abs(alpha - 0.5) < 1e-9 and os.path.isfile(orig_p):
            per_exp[name] = np.load(orig_p)
        elif os.path.isfile(bg_path) and os.path.isfile(fg_path):
            bg = np.load(bg_path); fg = np.load(fg_path)
            fg_sig = 1.0 / (1.0 + np.exp(-fg * 10.0))
            per_exp[name] = np.clip(bg, 1e-6, None)**alpha * np.clip(fg_sig, 1e-6, None)**(1 - alpha)
    if not per_exp:
        return None, 0, {}
    maxes = [float(r.max()) for r in per_exp.values()]
    return float(np.mean(maxes)), len(maxes), per_exp


def aggregate_exh_random_max(per_exp_rewards, n_bits, max_ep, batch=BATCH, n_samples=640, seed=42):
    """For each experiment, sample n_samples masks from exhaustive, compute running-max
    per episode (batch of `batch`), then average across experiments."""
    if not per_exp_rewards:
        return None, 0
    rng = np.random.default_rng(seed)
    arr = []
    for name, rewards in per_exp_rewards.items():
        idx = rng.choice(len(rewards), size=min(n_samples, len(rewards)), replace=False)
        sampled = rewards[idx]
        n_full = len(sampled) // batch
        row = np.full(max_ep, np.nan)
        for ep in range(min(n_full, max_ep)):
            row[ep] = sampled[: (ep + 1) * batch].max()
        arr.append(row)
    A = np.stack(arr)
    return np.nanmean(A, axis=0), len(arr)


def regime_alpha(label):
    return 0.7 if "α=0.7" in label else 0.5


def main():
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))

    # Panel (0,0): α=0.7 regimes (exhaustive at α=0.7 as upper bound)
    # Panel (0,1): α=0.5 regimes (exhaustive at α=0.5 as upper bound)
    # Panel (1,0): all 4 REINFORCE curves side-by-side (solid=reinforce, dashed=random-from-exhaustive at regime's α)
    # Panel (1,1): numerical summary as text

    # Pre-compute exhaustives
    exh07 = load_exhaustive_max(alpha=0.7)  # (mean, n, per_exp)
    exh05 = load_exhaustive_max(alpha=0.5)

    def plot_alpha_panel(ax, alpha_tag, title):
        alpha_val = float(alpha_tag.split("=")[1])
        per_exp_exh = exh07[2] if alpha_val == 0.7 else exh05[2]
        exh_mean = exh07[0] if alpha_val == 0.7 else exh05[0]
        n_exh = exh07[1] if alpha_val == 0.7 else exh05[1]

        for label, (subdir, color, is_14bit) in REGIMES.items():
            if alpha_tag not in label:
                continue
            runs, rr, ao = collect_regime(subdir)
            best_mean, best_std, n = aggregate_best(runs)
            if best_mean is None:
                continue
            x = np.arange(len(best_mean))
            ax.plot(x, best_mean, color=color, linewidth=2.5, label=f"REINFORCE {label} (N={n})")
            ax.fill_between(x, best_mean - best_std, best_mean + best_std, color=color, alpha=0.15)

            # Random baseline — recompute from exhaustive at regime's α for fairness
            if is_14bit and per_exp_exh:
                rand_mean, n_r = aggregate_exh_random_max(per_exp_exh, 14, len(best_mean))
            else:
                # 28-bit: use the ORIGINAL random_rewards.npy (already at α=0.5 by default,
                # but for α=0.7 28-bit regime these are mismatched)
                rand_mean, n_r = aggregate_random_max(rr, len(best_mean))
            if rand_mean is not None:
                src = "from exhaustive" if is_14bit and per_exp_exh else "original (α=0.5)"
                ax.plot(x, rand_mean, color=color, linewidth=1.5, linestyle="--",
                        label=f"random {label} ({src}, N={n_r})")

            if ao:
                mean_ao = np.mean(list(ao.values()))  # all-ones saved was at α=0.5
                ax.axhline(mean_ao, color=color, linestyle=":", linewidth=1.3,
                           label=f"all-ones {label} (at α=0.5: {mean_ao:.3f})")

        if exh_mean is not None:
            ax.axhline(exh_mean, color="black", linewidth=2.2, linestyle="-",
                       label=f"Exhaustive α={alpha_val} (best={exh_mean:.3f}, N={n_exh})")
        ax.set_xlabel("Episode"); ax.set_ylabel("best_reward_ever")
        ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    plot_alpha_panel(axes[0, 0], "α=0.7", "α=0.7: REINFORCE + random + exhaustive (α=0.7)")
    plot_alpha_panel(axes[0, 1], "α=0.5", "α=0.5: REINFORCE + random + exhaustive (α=0.5)")

    # Panel (1,0): all 4 REINFORCE curves side-by-side (like before, for overview)
    ax = axes[1, 0]
    for label, (subdir, color, is_14bit) in REGIMES.items():
        runs, rr, ao = collect_regime(subdir)
        best_mean, best_std, n = aggregate_best(runs)
        if best_mean is None:
            continue
        x = np.arange(len(best_mean))
        ax.plot(x, best_mean, color=color, linewidth=2.5, label=f"REINFORCE {label}")
        ax.fill_between(x, best_mean - best_std, best_mean + best_std, color=color, alpha=0.12)
    if exh07[0] is not None:
        ax.axhline(exh07[0], color="darkblue", linewidth=1.6, linestyle="-",
                   label=f"Exhaustive α=0.7 ({exh07[0]:.3f})")
    if exh05[0] is not None:
        ax.axhline(exh05[0], color="maroon", linewidth=1.6, linestyle="-",
                   label=f"Exhaustive α=0.5 ({exh05[0]:.3f})")
    ax.set_xlabel("Episode"); ax.set_ylabel("best_reward_ever")
    ax.set_title("All 4 REINFORCE regimes (overlaid)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")

    # Panel (1,1): table summary
    ax = axes[1, 1]
    ax.axis("off")
    table_text = f"{'regime':<16} {'REINFORCE':>10} {'random':>8} {'all-ones':>10} {'exhaustive':>12}\n"
    table_text += "─" * 62 + "\n"
    for label, (subdir, color, is_14bit) in REGIMES.items():
        runs, rr, ao = collect_regime(subdir)
        bm, _, _ = aggregate_best(runs)
        if bm is None:
            continue
        alpha_val = regime_alpha(label)
        if is_14bit:
            per_exp = exh07[2] if alpha_val == 0.7 else exh05[2]
            rm, _ = aggregate_exh_random_max(per_exp, 14, len(bm))
            exh_v = exh07[0] if alpha_val == 0.7 else exh05[0]
        else:
            rm, _ = aggregate_random_max(rr, len(bm))
            exh_v = None
        ao_mean = np.mean(list(ao.values())) if ao else float("nan")
        exh_str = f"{exh_v:.4f}" if exh_v is not None else "—"
        rm_str = f"{rm[-1]:.4f}" if rm is not None else "—"
        table_text += f"{label:<16} {bm[-1]:>10.4f} {rm_str:>8} {ao_mean:>10.4f} {exh_str:>12}\n"
    ax.text(0.02, 0.97, table_text, family="monospace", fontsize=10, va="top")
    ax.set_title("Summary (reward at ep 80, avg over 15 experiments)")

    plt.suptitle("new_bgrich: 4 regimes — REINFORCE vs random vs exhaustive (per-α)",
                 fontsize=15, weight="bold")
    plt.tight_layout()
    out = os.path.join(ANALYSIS, "compare_4regimes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Print numerical summary
    print("\n=== Summary (ep 80, avg over 15 exps) ===")
    print(f"{'regime':<20} {'REINFORCE':>10} {'random':>10} {'all-ones':>10} {'exh':>8}")
    for label, (subdir, color, is_14bit) in REGIMES.items():
        runs, rr, ao = collect_regime(subdir)
        bm, _, _ = aggregate_best(runs)
        if bm is None:
            continue
        alpha_val = regime_alpha(label)
        if is_14bit:
            per_exp = exh07[2] if alpha_val == 0.7 else exh05[2]
            rm, _ = aggregate_exh_random_max(per_exp, 14, len(bm))
            exh_v = exh07[0] if alpha_val == 0.7 else exh05[0]
        else:
            rm, _ = aggregate_random_max(rr, len(bm))
            exh_v = None
        ao_mean = np.mean(list(ao.values())) if ao else float("nan")
        exh_str = f"{exh_v:.4f}" if exh_v is not None else "—"
        rm_str = f"{rm[-1]:.4f}" if rm is not None else "—"
        print(f"{label:<20} {bm[-1]:>10.4f} {rm_str:>10} {ao_mean:>10.4f} {exh_str:>8}")


if __name__ == "__main__":
    main()
