"""Generate all research/analysis plots for clean_v1 pipeline.

Outputs:
  clean_v1/clean_v1_summary.json          — per-exp stats
  clean_v1/clean_v1_summary.csv           — same as table
  clean_v1/clean_v1_training_curves.png   — REINFORCE curves + ceiling/random/all-ones
  clean_v1/clean_v1_histograms.png        — reward landscapes (16 panels)
  clean_v1/clean_v1_sample_efficiency.png — reward vs budget
  clean_v1/clean_v1_visual_grid.png       — source | target | REINFORCE | random | exhaustive
  clean_v1/clean_v1_reward_vs_ceiling.png — per-exp gap to ceiling
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1"
IMGS_ROOT = os.path.join(ROOT, "exhaustive_images")
ALPHA = 0.5
N_BITS = 14
BATCH = 8


def compute_reward(bg, fg, alpha=ALPHA):
    fg_sig = 1.0 / (1.0 + np.exp(-fg * 10.0))
    return (np.clip(bg, 1e-6, None) ** alpha
            * np.clip(fg_sig, 1e-6, None) ** (1 - alpha))


def parse_prompts(p):
    out = {}
    with open(p) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def collect_per_experiment():
    """Walk clean_v1/canonical/ and assemble full per-experiment data."""
    canon_root = os.path.join(ROOT, "canonical")
    exh_root = os.path.join(ROOT, "exhaustive")
    rein_root = os.path.join(ROOT, "reinforce_a05")
    exps = sorted(d for d in os.listdir(canon_root)
                  if os.path.isdir(os.path.join(canon_root, d)))
    rng = np.random.default_rng(42)

    records = []
    for name in exps:
        rec = {"name": name}
        # prompts
        pp = os.path.join(canon_root, name, "prompts.txt")
        rec["prompts"] = parse_prompts(pp) if os.path.isfile(pp) else {}

        # exhaustive
        bg_p = os.path.join(exh_root, name, "bg_ssim.npy")
        fg_p = os.path.join(exh_root, name, "fg_clip.npy")
        if os.path.isfile(bg_p) and os.path.isfile(fg_p):
            bg = np.load(bg_p); fg = np.load(fg_p)
            rewards_all = compute_reward(bg, fg, ALPHA)
            rec["exh"] = {
                "rewards": rewards_all,
                "bg_ssim": bg,
                "fg_clip": fg,
                "max": float(rewards_all.max()),
                "min": float(rewards_all.min()),
                "mean": float(rewards_all.mean()),
                "std": float(rewards_all.std()),
                "best_mask_int": int(np.argmax(rewards_all)),
                "all_ones_reward": float(rewards_all[-1]),  # all-ones mask = last index
                "source_self_reward": float(rewards_all[0]),  # all-zeros mask
            }
            # Random-640 baseline from exhaustive subset
            idx = rng.choice(len(rewards_all), 640, replace=False)
            rec["random_640"] = {
                "best": float(rewards_all[idx].max()),
                "mean": float(rewards_all[idx].mean()),
                "running_max_per_batch": np.maximum.accumulate(
                    [rewards_all[idx[ep * BATCH:(ep + 1) * BATCH]].max()
                     for ep in range(80)]
                ).tolist(),
            }
        else:
            rec["exh"] = None
            rec["random_640"] = None

        # REINFORCE
        csv_p = os.path.join(rein_root, name, "reinforce_log.csv")
        res_p = os.path.join(rein_root, name, "reinforce_result.pt")
        if os.path.isfile(csv_p):
            df = pd.read_csv(csv_p)
            rec["reinforce"] = {
                "log": df,
                "best_reward": float(df["best_reward_ever"].iloc[-1]),
                "mean_reward_final": float(df["mean_reward"].iloc[-10:].mean()),
                "n_episodes": len(df),
            }
            if os.path.isfile(res_p):
                ck = torch.load(res_p, map_location="cpu", weights_only=False)
                probs = ck.get("probs")
                if torch.is_tensor(probs):
                    probs = probs.numpy()
                rec["reinforce"]["final_probs"] = probs.tolist() if probs is not None else None
                rec["reinforce"]["best_mask_int"] = ck.get("best_mask_int", -1)
        else:
            rec["reinforce"] = None

        records.append(rec)

    return records


# ─────────────────────────────────────────────────────────────
# Summary JSON + CSV
# ─────────────────────────────────────────────────────────────
def save_summary(records):
    summary = []
    for r in records:
        entry = {"name": r["name"], "seg_prompt": r["prompts"].get("seg", "")}
        if r["exh"]:
            entry["exh_max"]       = r["exh"]["max"]
            entry["exh_mean"]      = r["exh"]["mean"]
            entry["exh_std"]       = r["exh"]["std"]
            entry["all_ones_R"]    = r["exh"]["all_ones_reward"]
            entry["source_self_R"] = r["exh"]["source_self_reward"]
            entry["best_mask_int"] = r["exh"]["best_mask_int"]
        if r["random_640"]:
            entry["random_640_best"] = r["random_640"]["best"]
            entry["random_640_mean"] = r["random_640"]["mean"]
        if r["reinforce"]:
            entry["reinforce_best"]     = r["reinforce"]["best_reward"]
            entry["reinforce_mean_fin"] = r["reinforce"]["mean_reward_final"]
            entry["reinforce_final_probs"] = r["reinforce"].get("final_probs")
        if r["exh"] and r["reinforce"]:
            entry["reinforce_vs_ceiling"] = r["reinforce"]["best_reward"] / r["exh"]["max"]
        summary.append(entry)

    json_p = os.path.join(ROOT, "clean_v1_summary.json")
    with open(json_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_p}")

    csv_p = os.path.join(ROOT, "clean_v1_summary.csv")
    scalar_keys = ["name", "seg_prompt", "exh_max", "exh_mean", "exh_std",
                   "all_ones_R", "source_self_R",
                   "random_640_best", "random_640_mean",
                   "reinforce_best", "reinforce_mean_fin", "reinforce_vs_ceiling"]
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        w.writeheader()
        for e in summary:
            w.writerow({k: e.get(k, "") for k in scalar_keys})
    print(f"Saved: {csv_p}")
    return summary


# ─────────────────────────────────────────────────────────────
# Training curves plot
# ─────────────────────────────────────────────────────────────
def plot_training_curves(records):
    runs = [r for r in records if r["reinforce"] and r["exh"]]
    if not runs:
        return
    N = len(runs)
    max_ep = max(len(r["reinforce"]["log"]) for r in runs)

    def stack(col):
        M = np.full((N, max_ep), np.nan)
        for i, r in enumerate(runs):
            v = r["reinforce"]["log"][col].values
            M[i, :len(v)] = v
        return M

    mean_r = stack("mean_reward")
    best_r = stack("best_reward_ever")
    bg_stk = stack("mean_bg_ssim")
    fg_stk = stack("mean_fg_clip")
    ent_stk = stack("entropy")

    # Random running-max aggregated
    rand_maxes = np.full((N, max_ep), np.nan)
    for i, r in enumerate(runs):
        vals = r["random_640"]["running_max_per_batch"]
        rand_maxes[i, :len(vals)] = vals
    rand_max_curve = np.nanmean(rand_maxes, axis=0)

    exh_max_mean = np.mean([r["exh"]["max"] for r in runs])
    all_ones_mean = np.mean([r["exh"]["all_ones_reward"] for r in runs])

    # Components for bg_ssim / fg_clip reference lines
    # all-ones = last index (2^n_bits - 1), exhaustive best = argmax of rewards
    all_ones_bg = np.mean([r["exh"]["bg_ssim"][-1] for r in runs])
    all_ones_fg = np.mean([r["exh"]["fg_clip"][-1] for r in runs])
    exh_best_bg = np.mean([r["exh"]["bg_ssim"][r["exh"]["best_mask_int"]] for r in runs])
    exh_best_fg = np.mean([r["exh"]["fg_clip"][r["exh"]["best_mask_int"]] for r in runs])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, N))

    def plot_panel(ax, M, title, ylabel, ref_lines=()):
        x = np.arange(M.shape[1])
        for i, r in enumerate(runs):
            vals = M[i]; m = ~np.isnan(vals)
            ax.plot(x[m], vals[m], color=colors[i], alpha=0.2, linewidth=0.7, label=r["name"])
        mean = np.nanmean(M, axis=0); std = np.nanstd(M, axis=0)
        ax.plot(x, mean, color="black", linewidth=2.8, label=f"REINFORCE mean ± std (N={N})")
        ax.fill_between(x, mean - std, mean + std, color="black", alpha=0.15)
        for line_val, line_col, line_style, line_label in ref_lines:
            ax.axhline(line_val, color=line_col, linestyle=line_style, linewidth=2,
                       label=line_label)
        ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        keep = [i for i, l in enumerate(labels)
                if "REINFORCE mean" in l or "ceiling" in l or "all-ones" in l
                   or "random" in l or "max entropy" in l or "exhaustive" in l]
        if keep:
            ax.legend([handles[i] for i in keep], [labels[i] for i in keep], fontsize=9,
                      loc="lower right" if "entropy" not in ylabel else "upper right")

    plot_panel(axes[0, 0], mean_r, "mean_reward per episode", "mean_reward",
               ref_lines=[(all_ones_mean, "#e41a1c", ":", f"all-ones ({all_ones_mean:.3f})")])
    # Best-so-far with random and ceiling
    ax = axes[0, 1]
    x = np.arange(max_ep)
    for i, r in enumerate(runs):
        vals = best_r[i]; m = ~np.isnan(vals)
        ax.plot(x[m], vals[m], color=colors[i], alpha=0.2, linewidth=0.7)
    mean = np.nanmean(best_r, axis=0); std = np.nanstd(best_r, axis=0)
    ax.plot(x, mean, color="black", linewidth=2.8, label=f"REINFORCE mean ± std (N={N})")
    ax.fill_between(x, mean - std, mean + std, color="black", alpha=0.15)
    ax.plot(x, rand_max_curve, color="#1f77b4", linewidth=3.0, linestyle="--",
            label=f"random-640 running max (avg, N={N})")
    ax.axhline(exh_max_mean, color="#2ca02c", linestyle="-", linewidth=2.2,
               label=f"exhaustive ceiling ({exh_max_mean:.3f}, N={N})")
    ax.axhline(all_ones_mean, color="#e41a1c", linestyle=":", linewidth=2.0,
               label=f"all-ones ({all_ones_mean:.3f})")
    ax.set_xlabel("Episode"); ax.set_ylabel("best_reward_ever")
    ax.set_title(f"best_reward_ever (α={ALPHA})"); ax.grid(alpha=0.3)
    # Zoom y-axis to focus on the gap between REINFORCE / random / ceiling.
    # Lower bound: just below all_ones; upper: slightly above ceiling.
    ymin = min(all_ones_mean - 0.01, float(np.nanmin(rand_max_curve)) - 0.005)
    ymax = max(exh_max_mean + 0.01, float(np.nanmax(best_r)) + 0.005)
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=9, loc="lower right")

    plot_panel(axes[1, 0], bg_stk, "mean bg_ssim per episode", "bg_ssim",
               ref_lines=[(all_ones_bg, "#e41a1c", ":", f"all-ones ({all_ones_bg:.3f})"),
                          (exh_best_bg, "#2ca02c", "-", f"exhaustive best ({exh_best_bg:.3f})")])
    plot_panel(axes[1, 1], fg_stk, "mean fg_clip per episode", "fg_clip",
               ref_lines=[(all_ones_fg, "#e41a1c", ":", f"all-ones ({all_ones_fg:.3f})"),
                          (exh_best_fg, "#2ca02c", "-", f"exhaustive best ({exh_best_fg:.3f})")])

    plt.suptitle(f"clean_v1 REINFORCE α={ALPHA}, {N_BITS}-bit, N={N} experiments",
                 fontsize=14, weight="bold")
    plt.tight_layout()
    out = os.path.join(ROOT, "clean_v1_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Histograms of reward landscape
# ─────────────────────────────────────────────────────────────
def plot_histograms(records):
    runs = [r for r in records if r["exh"]]
    if not runs:
        return
    N = len(runs)
    ncols = 4; nrows = (N + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), squeeze=False)

    for i, r in enumerate(runs):
        ax = axes[i // ncols, i % ncols]
        rewards = r["exh"]["rewards"]
        ax.hist(rewards, bins=60, color="#888", alpha=0.7, density=True)
        ax.axvline(r["exh"]["max"], color="#2ca02c", linewidth=2,
                   label=f"max={r['exh']['max']:.3f}")
        ax.axvline(r["exh"]["mean"], color="gray", linewidth=1, linestyle=":",
                   label=f"mean={r['exh']['mean']:.3f}")
        if r["random_640"]:
            ax.axvline(r["random_640"]["best"], color="#1f77b4", linewidth=1.5, linestyle="--",
                       label=f"random-640 best={r['random_640']['best']:.3f}")
        if r["reinforce"]:
            ax.axvline(r["reinforce"]["best_reward"], color="black", linewidth=1.8,
                       label=f"REINFORCE best={r['reinforce']['best_reward']:.3f}")
        ax.axvline(r["exh"]["all_ones_reward"], color="#e41a1c", linewidth=1.5, linestyle=":",
                   label=f"all-ones={r['exh']['all_ones_reward']:.3f}")
        ax.set_title(f"{r['name']}\nσ={r['exh']['std']:.3f}", fontsize=9)
        ax.set_xlabel("reward"); ax.set_ylabel("density")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

    for j in range(N, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    plt.suptitle(f"Reward landscapes α={ALPHA} (exhaustive 2^{N_BITS} = {1 << N_BITS} masks)",
                 fontsize=14, weight="bold")
    plt.tight_layout()
    out = os.path.join(ROOT, "clean_v1_histograms.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Sample efficiency
# ─────────────────────────────────────────────────────────────
def plot_sample_efficiency(records):
    runs = [r for r in records if r["exh"] and r["reinforce"]]
    if not runs:
        return
    N = len(runs)

    budgets = [1, 8, 40, 80, 160, 320, 640]
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # REINFORCE at various episode budgets
    reinf_curves = np.full((N, len(budgets)), np.nan)
    for i, r in enumerate(runs):
        df = r["reinforce"]["log"]
        for bi, b in enumerate(budgets):
            ep = b // BATCH
            if ep == 0 or ep > len(df):
                continue
            reinf_curves[i, bi] = df["best_reward_ever"].iloc[min(ep - 1, len(df) - 1)]
    reinf_mean = np.nanmean(reinf_curves, axis=0)
    reinf_std = np.nanstd(reinf_curves, axis=0)
    valid = np.isfinite(reinf_mean)
    xs = np.array(budgets)[valid]
    ax.plot(xs, reinf_mean[valid], "o-", color="tab:blue", linewidth=2, markersize=8,
            label=f"REINFORCE (N={N})")
    ax.fill_between(xs, reinf_mean[valid] - reinf_std[valid],
                    reinf_mean[valid] + reinf_std[valid], color="tab:blue", alpha=0.15)

    # Random at various budgets (sampled from exhaustive)
    rng = np.random.default_rng(42)
    rand_curves = np.full((N, len(budgets)), np.nan)
    for i, r in enumerate(runs):
        rewards = r["exh"]["rewards"]
        for bi, b in enumerate(budgets):
            if b > len(rewards):
                continue
            idx = rng.choice(len(rewards), b, replace=False)
            rand_curves[i, bi] = rewards[idx].max()
    rand_mean = np.nanmean(rand_curves, axis=0)
    rand_std = np.nanstd(rand_curves, axis=0)
    valid = np.isfinite(rand_mean)
    ax.plot(np.array(budgets)[valid], rand_mean[valid], "^-", color="tab:gray", linewidth=2,
            markersize=7, label=f"random (N={N})", linestyle="--")
    ax.fill_between(np.array(budgets)[valid], rand_mean[valid] - rand_std[valid],
                    rand_mean[valid] + rand_std[valid], color="tab:gray", alpha=0.15)

    # Exhaustive ceiling horizontal
    exh_mean = np.mean([r["exh"]["max"] for r in runs])
    ax.axhline(exh_mean, color="#2ca02c", linestyle="-", linewidth=2.5,
               label=f"Exhaustive ceiling ({exh_mean:.3f})")

    # All-ones
    ao = np.mean([r["exh"]["all_ones_reward"] for r in runs])
    ax.axhline(ao, color="#e41a1c", linestyle=":", linewidth=2,
               label=f"All-ones ({ao:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Images generated (log)")
    ax.set_ylabel("Best reward (avg over experiments)")
    ax.set_title(f"Sample efficiency α={ALPHA}, {N_BITS}-bit")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(ROOT, "clean_v1_sample_efficiency.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Visual comparison grid
# ─────────────────────────────────────────────────────────────
def build_visual_grid(records):
    CELL = 220; LABEL_H = 38
    runs = [r for r in records if r["exh"] and r["reinforce"]]
    if not runs:
        return

    def load_from_npy(exp_name, mask_int):
        imgs_npy = os.path.join(IMGS_ROOT, exp_name, "all_images.npy")
        if not os.path.isfile(imgs_npy):
            return None
        arr = np.load(imgs_npy, mmap_mode="r")
        img = np.array(arr[mask_int]).transpose(1, 2, 0)  # (H, W, 3)
        return Image.fromarray(img)

    def load_canonical(exp_name, fname):
        p = os.path.join(ROOT, "canonical", exp_name, fname)
        return Image.open(p).convert("RGB") if os.path.isfile(p) else None

    def label(pil, top, bot=""):
        img = pil.resize((CELL, CELL), Image.LANCZOS)
        canvas = Image.new("RGB", (CELL, CELL + LABEL_H), (245, 245, 245))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except OSError:
            f = ImageFont.load_default()
        draw.text((4, CELL + 2), top, fill=(0, 0, 0), font=f)
        if bot:
            draw.text((4, CELL + 18), bot, fill=(80, 80, 80), font=f)
        return canvas

    def placeholder(text):
        return label(Image.new("RGB", (CELL, CELL), (230, 230, 230)), text, "—")

    rng = np.random.default_rng(42)
    rows = []
    for r in runs:
        cells = []
        # 1. Source (canonical)
        src = load_canonical(r["name"], "source.png")
        cells.append(label(src or Image.new("RGB", (CELL, CELL)), r["name"], "source"))
        # 2. Target (canonical)
        tgt = load_canonical(r["name"], "target.png")
        ao_R = r["exh"]["all_ones_reward"]
        cells.append(label(tgt or Image.new("RGB", (CELL, CELL)), "target (all-1s)",
                           f"R={ao_R:.3f}"))
        # 3. REINFORCE best (lookup by best_mask_int)
        bmi = r["reinforce"].get("best_mask_int", -1)
        if bmi >= 0:
            img = load_from_npy(r["name"], bmi)
            if img:
                cells.append(label(img, "REINFORCE best", f"R={r['reinforce']['best_reward']:.3f}"))
            else:
                cells.append(placeholder("REINFORCE"))
        else:
            cells.append(placeholder("REINFORCE"))
        # 4. Random-640 best (find best in subset)
        rewards = r["exh"]["rewards"]
        idx = rng.choice(len(rewards), 640, replace=False)
        rand_best_mi = int(idx[np.argmax(rewards[idx])])
        img = load_from_npy(r["name"], rand_best_mi)
        if img:
            cells.append(label(img, "random-640 best",
                               f"R={float(rewards[rand_best_mi]):.3f}"))
        else:
            cells.append(placeholder("random"))
        # 5. Exhaustive max
        exh_mi = r["exh"]["best_mask_int"]
        img = load_from_npy(r["name"], exh_mi)
        if img:
            cells.append(label(img, "exhaustive max", f"R={r['exh']['max']:.3f}"))
        else:
            cells.append(placeholder("exhaustive"))

        row_w = sum(c.width for c in cells) + (len(cells) - 1) * 4
        row = Image.new("RGB", (row_w, cells[0].height), (220, 220, 220))
        x = 0
        for c in cells:
            row.paste(c, (x, 0)); x += c.width + 4
        rows.append(row)

    W = max(r.width for r in rows)
    H = sum(r.height + 6 for r in rows)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y)); y += r.height + 6

    out = os.path.join(ROOT, "clean_v1_visual_grid.png")
    canvas.save(out, quality=95)
    print(f"Saved: {out}  ({W}x{H})  {len(rows)} rows")


# ─────────────────────────────────────────────────────────────
# Reward vs Ceiling per-experiment plot
# ─────────────────────────────────────────────────────────────
def plot_per_experiment_components(records):
    """Per-experiment bg_ssim and fg_clip curves with experiment-specific
    all-ones and exhaustive-best reference lines."""
    runs = [r for r in records if r["reinforce"] and r["exh"]]
    if not runs:
        return
    N = len(runs)
    ncols = 4
    nrows = (N + ncols - 1) // ncols

    for metric, col, title_prefix in [("mean_bg_ssim", "bg_ssim", "bg_ssim"),
                                       ("mean_fg_clip", "fg_clip", "fg_clip")]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows), squeeze=False)
        for i, r in enumerate(runs):
            ax = axes[i // ncols, i % ncols]
            df = r["reinforce"]["log"]
            vals = df[metric].values
            x = np.arange(len(vals))
            ax.plot(x, vals, color="black", linewidth=1.5,
                    label=f"REINFORCE {col}")

            # all-ones: value at last index
            ao_val = float(r["exh"][col][-1])
            ax.axhline(ao_val, color="#e41a1c", linestyle=":", linewidth=1.8,
                       label=f"all-ones ({ao_val:.3f})")

            # exhaustive best: value at argmax reward
            exh_val = float(r["exh"][col][r["exh"]["best_mask_int"]])
            ax.axhline(exh_val, color="#2ca02c", linestyle="-", linewidth=1.8,
                       label=f"exhaustive best ({exh_val:.3f})")

            ax.set_title(r["name"], fontsize=9)
            ax.set_xlabel("Episode"); ax.set_ylabel(col)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")

        for j in range(N, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        plt.suptitle(f"Per-experiment {col} curves with all-ones and exhaustive best "
                     f"(α={ALPHA}, N={N})", fontsize=13, weight="bold")
        plt.tight_layout()
        out = os.path.join(ROOT, f"clean_v1_per_exp_{col}.png")
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def plot_per_experiment_all4(records):
    """One figure with 4 metrics per experiment: mean_reward | best_reward_ever | bg_ssim | fg_clip.
    Each row = one experiment, each column = one metric."""
    runs = [r for r in records if r["reinforce"] and r["exh"]]
    if not runs:
        return
    N = len(runs)
    # Tall rows so individual panels have enough vertical resolution to show tiny gaps.
    fig, axes = plt.subplots(N, 4, figsize=(22, 4.8 * N), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.30, top=0.97, bottom=0.025, left=0.05, right=0.98)

    col_titles = ["mean_reward", "best_reward_ever", "bg_ssim", "fg_clip"]

    for i, r in enumerate(runs):
        name = r["name"]
        df = r["reinforce"]["log"]
        mean_r = df["mean_reward"].values
        best_r = df["best_reward_ever"].values
        bg = df["mean_bg_ssim"].values
        fg = df["mean_fg_clip"].values
        x = np.arange(len(mean_r))

        exh_max = r["exh"]["max"]
        exh_best_int = r["exh"]["best_mask_int"]
        all_ones_r = r["exh"]["all_ones_reward"]
        all_ones_bg = float(r["exh"]["bg_ssim"][-1])
        all_ones_fg = float(r["exh"]["fg_clip"][-1])
        exh_best_bg = float(r["exh"]["bg_ssim"][exh_best_int])
        exh_best_fg = float(r["exh"]["fg_clip"][exh_best_int])
        rand_max_curve = r["random_640"]["running_max_per_batch"]

        def draw(ax, y, col_idx, ao_val, exh_val, extra_curve=None):
            ax.plot(x, y, color="black", linewidth=1.5)
            ax.axhline(ao_val, color="#e41a1c", linestyle=":", linewidth=1.5)
            ax.axhline(exh_val, color="#2ca02c", linestyle="-", linewidth=1.5)
            if extra_curve is not None:
                rx = np.arange(len(extra_curve))
                ax.plot(rx, extra_curve, color="#1f77b4", linewidth=1.2, linestyle="--")
            ax.grid(alpha=0.3)
            # Annotate values near right edge at their y-level
            xmax = x[-1] if len(x) else 80
            ax.annotate(f"ao={ao_val:.3f}", xy=(xmax, ao_val), xytext=(-2, 3),
                        textcoords="offset points", ha="right", va="bottom",
                        color="#e41a1c", fontsize=7)
            ax.annotate(f"exh={exh_val:.3f}", xy=(xmax, exh_val), xytext=(-2, 3),
                        textcoords="offset points", ha="right", va="bottom",
                        color="#2ca02c", fontsize=7)

        draw(axes[i, 0], mean_r, 0, all_ones_r, exh_max)
        draw(axes[i, 1], best_r, 1, all_ones_r, exh_max, extra_curve=rand_max_curve)
        draw(axes[i, 2], bg,     2, all_ones_bg, exh_best_bg)
        draw(axes[i, 3], fg,     3, all_ones_fg, exh_best_fg)

        # Zoom best_reward_ever to focus on the race region — where random curve sits.
        # y-range: from roughly random start (lowest of min rand, min best) to just above ceiling.
        lows = []
        if rand_max_curve:
            lows.append(min(rand_max_curve))
        if len(best_r):
            lows.append(float(best_r.min()))
        if lows:
            y_lo = min(lows) - 0.005
        else:
            y_lo = all_ones_r
        y_hi = exh_max + 0.003
        # Enforce minimum visible span to avoid flat-looking extremes
        if y_hi - y_lo < 0.03:
            mid = (y_hi + y_lo) / 2
            y_lo = mid - 0.02
            y_hi = mid + 0.02
        axes[i, 1].set_ylim(y_lo, y_hi)

        axes[i, 0].set_ylabel(name, fontsize=10, weight="bold")
        # Column titles only on first row
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=11)
        if i == N - 1:
            for j in range(4):
                axes[i, j].set_xlabel("Episode")

    # ONE shared legend at top
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=2, label="REINFORCE"),
        Line2D([0], [0], color="#1f77b4", linewidth=2, linestyle="--", label="random-640 running max"),
        Line2D([0], [0], color="#e41a1c", linewidth=2, linestyle=":", label="all-ones baseline"),
        Line2D([0], [0], color="#2ca02c", linewidth=2, linestyle="-", label="exhaustive best"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=11, frameon=True)

    fig.suptitle(f"Per-experiment full 4-metric view (α={ALPHA}, N={N})",
                 fontsize=14, weight="bold", y=0.978)

    out = os.path.join(ROOT, "clean_v1_per_exp_all4.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_reward_vs_ceiling(records):
    runs = [r for r in records if r["exh"] and r["reinforce"]]
    if not runs:
        return
    # Sort by exhaustive ceiling for readability
    runs = sorted(runs, key=lambda r: r["exh"]["max"])
    names = [r["name"] for r in runs]
    exh_max = np.array([r["exh"]["max"] for r in runs])
    exh_p95 = np.array([np.percentile(r["exh"]["rewards"], 95) for r in runs])
    exh_p50 = np.array([np.percentile(r["exh"]["rewards"], 50) for r in runs])
    reinf = np.array([r["reinforce"]["best_reward"] for r in runs])
    rand = np.array([r["random_640"]["best"] for r in runs])
    ao = np.array([r["exh"]["all_ones_reward"] for r in runs])

    ys = np.arange(len(runs))
    fig, ax = plt.subplots(1, 1, figsize=(12, max(6, 0.45 * len(runs))))
    ax.barh(ys, exh_max - exh_p50, left=exh_p50, color="#d9f0d9", alpha=0.9,
            label="Exhaustive (p50 → max)")
    ax.plot(exh_max, ys, "s", color="#2ca02c", markersize=8, label="Exhaustive max (ceiling)")
    ax.plot(exh_p95, ys, "|", color="#5fb75f", markersize=12, label="Exhaustive p95")
    ax.plot(reinf, ys, "o", color="tab:blue", markersize=8, label="REINFORCE best")
    ax.plot(rand, ys, "^", color="tab:gray", markersize=7, label="random-640 best")
    ax.plot(ao, ys, "x", color="#e41a1c", markersize=10, label="All-ones")

    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"reward (α={ALPHA})")
    ax.set_title(f"Per-experiment: REINFORCE vs random vs exhaustive ceiling "
                 f"(sorted by ceiling, N={len(runs)})")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    out = os.path.join(ROOT, "clean_v1_reward_vs_ceiling.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    print("Collecting per-experiment data...")
    records = collect_per_experiment()
    print(f"  Found {len(records)} experiments")

    print("\n=== Summary JSON + CSV ===")
    save_summary(records)

    print("\n=== Training curves ===")
    plot_training_curves(records)

    print("\n=== Histograms ===")
    plot_histograms(records)

    print("\n=== Sample efficiency ===")
    plot_sample_efficiency(records)

    print("\n=== Visual grid ===")
    build_visual_grid(records)

    print("\n=== Per-experiment components ===")
    plot_per_experiment_components(records)

    print("\n=== Per-experiment full 4-metric ===")
    plot_per_experiment_all4(records)

    print("\n=== Reward vs ceiling ===")
    plot_reward_vs_ceiling(records)

    print("\n=== All plots done ===")


if __name__ == "__main__":
    main()
