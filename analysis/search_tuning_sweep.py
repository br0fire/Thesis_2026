"""Hyperparameter tuning sweep for search methods.

Extends search_methods_sweep by testing many variants of key methods +
hybrid combinations. All on CPU via lookup → seconds.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_methods_sweep import (
    reward_all, random_uniform, random_prior, latin_hypercube, hill_climb,
    sim_anneal, reinforce, cem, thompson, evolutionary,
    int_to_bits, compute_prior_leave_one_out,
    ALPHA, N_BITS, EXH_DIR,
)

OUT_DIR = "/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1/search_tuning"
os.makedirs(OUT_DIR, exist_ok=True)


# ────────────────────────────────────────────
# Hybrid methods
# ────────────────────────────────────────────
def cem_then_reinforce(R_all, bg, fg, budget, n_bits, seed, prior=None,
                       batch_size=8, split=0.5, **cem_kw):
    """Run CEM for first half of budget, then continue with REINFORCE warm-started."""
    b1 = int(budget * split); b2 = budget - b1
    # Run CEM, capture final probs
    rng = np.random.default_rng(seed)
    p = prior.copy() if prior is not None else np.full(n_bits, 0.5)
    p = np.clip(p, 0.05, 0.95)
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    rewards_seen = []
    elite_k = max(1, int(batch_size * 0.25))
    smoothing = 0.1  # aggressive
    n_iter = b1 // batch_size
    for _ in range(n_iter):
        masks = (rng.random((batch_size, n_bits)) < p).astype(np.int32)
        ints = (masks * weights).sum(axis=1)
        r_vals = R_all[ints]
        rewards_seen.extend(r_vals.tolist())
        idx = np.argsort(-r_vals)[:elite_k]
        empirical_p = masks[idx].mean(axis=0)
        p = (1 - smoothing) * empirical_p + smoothing * p
        p = np.clip(p, 0.05, 0.95)
    # Then REINFORCE from final p
    curve_rf = reinforce(R_all, bg, fg, b2, n_bits, seed + 1, prior=p,
                         batch_size=batch_size, lr=0.10, entropy_coeff=0.05)
    # Concatenate
    rewards_seen.extend(curve_rf.tolist())
    return np.maximum.accumulate(np.array(rewards_seen[:budget]))


def hill_prior_then_reinforce(R_all, bg, fg, budget, n_bits, seed, prior,
                               batch_size=8, hill_budget=16):
    """Hill-climb from prior for hill_budget steps, then REINFORCE from learned mask."""
    # Run hill_prior
    rewards = hill_climb(R_all, bg, fg, hill_budget, n_bits, seed,
                         prior=prior, start_greedy=True)
    # Use final best mask position as prior for REINFORCE
    # We reconstruct: start with greedy prior, note that hill_climb has moved current
    # Simplest: use prior directly as REINFORCE init
    remaining = budget - hill_budget
    if remaining <= 0:
        return rewards[:budget]
    curve_rf = reinforce(R_all, bg, fg, remaining, n_bits, seed + 1, prior=prior,
                         batch_size=batch_size, lr=0.10, entropy_coeff=0.05)
    combined = np.concatenate([rewards, curve_rf])
    return np.maximum.accumulate(combined[:budget])


def evolutionary_prior_hill(R_all, bg, fg, budget, n_bits, seed, prior,
                             pop_size=8, hill_per_child=1):
    """Evolutionary with hill-climb refinement on each child."""
    rng = np.random.default_rng(seed)
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    pop = (rng.random((pop_size, n_bits)) < prior).astype(np.int32)
    pop_ints = (pop * weights).sum(axis=1)
    pop_rewards = R_all[pop_ints].astype(float)
    rewards_seen = list(pop_rewards)
    while len(rewards_seen) < budget:
        # Tournament
        parent_idx = rng.integers(0, pop_size, size=(pop_size, 2))
        winners = np.where(pop_rewards[parent_idx[:, 0]] > pop_rewards[parent_idx[:, 1]],
                           parent_idx[:, 0], parent_idx[:, 1])
        children = pop[winners].copy()
        mut_mask = rng.random(children.shape) < (1.0 / n_bits)
        children = np.where(mut_mask, 1 - children, children)
        c_ints = (children * weights).sum(axis=1)
        c_rewards = R_all[c_ints].astype(float)
        rewards_seen.extend(c_rewards.tolist())
        # Hill-climb refinement per child
        for j in range(pop_size):
            if len(rewards_seen) >= budget:
                break
            for _ in range(hill_per_child):
                if len(rewards_seen) >= budget:
                    break
                flip = rng.integers(0, n_bits)
                new = children[j].copy()
                new[flip] ^= 1
                new_int = int((new * weights).sum())
                new_r = float(R_all[new_int])
                rewards_seen.append(new_r)
                if new_r >= c_rewards[j]:
                    children[j] = new
                    c_ints[j] = new_int
                    c_rewards[j] = new_r
        combined = np.concatenate([pop, children])
        combined_r = np.concatenate([pop_rewards, c_rewards])
        top = np.argsort(-combined_r)[:pop_size]
        pop = combined[top]; pop_rewards = combined_r[top]
    return np.maximum.accumulate(np.array(rewards_seen[:budget]))


# ────────────────────────────────────────────
# Registry of tuned variants
# ────────────────────────────────────────────
METHODS_TUNED = {}

# REINFORCE lr sweep (no prior)
for lr in [0.05, 0.10, 0.20, 0.50]:
    METHODS_TUNED[f"reinforce_lr{lr}"] = (
        lambda R, bg, fg, bu, nb, sd, _lr=lr: reinforce(R, bg, fg, bu, nb, sd, lr=_lr), False)

# REINFORCE entropy sweep (no prior)
for ec in [0.0, 0.01, 0.05, 0.20]:
    METHODS_TUNED[f"reinforce_ent{ec}"] = (
        lambda R, bg, fg, bu, nb, sd, _ec=ec: reinforce(R, bg, fg, bu, nb, sd, entropy_coeff=_ec), False)

# REINFORCE_prior with varying lr
for lr in [0.05, 0.10, 0.20]:
    METHODS_TUNED[f"reinforce_prior_lr{lr}"] = (
        lambda R, bg, fg, bu, nb, sd, prior, _lr=lr:
            reinforce(R, bg, fg, bu, nb, sd, prior=prior, lr=_lr), True)

# REINFORCE_prior with varying entropy
for ec in [0.0, 0.05, 0.20]:
    METHODS_TUNED[f"reinforce_prior_ent{ec}"] = (
        lambda R, bg, fg, bu, nb, sd, prior, _ec=ec:
            reinforce(R, bg, fg, bu, nb, sd, prior=prior, entropy_coeff=_ec), True)

# CEM smoothing sweep (no prior)
for sm in [0.0, 0.1, 0.3, 0.5, 0.7]:
    METHODS_TUNED[f"cem_sm{sm}"] = (
        lambda R, bg, fg, bu, nb, sd, _sm=sm: cem(R, bg, fg, bu, nb, sd, smoothing=_sm), False)

# CEM elite_frac sweep (no prior)
for ef in [0.125, 0.25, 0.5, 0.75]:
    METHODS_TUNED[f"cem_ef{ef}"] = (
        lambda R, bg, fg, bu, nb, sd, _ef=ef: cem(R, bg, fg, bu, nb, sd, elite_frac=_ef), False)

# CEM_prior variants
for sm in [0.1, 0.3, 0.5]:
    METHODS_TUNED[f"cem_prior_sm{sm}"] = (
        lambda R, bg, fg, bu, nb, sd, prior, _sm=sm:
            cem(R, bg, fg, bu, nb, sd, prior=prior, smoothing=_sm), True)

# Sim_anneal T0 sweep
for T0 in [0.02, 0.05, 0.10, 0.20]:
    METHODS_TUNED[f"sa_T{T0}"] = (
        lambda R, bg, fg, bu, nb, sd, _T0=T0: sim_anneal(R, bg, fg, bu, nb, sd, T0=_T0), False)

# Evolutionary pop_size sweep (no prior)
for ps in [4, 8, 16, 32]:
    METHODS_TUNED[f"ev_pop{ps}"] = (
        lambda R, bg, fg, bu, nb, sd, _ps=ps: evolutionary(R, bg, fg, bu, nb, sd, pop_size=_ps), False)

# Evolutionary mutation rate sweep (no prior)
for mr_mul in [0.5, 1.0, 2.0, 4.0]:
    METHODS_TUNED[f"ev_mut{mr_mul}x"] = (
        lambda R, bg, fg, bu, nb, sd, _m=mr_mul:
            evolutionary(R, bg, fg, bu, nb, sd, mutation_rate=_m / N_BITS), False)

# Evolutionary_prior pop_size sweep
for ps in [4, 8, 16, 32]:
    METHODS_TUNED[f"ev_prior_pop{ps}"] = (
        lambda R, bg, fg, bu, nb, sd, prior, _ps=ps:
            evolutionary(R, bg, fg, bu, nb, sd, prior=prior, pop_size=_ps), True)

# HYBRID: CEM → REINFORCE
for split in [0.25, 0.5, 0.75]:
    METHODS_TUNED[f"cem→reinforce_{split}"] = (
        lambda R, bg, fg, bu, nb, sd, _sp=split:
            cem_then_reinforce(R, bg, fg, bu, nb, sd, split=_sp), False)

# HYBRID: hill_prior → REINFORCE
for hb in [8, 16, 32, 64]:
    METHODS_TUNED[f"hill→rein_{hb}"] = (
        lambda R, bg, fg, bu, nb, sd, prior, _hb=hb:
            hill_prior_then_reinforce(R, bg, fg, bu, nb, sd, prior=prior, hill_budget=_hb), True)

# HYBRID: evolutionary_prior with hill refinement
for hpc in [1, 2, 3]:
    METHODS_TUNED[f"ev_prior_hill{hpc}"] = (
        lambda R, bg, fg, bu, nb, sd, prior, _h=hpc:
            evolutionary_prior_hill(R, bg, fg, bu, nb, sd, prior=prior, hill_per_child=_h), True)


# ────────────────────────────────────────────
# Runner (same as search_methods_sweep)
# ────────────────────────────────────────────
def run_one(job):
    method_name, exp_name, seed, budget, bg_path, fg_path, prior = job
    bg = np.load(bg_path); fg = np.load(fg_path)
    R_all = reward_all(bg, fg, ALPHA)
    fn, takes_prior = METHODS_TUNED[method_name]
    if takes_prior and prior is not None:
        curve = fn(R_all, bg, fg, budget, N_BITS, seed, prior)
    else:
        curve = fn(R_all, bg, fg, budget, N_BITS, seed)
    sparse_budgets = [1, 8, 16, 40, 80, 160, 320, 640]
    points = {str(b): float(curve[b - 1]) for b in sparse_budgets if b <= len(curve)}
    return {"method": method_name, "exp": exp_name, "seed": seed,
            "points": points, "ceiling": float(R_all.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=640)
    ap.add_argument("--workers", type=int, default=120)
    args = ap.parse_args()

    exp_names = sorted(d for d in os.listdir(EXH_DIR)
                       if os.path.isdir(os.path.join(EXH_DIR, d))
                       and os.path.isfile(os.path.join(EXH_DIR, d, "bg_ssim.npy")))
    bg_paths = [os.path.join(EXH_DIR, n, "bg_ssim.npy") for n in exp_names]
    fg_paths = [os.path.join(EXH_DIR, n, "fg_clip.npy") for n in exp_names]
    print(f"Using {len(exp_names)} experiments")

    priors = compute_prior_leave_one_out(exp_names, bg_paths, fg_paths)

    jobs = []
    for method_name, (_, takes_prior) in METHODS_TUNED.items():
        for name, bp, fp in zip(exp_names, bg_paths, fg_paths):
            prior = priors[name] if takes_prior else None
            for seed in range(args.seeds):
                jobs.append((method_name, name, seed, args.budget, bp, fp, prior))
    print(f"Total jobs: {len(jobs)}  "
          f"({len(METHODS_TUNED)} methods × {len(exp_names)} exps × {args.seeds} seeds)")

    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(run_one, jobs, chunksize=4)):
            results.append(res)
    print(f"All done in {time.perf_counter() - t0:.0f}s")

    with open(os.path.join(OUT_DIR, "tuning_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    budgets = [1, 8, 16, 40, 80, 160, 320, 640]
    agg = {m: {b: [] for b in budgets} for m in METHODS_TUNED}
    for res in results:
        for b_str, v in res["points"].items():
            b = int(b_str)
            if b in budgets:
                agg[res["method"]][b].append(v / res["ceiling"])

    # Print summary: method × budget
    print("\n=== Mean reward/ceiling by method × budget ===")
    print(f"{'method':<24} " + " ".join(f"{b:>8}" for b in budgets))
    sorted_methods = sorted(METHODS_TUNED.keys(),
                            key=lambda m: -np.mean(agg[m][640]) if agg[m][640] else 0)
    for m in sorted_methods:
        line = f"{m:<24} "
        for b in budgets:
            v = agg[m][b]
            line += f" {np.mean(v):>7.4f}" if v else " " * 8
        print(line)

    # Plot top-15 methods + baseline random from original sweep
    top_methods = sorted_methods[:15]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(top_methods)))
    for i, m in enumerate(top_methods):
        means = [np.mean(agg[m][b]) if agg[m][b] else np.nan for b in budgets]
        ax.plot(budgets, means, "o-", color=cmap[i], linewidth=1.5, markersize=5, label=m)
    ax.axhline(1.0, color="green", linewidth=2, linestyle="-", label="ceiling")
    # Horizontal line = random@640 for reference (0.9918 from earlier sweep)
    ax.axhline(0.9918, color="#1f77b4", linewidth=2, linestyle="--",
               label="random@640=0.9918 (ref)")
    ax.set_xscale("log")
    ax.set_xlabel("Budget (images)"); ax.set_ylabel("reward / ceiling")
    ax.set_title(f"TOP-15 tuned methods (α={ALPHA}, {N_BITS}-bit, N_exp=8, seeds={args.seeds})")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tuning_top15.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR}/tuning_top15.png")

    # Summary bar chart: all methods sorted by final reward
    fig, ax = plt.subplots(1, 1, figsize=(12, max(8, 0.25 * len(METHODS_TUNED))))
    final = [(m, np.mean(agg[m][640]) if agg[m][640] else 0) for m in sorted_methods]
    final.reverse()  # smallest at bottom, largest at top
    ys = np.arange(len(final))
    cmap_big = plt.cm.tab20(np.linspace(0, 1, len(final)))
    ax.barh(ys, [v for _, v in final], color=cmap_big)
    ax.set_yticks(ys); ax.set_yticklabels([m for m, _ in final], fontsize=7)
    ax.axvline(1.0, color="green", linewidth=2, linestyle="-", label="ceiling")
    ax.axvline(0.9918, color="#1f77b4", linewidth=2, linestyle="--", label="random@640")
    ax.set_xlim(0.95, 1.01); ax.set_xlabel("reward/ceiling @ budget=640")
    ax.set_title("All tuned methods ranked")
    ax.grid(alpha=0.3, axis="x"); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tuning_ranking.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR}/tuning_ranking.png")


if __name__ == "__main__":
    main()
