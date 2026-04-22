"""Benchmark many search algorithms against the exhaustive reward oracle.

Uses saved bg_ssim + fg_clip arrays per experiment — rewards are O(1) lookups,
no FLUX needed. Runs full sweep in minutes on CPU via multiprocessing.

Methods compared:
  random                   - uniform Bernoulli(0.5) sampling
  random_prior             - sample from population-mean probs (leave-one-out)
  hill_climb               - random init, flip 1 bit if improves
  hill_climb_prior         - greedy from argmax of prior
  sim_anneal               - simulated annealing with exponential cooling
  reinforce                - vanilla REINFORCE (Bernoulli logits + policy gradient)
  reinforce_prior          - REINFORCE initialized from prior
  cem                      - Cross-Entropy Method (elite=25%, smoothing=0.3)
  cem_prior                - CEM initialized from prior
  thompson                 - Beta-Bernoulli Thompson sampling per bit
  evolutionary             - (μ+λ) Evolution Strategy with bit-flip mutation
  latin_hypercube          - stratified Bernoulli sampling (better coverage)

Output:
  clean_v1/search_sweep/sweep_results.json   # per-method × budget × experiment
  clean_v1/search_sweep/sweep_curves.png     # running-max curves
  clean_v1/search_sweep/sweep_summary.png    # bar chart of final rewards
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1"
EXH_DIR = os.path.join(ROOT, "exhaustive")
OUT_DIR = os.path.join(ROOT, "search_sweep")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.5
N_BITS = 14
TOTAL_MASKS = 1 << N_BITS


# ─────────────────────────────────────────────────────────────
# Reward helpers (analytical from bg/fg components)
# ─────────────────────────────────────────────────────────────
def reward_at(bg_arr, fg_arr, mask_int, alpha=ALPHA):
    bg = bg_arr[mask_int]
    fg = fg_arr[mask_int]
    fg_sig = 1.0 / (1.0 + np.exp(-fg * 10.0))
    return float(np.maximum(bg, 1e-6) ** alpha * np.maximum(fg_sig, 1e-6) ** (1 - alpha))


def reward_all(bg_arr, fg_arr, alpha=ALPHA):
    fg_sig = 1.0 / (1.0 + np.exp(-fg_arr * 10.0))
    return np.maximum(bg_arr, 1e-6) ** alpha * np.maximum(fg_sig, 1e-6) ** (1 - alpha)


def bits_to_int(bits):
    """bits: (n_bits,) np array of 0/1 → int MSB-first."""
    n = len(bits)
    acc = 0
    for i in range(n):
        if bits[i]:
            acc |= (1 << (n - 1 - i))
    return acc


def int_to_bits(n, n_bits):
    bits = np.zeros(n_bits, dtype=np.int32)
    for i in range(n_bits):
        if n & (1 << (n_bits - 1 - i)):
            bits[i] = 1
    return bits


# ─────────────────────────────────────────────────────────────
# Search algorithms — each returns a running-max reward curve of length `budget`
# ─────────────────────────────────────────────────────────────
def random_uniform(R_all, bg, fg, budget, n_bits, seed, prior=None):
    rng = np.random.default_rng(seed)
    ints = rng.integers(0, 1 << n_bits, size=budget)
    rewards = R_all[ints]
    return np.maximum.accumulate(rewards)


def random_prior(R_all, bg, fg, budget, n_bits, seed, prior):
    rng = np.random.default_rng(seed)
    masks = (rng.random((budget, n_bits)) < prior).astype(np.int32)
    # Convert each row to int
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    ints = (masks * weights).sum(axis=1)
    rewards = R_all[ints]
    return np.maximum.accumulate(rewards)


def latin_hypercube(R_all, bg, fg, budget, n_bits, seed, prior=None):
    """Latin-hypercube-style stratified Bernoulli sampling: ensures each bit has
    equal-ish distribution of 0/1 across the batch."""
    rng = np.random.default_rng(seed)
    masks = np.zeros((budget, n_bits), dtype=np.int32)
    for b in range(n_bits):
        p = prior[b] if prior is not None else 0.5
        n_ones = int(round(p * budget))
        col = np.zeros(budget, dtype=np.int32)
        col[:n_ones] = 1
        rng.shuffle(col)
        masks[:, b] = col
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    ints = (masks * weights).sum(axis=1)
    rewards = R_all[ints]
    return np.maximum.accumulate(rewards)


def hill_climb(R_all, bg, fg, budget, n_bits, seed, prior=None, start_greedy=False):
    rng = np.random.default_rng(seed)
    if start_greedy and prior is not None:
        current = (prior > 0.5).astype(np.int32)
    else:
        current = rng.integers(0, 2, n_bits).astype(np.int32)
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    cur_int = int((current * weights).sum())
    cur_r = float(R_all[cur_int])
    rewards = [cur_r]
    while len(rewards) < budget:
        flip_idx = rng.integers(0, n_bits)
        new = current.copy()
        new[flip_idx] ^= 1
        new_int = int((new * weights).sum())
        new_r = float(R_all[new_int])
        rewards.append(new_r)
        if new_r >= cur_r:
            current = new; cur_int = new_int; cur_r = new_r
    return np.maximum.accumulate(np.array(rewards))


def sim_anneal(R_all, bg, fg, budget, n_bits, seed, prior=None, T0=0.05, T_min=1e-3):
    rng = np.random.default_rng(seed)
    current = rng.integers(0, 2, n_bits).astype(np.int32)
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    cur_int = int((current * weights).sum())
    cur_r = float(R_all[cur_int])
    rewards = [cur_r]
    T = T0
    # Cool from T0 to T_min over budget steps
    cooling = (T_min / T0) ** (1.0 / max(budget - 1, 1))
    while len(rewards) < budget:
        flip_idx = rng.integers(0, n_bits)
        new = current.copy()
        new[flip_idx] ^= 1
        new_int = int((new * weights).sum())
        new_r = float(R_all[new_int])
        rewards.append(new_r)
        delta = new_r - cur_r
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-12)):
            current = new; cur_int = new_int; cur_r = new_r
        T *= cooling
    return np.maximum.accumulate(np.array(rewards))


def reinforce(R_all, bg, fg, budget, n_bits, seed, prior=None,
              batch_size=8, lr=0.10, entropy_coeff=0.05, baseline_ema=0.9):
    rng = np.random.default_rng(seed)
    if prior is not None:
        # Clip to avoid log(0)
        p_init = np.clip(prior, 0.02, 0.98)
        logits = np.log(p_init / (1 - p_init))
    else:
        logits = np.zeros(n_bits)
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    rewards_seen = []
    baseline = 0.0
    n_episodes = budget // batch_size
    for ep in range(n_episodes):
        p = 1.0 / (1.0 + np.exp(-logits))
        masks = (rng.random((batch_size, n_bits)) < p).astype(np.int32)
        ints = (masks * weights).sum(axis=1)
        r_vals = R_all[ints]
        for r in r_vals:
            rewards_seen.append(float(r))
        mean_r = float(r_vals.mean())
        baseline = baseline_ema * baseline + (1 - baseline_ema) * mean_r
        adv = r_vals - baseline
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # Policy gradient: ∇log_p = mask * (1-p) - (1-mask) * p
        grad = ((masks - p) * adv[:, None]).mean(axis=0)
        # Entropy gradient: ∇H = -(log(p) - log(1-p)) * p * (1-p) = -logits * p * (1-p)
        ent_grad = -(logits * p * (1 - p))
        logits += lr * (grad + entropy_coeff * ent_grad)
    # Tail: random samples to fill up to budget exactly
    while len(rewards_seen) < budget:
        p = 1.0 / (1.0 + np.exp(-logits))
        masks = (rng.random((min(batch_size, budget - len(rewards_seen)), n_bits)) < p).astype(np.int32)
        ints = (masks * weights).sum(axis=1)
        r_vals = R_all[ints]
        for r in r_vals:
            rewards_seen.append(float(r))
    return np.maximum.accumulate(np.array(rewards_seen[:budget]))


def cem(R_all, bg, fg, budget, n_bits, seed, prior=None,
        batch_size=8, elite_frac=0.25, smoothing=0.3, clip=0.05):
    rng = np.random.default_rng(seed)
    p = prior.copy() if prior is not None else np.full(n_bits, 0.5)
    p = np.clip(p, clip, 1 - clip)
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    elite_k = max(1, int(batch_size * elite_frac))
    rewards_seen = []
    n_iter = budget // batch_size
    for _ in range(n_iter):
        masks = (rng.random((batch_size, n_bits)) < p).astype(np.int32)
        ints = (masks * weights).sum(axis=1)
        r_vals = R_all[ints]
        for r in r_vals:
            rewards_seen.append(float(r))
        idx = np.argsort(-r_vals)[:elite_k]
        empirical_p = masks[idx].mean(axis=0)
        p = (1 - smoothing) * empirical_p + smoothing * p
        p = np.clip(p, clip, 1 - clip)
    while len(rewards_seen) < budget:
        masks = (rng.random((1, n_bits)) < p).astype(np.int32)
        ints = (masks * weights).sum(axis=1)
        rewards_seen.append(float(R_all[ints[0]]))
    return np.maximum.accumulate(np.array(rewards_seen[:budget]))


def thompson(R_all, bg, fg, budget, n_bits, seed, prior=None):
    """Per-bit Beta-Bernoulli Thompson: pull mask, update each bit's posterior by
    whether the reward is above moving mean."""
    rng = np.random.default_rng(seed)
    if prior is not None:
        # Warm-start posterior: prior * k and (1-prior) * k for k=2 pseudo-samples
        alpha_p = prior * 2 + 1
        beta_p = (1 - prior) * 2 + 1
    else:
        alpha_p = np.ones(n_bits)
        beta_p = np.ones(n_bits)
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    rewards = []
    for _ in range(budget):
        p = rng.beta(alpha_p, beta_p)
        mask = (rng.random(n_bits) < p).astype(np.int32)
        mi = int((mask * weights).sum())
        r = float(R_all[mi])
        rewards.append(r)
        if len(rewards) > 1:
            mean_r = float(np.mean(rewards[-20:]))
        else:
            mean_r = r
        signal = max(0.0, r - mean_r)  # only reward above-mean
        for b in range(n_bits):
            if mask[b] == 1:
                alpha_p[b] += signal
            else:
                beta_p[b] += signal
    return np.maximum.accumulate(np.array(rewards))


def evolutionary(R_all, bg, fg, budget, n_bits, seed, prior=None,
                  pop_size=8, mutation_rate=None):
    rng = np.random.default_rng(seed)
    mutation_rate = mutation_rate if mutation_rate is not None else 1.0 / n_bits
    # Init population
    if prior is not None:
        pop = (rng.random((pop_size, n_bits)) < prior).astype(np.int32)
    else:
        pop = rng.integers(0, 2, (pop_size, n_bits)).astype(np.int32)
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    pop_ints = (pop * weights).sum(axis=1)
    pop_rewards = R_all[pop_ints].astype(float)
    rewards_seen = list(pop_rewards)
    while len(rewards_seen) < budget:
        # Select parents (tournament size 2)
        parent_idx = rng.integers(0, pop_size, size=(pop_size, 2))
        tournament_winners = np.where(pop_rewards[parent_idx[:, 0]] > pop_rewards[parent_idx[:, 1]],
                                       parent_idx[:, 0], parent_idx[:, 1])
        # Crossover + mutation
        children = pop[tournament_winners].copy()
        mutation_mask = rng.random(children.shape) < mutation_rate
        children = np.where(mutation_mask, 1 - children, children)
        c_ints = (children * weights).sum(axis=1)
        c_rewards = R_all[c_ints].astype(float)
        for r in c_rewards:
            rewards_seen.append(float(r))
        # Keep top pop_size of combined (μ+λ)
        combined = np.concatenate([pop, children])
        combined_r = np.concatenate([pop_rewards, c_rewards])
        top = np.argsort(-combined_r)[:pop_size]
        pop = combined[top]
        pop_rewards = combined_r[top]
    return np.maximum.accumulate(np.array(rewards_seen[:budget]))


# Registry: name → (function, takes_prior)
METHODS = {
    "random":           (random_uniform,    False),
    "random_prior":     (random_prior,      True),
    "latin":            (latin_hypercube,   False),
    "latin_prior":      (latin_hypercube,   True),
    "hill_climb":       (hill_climb,        False),
    "hill_prior":       (lambda *a, **kw: hill_climb(*a, **kw, start_greedy=True), True),
    "sim_anneal":       (sim_anneal,        False),
    "reinforce":        (reinforce,         False),
    "reinforce_prior":  (reinforce,         True),
    "cem":              (cem,               False),
    "cem_prior":        (cem,               True),
    "thompson":         (thompson,          False),
    "thompson_prior":   (thompson,          True),
    "evolutionary":     (evolutionary,      False),
    "evolutionary_prior": (evolutionary,    True),
}


# ─────────────────────────────────────────────────────────────
# Sweep runner (one job = one method × experiment × seed × budget)
# ─────────────────────────────────────────────────────────────
def run_one(job):
    """job = (method_name, exp_name, seed, budget, bg_path, fg_path, prior_or_none)"""
    method_name, exp_name, seed, budget, bg_path, fg_path, prior = job
    bg = np.load(bg_path)
    fg = np.load(fg_path)
    R_all = reward_all(bg, fg, ALPHA)
    fn, takes_prior = METHODS[method_name]
    kwargs = {}
    if takes_prior and prior is not None:
        kwargs["prior"] = prior
    curve = fn(R_all, bg, fg, budget, N_BITS, seed, **kwargs)
    # Record sparse budget points + final
    sparse_budgets = [1, 8, 16, 40, 80, 160, 320, 640]
    points = {}
    for b in sparse_budgets:
        if b <= len(curve):
            points[str(b)] = float(curve[b - 1])
    ceiling = float(R_all.max())
    return {
        "method": method_name,
        "exp": exp_name,
        "seed": seed,
        "points": points,
        "ceiling": ceiling,
    }


def compute_prior_leave_one_out(exp_names, bg_paths, fg_paths):
    """For each experiment, return the average 'bit-frequency in exhaustive-top-64' of
    the OTHER experiments, as a prior on bit probability."""
    # For each exp, compute the mean mask-probability over its top-K exhaustive masks
    per_exp_prior = {}
    weights = 1 << np.arange(N_BITS - 1, -1, -1)
    K = 64
    for name, bp, fp in zip(exp_names, bg_paths, fg_paths):
        bg = np.load(bp); fg = np.load(fp)
        R = reward_all(bg, fg, ALPHA)
        top_idx = np.argsort(-R)[:K]
        bits = np.stack([int_to_bits(i, N_BITS) for i in top_idx])
        per_exp_prior[name] = bits.mean(axis=0)
    # Leave-one-out aggregation
    priors = {}
    for name in exp_names:
        others = [v for k, v in per_exp_prior.items() if k != name]
        priors[name] = np.mean(others, axis=0)
    return priors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=640)
    ap.add_argument("--workers", type=int, default=120)
    args = ap.parse_args()

    exp_names = sorted(d for d in os.listdir(EXH_DIR)
                       if os.path.isdir(os.path.join(EXH_DIR, d))
                       and os.path.isfile(os.path.join(EXH_DIR, d, "bg_ssim.npy")))
    bg_paths = [os.path.join(EXH_DIR, n, "bg_ssim.npy") for n in exp_names]
    fg_paths = [os.path.join(EXH_DIR, n, "fg_clip.npy") for n in exp_names]
    print(f"Using {len(exp_names)} experiments with exhaustive data")

    print("Computing leave-one-out priors from exhaustive top-64...")
    priors = compute_prior_leave_one_out(exp_names, bg_paths, fg_paths)

    # Build jobs
    jobs = []
    for method_name, (_, takes_prior) in METHODS.items():
        for name, bp, fp in zip(exp_names, bg_paths, fg_paths):
            prior = priors[name] if takes_prior else None
            for seed in range(args.seeds):
                jobs.append((method_name, name, seed, args.budget, bp, fp, prior))
    print(f"Total jobs: {len(jobs)}  (methods={len(METHODS)}, exps={len(exp_names)}, "
          f"seeds={args.seeds})")

    # Run pool
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(run_one, jobs, chunksize=4)):
            results.append(res)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(jobs)}] "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"All done in {time.perf_counter() - t0:.0f}s")

    # Save raw results
    with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Aggregate: method × budget → mean, std (averaged over experiments × seeds)
    budgets = [1, 8, 16, 40, 80, 160, 320, 640]
    agg = {m: {b: [] for b in budgets} for m in METHODS}
    ceiling_per_exp = {}
    for res in results:
        ceiling_per_exp[res["exp"]] = res["ceiling"]
        for b_str, v in res["points"].items():
            b = int(b_str)
            if b in budgets:
                # Normalize by per-experiment ceiling for fair aggregation
                agg[res["method"]][b].append(v / res["ceiling"])

    # Plot: running max reward (normalized) vs budget, one line per method
    fig, ax = plt.subplots(1, 1, figsize=(13, 8))
    # Order methods so "prior" variants are easy to find
    method_order = ["random", "random_prior", "latin", "latin_prior",
                    "reinforce", "reinforce_prior", "cem", "cem_prior",
                    "hill_climb", "hill_prior", "sim_anneal",
                    "thompson", "thompson_prior",
                    "evolutionary", "evolutionary_prior"]
    cmap = plt.cm.tab20(np.linspace(0, 1, len(method_order)))
    for i, m in enumerate(method_order):
        means, stds = [], []
        for b in budgets:
            v = agg[m][b]
            means.append(np.mean(v) if v else np.nan)
            stds.append(np.std(v) if v else np.nan)
        means = np.array(means); stds = np.array(stds)
        valid = ~np.isnan(means)
        style = "-" if "prior" not in m else "--"
        lw = 2 if m in ("random", "reinforce", "cem", "thompson") else 1.5
        ax.plot(np.array(budgets)[valid], means[valid], style, color=cmap[i],
                linewidth=lw, markersize=6, marker="o", label=m)

    ax.axhline(1.0, color="green", linewidth=2, linestyle="-",
               label=f"exhaustive ceiling (normalized = 1.0)")
    ax.set_xscale("log")
    ax.set_xlabel("Budget (images)")
    ax.set_ylabel("Best reward / exhaustive ceiling (avg over experiments × seeds)")
    ax.set_title(f"Search methods comparison (α={ALPHA}, {N_BITS}-bit, N_exp=8, seeds=5)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "sweep_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Summary bar chart: final reward (budget=640) per method
    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    final_means = []
    final_stds = []
    for m in method_order:
        v = agg[m][640] if 640 in agg[m] else agg[m][max(agg[m].keys())]
        final_means.append(np.mean(v) if v else 0)
        final_stds.append(np.std(v) if v else 0)
    ys = np.arange(len(method_order))
    order = np.argsort(final_means)
    ax.barh(ys, np.array(final_means)[order], xerr=np.array(final_stds)[order],
            color=[cmap[i] for i in order])
    ax.set_yticks(ys)
    ax.set_yticklabels([method_order[i] for i in order])
    ax.axvline(1.0, color="green", linewidth=2, linestyle="-", label="ceiling")
    # Reference: random at budget=640
    random_val = np.mean(agg["random"][640])
    ax.axvline(random_val, color="#1f77b4", linewidth=2, linestyle="--",
               label=f"random ({random_val:.4f})")
    ax.set_xlim(0.9, 1.01)
    ax.set_xlabel("Best reward / ceiling @ budget=640 (avg × seeds)")
    ax.set_title(f"Method ranking at budget=640 (sorted)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "sweep_summary.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # Print numerical summary
    print("\n=== Mean reward/ceiling by method × budget ===")
    print(f"{'method':<22} " + " ".join(f"{b:>8}" for b in budgets))
    for m in method_order:
        line = f"{m:<22} "
        for b in budgets:
            v = agg[m][b]
            line += f" {np.mean(v):>7.4f}" if v else " " * 8
        print(line)


if __name__ == "__main__":
    main()
