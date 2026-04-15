"""
REINFORCE hyperparameter sweep on a single prompt pair.

Loads the FLUX generator + SigLIP reward computer ONCE, then sequentially
runs multiple REINFORCE configs (varying alpha, lr, entropy_coeff, etc.)
and two baselines:

  - all-ones mask: single reward value, drawn as a horizontal dashed line
  - random search: samples N random masks, plots running-max mean_reward
    as a function of cumulative images evaluated

All curves land on one matplotlib plot for direct comparison.

Usage:
  python generation/reinforce_sweep.py \
      --source_prompt "a photo of a cat" \
      --target_prompt "a photo of a dog" \
      --seg_prompt "cat" \
      --output_dir /path/to/sweep_output \
      --gpu 0 \
      --episodes_per_config 80 \
      --random_samples 400
"""
import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.distributions import Bernoulli

import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Reuse the shared components from reinforce_search.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reinforce_search import (
    BernoulliPolicy,
    DiffusionGenerator,
    RewardComputer,
    compute_segmentation,
    mask_to_int_batch,
)


# ────────────────────────────────────────────
# Default sweep — edit this list to sweep different hyperparameters.
# Each config gets its own REINFORCE run; they all share the same
# generator + reward computer + initial noise.
# ────────────────────────────────────────────
DEFAULT_SWEEP = [
    {"name": "default",     "lr": 0.1,  "alpha": 0.5, "entropy_coeff": 0.05},
    {"name": "alpha_low",   "lr": 0.1,  "alpha": 0.3, "entropy_coeff": 0.05},
    {"name": "alpha_high",  "lr": 0.1,  "alpha": 0.7, "entropy_coeff": 0.05},
    {"name": "lr_low",      "lr": 0.05, "alpha": 0.5, "entropy_coeff": 0.05},
    {"name": "lr_high",     "lr": 0.2,  "alpha": 0.5, "entropy_coeff": 0.05},
    {"name": "no_entropy",  "lr": 0.1,  "alpha": 0.5, "entropy_coeff": 0.0},
    {"name": "high_entropy","lr": 0.1,  "alpha": 0.5, "entropy_coeff": 0.10},
]


def train_one_config(generator, reward_computer, config, episodes, batch_size, n_bits, device):
    """Run one REINFORCE training loop. Returns dict with history arrays."""
    name = config["name"]
    lr = config["lr"]
    alpha = config["alpha"]
    ent_c = config["entropy_coeff"]
    baseline_ema = config.get("baseline_ema", 0.9)

    print(f"\n--- Config: {name} (lr={lr}, alpha={alpha}, ent={ent_c}) ---", flush=True)

    policy = BernoulliPolicy(n_bits, init_logit=0.0, device=device)
    optimizer = torch.optim.Adam([policy.logits], lr=lr)

    baseline = 0.0
    best_reward = -float("inf")
    best_mask = None
    mean_rewards = []
    best_rewards_ever = []
    bg_ssims = []
    fg_clips = []
    entropies = []

    t0 = time.perf_counter()
    for ep in range(episodes):
        masks, log_probs = policy.sample(batch_size)
        with torch.no_grad():
            images = generator.generate(masks)
            rewards, bg, fg = reward_computer.compute_rewards(images, alpha=alpha)

        mean_r = rewards.mean().item()
        mean_rewards.append(mean_r)
        bg_ssims.append(bg.mean().item())
        fg_clips.append(fg.mean().item())

        baseline = baseline_ema * baseline + (1 - baseline_ema) * mean_r
        advantage = rewards - baseline
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        policy_loss = -(advantage.detach() * log_probs).mean()
        entropy = policy.entropy()
        loss = policy_loss - ent_c * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        entropies.append(entropy.item())

        # Track best
        batch_best_idx = rewards.argmax().item()
        if rewards[batch_best_idx] > best_reward:
            best_reward = rewards[batch_best_idx].item()
            best_mask = masks[batch_best_idx].detach().clone()
        best_rewards_ever.append(best_reward)

        if ep % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{ep:3d}/{episodes}] R={mean_r:.4f} best={best_reward:.4f} "
                  f"bg={bg.mean():.3f} fg={fg.mean():.3f} H={entropy.item():.2f} "
                  f"({elapsed:.0f}s)", flush=True)

    print(f"  done: best_reward={best_reward:.4f}, total_time={time.perf_counter()-t0:.0f}s", flush=True)

    return {
        "name": name,
        "config": config,
        "mean_rewards": np.array(mean_rewards),
        "best_rewards_ever": np.array(best_rewards_ever),
        "bg_ssims": np.array(bg_ssims),
        "fg_clips": np.array(fg_clips),
        "entropies": np.array(entropies),
        "best_mask": best_mask.cpu().numpy().astype(int) if best_mask is not None else None,
        "best_reward": best_reward,
    }


def random_baseline(generator, reward_computer, n_samples, batch_size, n_bits, alpha, device):
    """Sample N random masks, evaluate all, return running-max mean_reward by batch."""
    print(f"\n--- Random baseline ({n_samples} samples) ---", flush=True)
    all_rewards = []
    t0 = time.perf_counter()
    n_batches = (n_samples + batch_size - 1) // batch_size

    for bi in range(n_batches):
        b = min(batch_size, n_samples - bi * batch_size)
        masks = torch.bernoulli(torch.full((b, n_bits), 0.5, device=device))
        with torch.no_grad():
            images = generator.generate(masks)
            rewards, _, _ = reward_computer.compute_rewards(images, alpha=alpha)
        all_rewards.append(rewards.cpu().numpy())
        if bi % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [batch {bi:3d}/{n_batches}] ({elapsed:.0f}s)", flush=True)

    all_rewards = np.concatenate(all_rewards)  # (n_samples,)
    print(f"  done: min={all_rewards.min():.4f}, max={all_rewards.max():.4f}, "
          f"mean={all_rewards.mean():.4f}, time={time.perf_counter()-t0:.0f}s", flush=True)

    # Compute running max and per-batch mean so they can be plotted vs episode
    running_max = np.maximum.accumulate(all_rewards)
    # Per-batch mean (groups of batch_size)
    batch_means = []
    for bi in range(n_batches):
        chunk = all_rewards[bi * batch_size: (bi + 1) * batch_size]
        batch_means.append(chunk.mean())
    batch_means = np.array(batch_means)
    # Running max per-batch
    running_max_per_batch = np.maximum.accumulate(batch_means)

    return {
        "all_rewards": all_rewards,
        "running_max": running_max,               # (n_samples,)
        "batch_means": batch_means,               # (n_batches,)
        "running_max_per_batch": running_max_per_batch,
        "best": all_rewards.max(),
    }


def all_ones_baseline(generator, reward_computer, n_bits, alpha, device):
    """Evaluate the all-ones mask (pure target prompt). Returns scalar reward."""
    print(f"\n--- All-ones baseline (b=all 1) ---", flush=True)
    ones_mask = torch.ones(1, n_bits, device=device)
    with torch.no_grad():
        img = generator.generate(ones_mask)
        reward, bg, fg = reward_computer.compute_rewards(img, alpha=alpha)
    val = reward.item()
    print(f"  reward={val:.4f}  bg={bg.item():.4f}  fg={fg.item():.4f}", flush=True)
    return {
        "reward": val,
        "bg_ssim": bg.item(),
        "fg_clip": fg.item(),
        "image": img[0].cpu(),
    }


def plot_sweep(results, random_res, ones_res, out_path, title):
    """Build the comparison plot: mean_reward per episode + baselines."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # ── Panel 1: mean_reward per episode ──
    ax = axes[0, 0]
    for r, c in zip(results, colors):
        ax.plot(r["mean_rewards"], label=r["name"], color=c, linewidth=1.5, alpha=0.85)
    # Random baseline: per-batch mean (noisy, but shows untrained reward level)
    if random_res is not None:
        ax.plot(random_res["batch_means"], label=f"random (per-batch)",
                color="gray", linewidth=1, alpha=0.5, linestyle=":")
    # All-ones: horizontal reference
    if ones_res is not None:
        ax.axhline(ones_res["reward"], label=f"all-ones ({ones_res['reward']:.3f})",
                   color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("mean_reward (per batch)")
    ax.set_title("Mean reward per episode")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # ── Panel 2: best_reward_ever ──
    ax = axes[0, 1]
    for r, c in zip(results, colors):
        ax.plot(r["best_rewards_ever"], label=r["name"], color=c, linewidth=1.5)
    # Random: running max per batch (direct comparison to best-so-far)
    if random_res is not None:
        n = len(random_res["running_max_per_batch"])
        ax.plot(range(n), random_res["running_max_per_batch"],
                label=f"random running-max", color="gray", linewidth=1.5, linestyle="--")
    if ones_res is not None:
        ax.axhline(ones_res["reward"], label=f"all-ones ({ones_res['reward']:.3f})",
                   color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Episode / Batch index")
    ax.set_ylabel("best_reward_ever")
    ax.set_title("Best-so-far reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # ── Panel 3: mean fg_clip ──
    ax = axes[1, 0]
    for r, c in zip(results, colors):
        ax.plot(r["fg_clips"], label=r["name"], color=c, linewidth=1.5, alpha=0.85)
    if ones_res is not None:
        ax.axhline(ones_res["fg_clip"], label=f"all-ones fg ({ones_res['fg_clip']:+.3f})",
                   color="black", linewidth=1.5, linestyle="--")
    ax.axhline(0, color="red", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("mean fg_clip_score (raw)")
    ax.set_title("Foreground CLIP score (per episode)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # ── Panel 4: policy entropy ──
    ax = axes[1, 1]
    for r, c in zip(results, colors):
        ax.plot(r["entropies"], label=r["name"], color=c, linewidth=1.5, alpha=0.85)
    ax.axhline(14 * np.log(2), color="red", linewidth=0.5, linestyle=":", label="max entropy (14 bits)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("policy entropy H(π)")
    ax.set_title("Policy entropy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    plt.suptitle(title, fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="REINFORCE hyperparameter sweep on one prompt pair")
    p.add_argument("--source_prompt", required=True)
    p.add_argument("--target_prompt", required=True)
    p.add_argument("--seg_prompt", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_bits", type=int, default=14)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--mask", default=None, help="Pre-computed background mask (.npy)")
    p.add_argument("--vision_model", default="google/siglip2-so400m-patch14-384")
    # Sweep-specific
    p.add_argument("--episodes_per_config", type=int, default=80,
                   help="How many episodes each config runs (fixed, no early stop)")
    p.add_argument("--random_samples", type=int, default=400,
                   help="Number of random masks for the random-search baseline")
    p.add_argument("--config_file", default=None,
                   help="Optional JSON file with a list of configs to sweep. "
                        "Each config is a dict with keys: name, lr, alpha, entropy_coeff.")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    os.makedirs(args.output_dir, exist_ok=True)

    n_bits = args.n_bits
    steps = args.steps if args.steps else n_bits * 2

    # ── Phase 1: load generator ──
    print("=== Phase 1: load diffusion generator ===", flush=True)
    generator = DiffusionGenerator(
        device=device,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        height=args.height, width=args.width,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        n_bits=n_bits, steps=steps,
    )

    # ── Phase 2: source image + segmentation ──
    print("\n=== Phase 2: source image + segmentation ===", flush=True)
    zeros_mask = torch.zeros(1, n_bits, device=device)
    source_img = generator.generate(zeros_mask)

    if args.mask:
        print(f"  Using pre-computed mask: {args.mask}")
        bg_mask = np.load(args.mask)
    else:
        seg_prompt = args.seg_prompt or args.source_prompt
        bg_mask = compute_segmentation(source_img, seg_prompt, device)

    # Save source
    src_pil = Image.fromarray(
        (source_img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    src_pil.save(os.path.join(args.output_dir, "source_b0.jpg"), quality=95)

    # ── Phase 3: reward computer ──
    print("\n=== Phase 3: reward computer ===", flush=True)
    reward_computer = RewardComputer(
        device=device,
        source_image=source_img,
        bg_mask=bg_mask,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        img_size=args.height,
        vision_model=args.vision_model,
    )

    # ── Phase 4: baselines ──
    print("\n=== Phase 4: baselines ===", flush=True)
    ones_res = all_ones_baseline(generator, reward_computer, n_bits, 0.5, device)
    # Save the target image
    tgt_pil = Image.fromarray(
        (ones_res["image"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8))
    target_b = (1 << n_bits) - 1
    tgt_pil.save(os.path.join(args.output_dir, f"target_b{target_b}.jpg"), quality=95)

    random_res = random_baseline(
        generator, reward_computer,
        n_samples=args.random_samples,
        batch_size=args.batch_size,
        n_bits=n_bits, alpha=0.5, device=device,
    )

    # ── Phase 5: config sweep ──
    print("\n=== Phase 5: config sweep ===", flush=True)
    if args.config_file:
        with open(args.config_file) as f:
            configs = json.load(f)
    else:
        configs = DEFAULT_SWEEP
    print(f"  {len(configs)} configs to run")

    results = []
    for cfg in configs:
        r = train_one_config(
            generator, reward_computer, cfg,
            episodes=args.episodes_per_config,
            batch_size=args.batch_size,
            n_bits=n_bits, device=device,
        )
        results.append(r)

    # ── Phase 6: save everything ──
    print("\n=== Phase 6: save results ===", flush=True)
    # Save raw numpy
    np.savez(
        os.path.join(args.output_dir, "sweep_results.npz"),
        **{f"{r['name']}_mean_rewards": r["mean_rewards"] for r in results},
        **{f"{r['name']}_best_rewards": r["best_rewards_ever"] for r in results},
        **{f"{r['name']}_fg_clips": r["fg_clips"] for r in results},
        **{f"{r['name']}_entropies": r["entropies"] for r in results},
        random_rewards=random_res["all_rewards"],
        random_running_max=random_res["running_max"],
        ones_reward=np.array([ones_res["reward"]]),
    )
    # Save config + summary JSON
    summary = {
        "configs": [r["config"] | {"best_reward": r["best_reward"]} for r in results],
        "all_ones": ones_res["reward"],
        "random_best": float(random_res["best"]),
        "random_mean": float(random_res["all_rewards"].mean()),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    short_src = args.source_prompt[:40] + "..." if len(args.source_prompt) > 40 else args.source_prompt
    short_tgt = args.target_prompt[:40] + "..." if len(args.target_prompt) > 40 else args.target_prompt
    title = f"{short_src} → {short_tgt}"
    plot_sweep(results, random_res, ones_res,
               os.path.join(args.output_dir, "sweep_curves.png"), title)

    print(f"\nAll done. Results in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
