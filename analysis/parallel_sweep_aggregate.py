"""Aggregate results from a parallel hyperparameter sweep.

Reads per-config ``reinforce_log.csv`` files written by ``generation/reinforce_search.py``
(launched in parallel by ``scripts/parallel_sweep.sh``), then — on a single GPU — computes
the two baselines (all-ones mask + random search) using the same FLUX + SigLIP 2 pipeline
so every curve lives on directly comparable axes. Finally produces a 4-panel comparison
plot matching ``generation/reinforce_sweep.py``'s layout.

Usage:
  python analysis/parallel_sweep_aggregate.py --sweep_dir <path> --gpu 0
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

PROJECT_ROOT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))
from reinforce_search import DiffusionGenerator, RewardComputer, compute_segmentation  # noqa: E402


CONFIG_ORDER = [
    "default", "alpha_low", "alpha_high", "lr_low",
    "lr_high", "no_entropy", "high_entropy", "combined",
]


def load_config_logs(sweep_dir):
    configs_dir = os.path.join(sweep_dir, "configs")
    results = []
    for name in CONFIG_ORDER:
        csv_path = os.path.join(configs_dir, name, "reinforce_log.csv")
        if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
            print(f"  [skip] {name}: no CSV")
            continue
        df = pd.read_csv(csv_path)
        results.append({
            "name": name,
            "mean_rewards": df["mean_reward"].values,
            "best_rewards_ever": df["best_reward_ever"].values,
            "fg_clips": df["mean_fg_clip"].values,
            "bg_ssims": df["mean_bg_ssim"].values,
            "entropies": df["entropy"].values,
            "n_episodes": len(df),
            "final_best": float(df["best_reward_ever"].iloc[-1]),
        })
        print(f"  [ok]   {name}: {len(df)} episodes, best={results[-1]['final_best']:.4f}")
    return results


def parse_prompts(sweep_dir):
    # Take prompts from the first config's prompts.txt (all 8 share the same ones).
    for name in CONFIG_ORDER:
        p = os.path.join(sweep_dir, "configs", name, "prompts.txt")
        if not os.path.isfile(p):
            continue
        out = {}
        with open(p) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    out[k.strip()] = v.strip()
        return out
    raise FileNotFoundError(f"No prompts.txt found under {sweep_dir}/configs/*/")


def all_ones_baseline(generator, reward_computer, n_bits, alpha, device):
    print("  running all-ones baseline...", flush=True)
    ones_mask = torch.ones(1, n_bits, device=device)
    with torch.no_grad():
        img = generator.generate(ones_mask)
        reward, bg, fg = reward_computer.compute_rewards(img, alpha=alpha)
    print(f"    all-ones: R={reward.item():.4f}  bg={bg.item():.4f}  fg={fg.item():+.4f}", flush=True)
    return {
        "reward": reward.item(),
        "bg_ssim": bg.item(),
        "fg_clip": fg.item(),
        "image": img[0].cpu(),
    }


def random_baseline(generator, reward_computer, n_samples, batch_size, n_bits, alpha, device):
    print(f"  running random baseline ({n_samples} samples)...", flush=True)
    all_rewards = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    for bi in range(n_batches):
        b = min(batch_size, n_samples - bi * batch_size)
        masks = torch.bernoulli(torch.full((b, n_bits), 0.5, device=device))
        with torch.no_grad():
            images = generator.generate(masks)
            rewards, _, _ = reward_computer.compute_rewards(images, alpha=alpha)
        all_rewards.append(rewards.cpu().numpy())
        if bi % 10 == 0:
            print(f"    random batch {bi}/{n_batches}", flush=True)
    all_rewards = np.concatenate(all_rewards)
    running_max = np.maximum.accumulate(all_rewards)
    print(f"    random: best={running_max[-1]:.4f} mean={all_rewards.mean():.4f}", flush=True)
    return {
        "rewards": all_rewards,
        "running_max": running_max,
        "best": float(all_rewards.max()),
    }


def plot_sweep(results, random_res, ones_res, out_path, title, batch_size):
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Panel 1: mean reward
    ax = axes[0, 0]
    for r, c in zip(results, colors):
        ax.plot(r["mean_rewards"], label=r["name"], color=c, linewidth=1.5, alpha=0.85)
    if ones_res is not None:
        ax.axhline(ones_res["reward"], label=f"all-ones ({ones_res['reward']:.3f})",
                   color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("mean_reward (per batch)")
    ax.set_title("Mean reward per episode")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    # Panel 2: best-so-far
    ax = axes[0, 1]
    for r, c in zip(results, colors):
        ax.plot(r["best_rewards_ever"], label=r["name"], color=c, linewidth=1.5)
    if random_res is not None:
        n = len(random_res["running_max"])
        x_random = np.arange(n) / batch_size
        ax.plot(x_random, random_res["running_max"],
                label=f"random N={n} (best seen)", color="gray", linewidth=2, linestyle="--")
    if ones_res is not None:
        ax.axhline(ones_res["reward"], label=f"all-ones ({ones_res['reward']:.3f})",
                   color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Episode (= total images / batch_size)")
    ax.set_ylabel("best_reward_ever")
    ax.set_title("Best-so-far reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    # Panel 3: bg_ssim
    ax = axes[1, 0]
    for r, c in zip(results, colors):
        ax.plot(r["bg_ssims"], label=r["name"], color=c, linewidth=1.5, alpha=0.85)
    if ones_res is not None:
        ax.axhline(ones_res["bg_ssim"], label=f"all-ones bg ({ones_res['bg_ssim']:.3f})",
                   color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("mean bg_ssim")
    ax.set_title("Background SSIM (per episode)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    # Panel 4: fg_clip
    ax = axes[1, 1]
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
    ax.legend(fontsize=7, loc="lower right")

    # Panel 5: entropy
    ax = axes[2, 0]
    for r, c in zip(results, colors):
        ax.plot(r["entropies"], label=r["name"], color=c, linewidth=1.5, alpha=0.85)
    ax.axhline(14 * np.log(2), color="red", linewidth=0.5, linestyle=":",
               label="max entropy (14 bits)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("policy entropy H(π)")
    ax.set_title("Policy entropy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    # Panel 6: empty
    axes[2, 1].axis("off")

    plt.suptitle(title, fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True,
                   help="Sweep directory containing configs/<name>/reinforce_log.csv")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_bits", type=int, default=14)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--random_samples", type=int, default=640)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--vision_model", default="google/siglip2-so400m-patch14-384")
    args = p.parse_args()

    sweep_dir = args.sweep_dir
    print(f"=== Aggregator for {sweep_dir} ===", flush=True)

    # ── Load config CSVs ──
    print("\n--- Loading config logs ---", flush=True)
    results = load_config_logs(sweep_dir)
    if not results:
        raise SystemExit("No config CSVs found.")

    # ── Prompts ──
    prompts = parse_prompts(sweep_dir)
    src = prompts.get("source", "")
    tgt = prompts.get("target", "")
    seg = prompts.get("seg", "") or src
    print(f"\nprompts.source: {src[:80]}...")
    print(f"prompts.target: {tgt[:80]}...")
    print(f"prompts.seg:    {seg}")

    # ── Load generator + reward ──
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    n_bits = args.n_bits
    steps = args.steps if args.steps else n_bits * 2

    print("\n--- Loading FLUX generator ---", flush=True)
    generator = DiffusionGenerator(
        device=device,
        source_prompt=src,
        target_prompt=tgt,
        height=args.height, width=args.width,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        n_bits=n_bits, steps=steps,
    )

    print("\n--- Source image + segmentation ---", flush=True)
    zeros_mask = torch.zeros(1, n_bits, device=device)
    source_img = generator.generate(zeros_mask)
    bg_mask = compute_segmentation(source_img, seg, device)

    src_pil = Image.fromarray(
        (source_img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    src_pil.save(os.path.join(sweep_dir, "source_b0.jpg"), quality=95)

    print("\n--- Loading reward computer ---", flush=True)
    reward_computer = RewardComputer(
        device=device,
        source_image=source_img,
        bg_mask=bg_mask,
        source_prompt=src,
        target_prompt=tgt,
        img_size=args.height,
        vision_model=args.vision_model,
    )

    # ── Baselines ──
    print("\n--- Baselines ---", flush=True)
    ones_res = all_ones_baseline(generator, reward_computer, n_bits, args.alpha, device)
    tgt_pil = Image.fromarray(
        (ones_res["image"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8))
    tgt_pil.save(os.path.join(sweep_dir, f"target_b{(1 << n_bits) - 1}.jpg"), quality=95)

    random_res = random_baseline(
        generator, reward_computer,
        n_samples=args.random_samples, batch_size=args.batch_size,
        n_bits=n_bits, alpha=args.alpha, device=device,
    )

    # ── Summary + plot ──
    print("\n--- Saving outputs ---", flush=True)
    summary = {
        "sweep_dir": sweep_dir,
        "configs": [
            {"name": r["name"], "n_episodes": r["n_episodes"], "final_best": r["final_best"]}
            for r in results
        ],
        "all_ones_reward": ones_res["reward"],
        "all_ones_bg": ones_res["bg_ssim"],
        "all_ones_fg": ones_res["fg_clip"],
        "random_best": random_res["best"],
        "random_mean": float(random_res["rewards"].mean()),
        "random_n_samples": len(random_res["rewards"]),
    }
    with open(os.path.join(sweep_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary.json")

    np.savez(
        os.path.join(sweep_dir, "sweep_results.npz"),
        **{f"{r['name']}_mean_rewards": r["mean_rewards"] for r in results},
        **{f"{r['name']}_best_rewards": r["best_rewards_ever"] for r in results},
        **{f"{r['name']}_fg_clips": r["fg_clips"] for r in results},
        **{f"{r['name']}_entropies": r["entropies"] for r in results},
        random_rewards=random_res["rewards"],
        random_running_max=random_res["running_max"],
        ones_reward=np.array([ones_res["reward"]]),
    )
    print(f"  sweep_results.npz")

    short_src = src[:40] + "..." if len(src) > 40 else src
    short_tgt = tgt[:40] + "..." if len(tgt) > 40 else tgt
    title = f"{os.path.basename(sweep_dir)}: {short_src} → {short_tgt}"
    plot_sweep(results, random_res, ones_res,
               os.path.join(sweep_dir, "sweep_curves.png"), title,
               batch_size=args.batch_size)

    print(f"\nAll done. Results in {sweep_dir}", flush=True)


if __name__ == "__main__":
    main()
