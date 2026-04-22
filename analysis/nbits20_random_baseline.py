"""Compute random baseline for n_bits=20 experiments at equal image budget.

For each experiment, samples the same number of random masks as REINFORCE used,
computes rewards, and reports running-max. Saves comparison summary.

Usage:
  python analysis/nbits20_random_baseline.py --exp_dir analysis/reinforce_analysis/nbits20/<name> --gpu 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PROJECT_ROOT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))
from reinforce_search import DiffusionGenerator, RewardComputer, compute_segmentation


def parse_prompts(exp_dir):
    p = os.path.join(exp_dir, "prompts.txt")
    out = {}
    with open(p) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_bits", type=int, default=20)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--vision_model", default="google/siglip2-so400m-patch14-384")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    n_bits = args.n_bits
    steps = n_bits * 2

    ckpt = torch.load(os.path.join(args.exp_dir, "reinforce_result.pt"),
                       map_location="cpu", weights_only=False)
    total_images = ckpt.get("total_images", 0)
    reinforce_best = ckpt.get("best_reward", 0)
    print(f"REINFORCE: {total_images} images, best={reinforce_best:.4f}", flush=True)

    prompts = parse_prompts(args.exp_dir)
    src = prompts["source"]
    tgt = prompts["target"]
    seg = prompts.get("seg", src)

    print("Loading generator...", flush=True)
    generator = DiffusionGenerator(
        device=device, source_prompt=src, target_prompt=tgt,
        height=args.height, width=args.width, guidance_scale=args.guidance_scale,
        seed=args.seed, n_bits=n_bits, steps=steps,
    )

    print("Source + segmentation...", flush=True)
    with torch.no_grad():
        source_img = generator.generate(torch.zeros(1, n_bits, device=device))
    bg_mask = compute_segmentation(source_img, seg, device)

    print("Loading reward computer...", flush=True)
    reward_computer = RewardComputer(
        device=device, source_image=source_img, bg_mask=bg_mask,
        source_prompt=src, target_prompt=tgt,
        img_size=args.height, vision_model=args.vision_model,
    )

    n_samples = total_images
    n_batches = (n_samples + args.batch_size - 1) // args.batch_size
    print(f"Random baseline: {n_samples} samples ({n_batches} batches)...", flush=True)

    all_rewards = []
    t0 = time.perf_counter()
    for bi in range(n_batches):
        b = min(args.batch_size, n_samples - bi * args.batch_size)
        masks = torch.bernoulli(torch.full((b, n_bits), 0.5, device=device))
        with torch.no_grad():
            images = generator.generate(masks)
            rewards, _, _ = reward_computer.compute_rewards(images, alpha=args.alpha)
        all_rewards.append(rewards.cpu().numpy())
        if bi % 10 == 0:
            elapsed = time.perf_counter() - t0
            best_so_far = np.max(np.concatenate(all_rewards))
            print(f"  batch {bi}/{n_batches}  best={best_so_far:.4f}  ({elapsed:.0f}s)", flush=True)

    all_rewards = np.concatenate(all_rewards)
    running_max = np.maximum.accumulate(all_rewards)
    random_best = float(running_max[-1])
    random_mean = float(all_rewards.mean())
    elapsed = time.perf_counter() - t0

    edge = reinforce_best - random_best
    print(f"\n{'='*60}")
    print(f"REINFORCE best:  {reinforce_best:.4f}  ({total_images} images)")
    print(f"Random best:     {random_best:.4f}  ({n_samples} images)")
    print(f"Edge:            {edge:+.4f}")
    print(f"Random mean:     {random_mean:.4f}")
    print(f"Time:            {elapsed:.0f}s")
    print(f"{'='*60}")

    result = {
        "reinforce_best": reinforce_best,
        "reinforce_images": total_images,
        "random_best": random_best,
        "random_mean": random_mean,
        "random_images": n_samples,
        "edge": edge,
    }
    out_path = os.path.join(args.exp_dir, "random_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    np.save(os.path.join(args.exp_dir, "random_rewards.npy"), all_rewards)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
