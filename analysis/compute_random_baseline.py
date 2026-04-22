"""Compute random-path baseline rewards for experiments that don't have them.

For each experiment in --dir, generates N random binary masks, runs FLUX,
and computes reward. Stores N-length array to random_rewards.npy per experiment.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))
from reinforce_search import DiffusionGenerator, RewardComputer, compute_segmentation


def parse_prompts(p):
    out = {}
    with open(p) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="Single experiment dir with prompts.txt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=640)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_bits", type=int, default=14)
    ap.add_argument("--steps", type=int, default=None,
                    help="Total diffusion steps; default = n_bits * 2 (repeat_factor=2)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--vision_model", default="google/siglip2-so400m-patch14-384")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    out_path = os.path.join(args.exp_dir, "random_rewards.npy")
    bg_exists = os.path.isfile(os.path.join(args.exp_dir, "random_bg_ssim.npy"))
    fg_exists = os.path.isfile(os.path.join(args.exp_dir, "random_fg_clip.npy"))
    if (os.path.isfile(out_path) and bg_exists and fg_exists
            and np.load(out_path).shape[0] >= args.n_samples):
        print(f"{args.exp_dir}: has reward+bg+fg for {args.n_samples}+ samples, skipping")
        return

    prompts = parse_prompts(os.path.join(args.exp_dir, "prompts.txt"))

    steps = args.steps if args.steps is not None else args.n_bits * 2
    generator = DiffusionGenerator(
        device=device, source_prompt=prompts["source"], target_prompt=prompts["target"],
        height=512, width=512, guidance_scale=4.0, seed=42,
        n_bits=args.n_bits, steps=steps,
    )

    # Source + seg
    source_img = generator.generate(torch.zeros(1, args.n_bits, device=device))
    mask_path = os.path.join(args.exp_dir, "bg_mask.npy")
    if os.path.isfile(mask_path):
        bg_mask = np.load(mask_path).astype(np.float32)
    else:
        bg_mask = compute_segmentation(source_img, prompts.get("seg", prompts["source"]), device)
        np.save(mask_path, (bg_mask > 0.5).astype(np.uint8))

    rc = RewardComputer(
        device=device, source_image=source_img, bg_mask=bg_mask,
        source_prompt=prompts["source"], target_prompt=prompts["target"],
        img_size=512, vision_model=args.vision_model,
    )

    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    rewards = []
    bg_all = []
    fg_all = []
    t0 = time.perf_counter()
    for bi in range(n_batches):
        b = min(args.batch_size, args.n_samples - bi * args.batch_size)
        masks = torch.bernoulli(torch.full((b, args.n_bits), 0.5, device=device))
        with torch.no_grad():
            images = generator.generate(masks)
            r, bg, fg = rc.compute_rewards(images, alpha=args.alpha)
        rewards.append(r.cpu().numpy())
        bg_all.append(bg.cpu().numpy())
        fg_all.append(fg.cpu().numpy())
        if bi % 10 == 0:
            print(f"  {os.path.basename(args.exp_dir)} batch {bi}/{n_batches} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    rewards = np.concatenate(rewards)
    bg_all = np.concatenate(bg_all)
    fg_all = np.concatenate(fg_all)
    np.save(out_path, rewards)
    np.save(os.path.join(args.exp_dir, "random_bg_ssim.npy"), bg_all)
    np.save(os.path.join(args.exp_dir, "random_fg_clip.npy"), fg_all)
    print(f"Saved: {out_path}  ({len(rewards)} rewards, best={rewards.max():.4f}, "
          f"mean={rewards.mean():.4f}, time={time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
