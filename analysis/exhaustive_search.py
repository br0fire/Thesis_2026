"""Exhaustive search: generate all 2^n_bits images, compute rewards, save top-K + array.

Upper bound for any binary-path search method. Saves:
  - exhaustive_rewards.npy: (2^n_bits,) float32 array of rewards
  - top_K images with reward + mask integer in filename
  - prompts.txt + bg_mask.npy (reused from REINFORCE run if available)
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))
from reinforce_search import DiffusionGenerator, RewardComputer, compute_segmentation, mask_to_int_batch


def int_to_bits(n, n_bits):
    """Convert int to (n_bits,) tensor of bits, MSB-first."""
    bits = torch.zeros(n_bits, dtype=torch.float32)
    for i in range(n_bits):
        if n & (1 << (n_bits - 1 - i)):
            bits[i] = 1.0
    return bits


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
    ap.add_argument("--exp_dir", required=True, help="Experiment dir with prompts.txt (and optionally source_b0.jpg + bg_mask.npy)")
    ap.add_argument("--out_dir", default=None, help="Metadata output dir (default: exp_dir) — rewards.npy + top-K only")
    ap.add_argument("--images_dir", default=None, help="Directory for full set of 16384 JPGs (large, default: on NFS3)")
    ap.add_argument("--save_all_images", action="store_true", help="Save every generated image (default off)")
    ap.add_argument("--jpg_quality", type=int, default=85)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_bits", type=int, default=14)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--vision_model", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--resume", action="store_true", help="Resume from saved rewards if exists")
    args = ap.parse_args()

    out_dir = args.out_dir or args.exp_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.save_all_images:
        if args.images_dir is None:
            raise SystemExit("--save_all_images requires --images_dir")
        os.makedirs(args.images_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    prompts = parse_prompts(os.path.join(args.exp_dir, "prompts.txt"))
    print(f"Source: {prompts['source'][:80]}...", flush=True)

    # Load FLUX
    print("Loading FLUX...", flush=True)
    generator = DiffusionGenerator(
        device=device, source_prompt=prompts["source"], target_prompt=prompts["target"],
        height=512, width=512, guidance_scale=4.0, seed=42,
        n_bits=args.n_bits, steps=args.n_bits * 2,
    )

    # Source + segmentation (reuse from exp_dir if possible)
    source_img = generator.generate(torch.zeros(1, args.n_bits, device=device))
    mask_path = os.path.join(args.exp_dir, "bg_mask.npy")
    if os.path.isfile(mask_path):
        bg_mask = np.load(mask_path).astype(np.float32)
        print(f"Reused saved bg_mask: {bg_mask.shape}", flush=True)
    else:
        bg_mask = compute_segmentation(source_img, prompts.get("seg", prompts["source"]), device)
        np.save(mask_path, (bg_mask > 0.5).astype(np.uint8))

    rc = RewardComputer(
        device=device, source_image=source_img, bg_mask=bg_mask,
        source_prompt=prompts["source"], target_prompt=prompts["target"],
        img_size=512, vision_model=args.vision_model,
    )

    total = 1 << args.n_bits
    rewards_path = os.path.join(out_dir, "exhaustive_rewards.npy")

    # Async save pool to hide JPG I/O behind GPU compute
    save_pool = ThreadPoolExecutor(max_workers=4) if args.save_all_images else None
    def async_save(img_arr, fn, quality):
        Image.fromarray(img_arr).save(fn, quality=quality)

    if args.resume and os.path.isfile(rewards_path):
        rewards = np.load(rewards_path)
        # Find first NaN as resume point
        start_i = int(np.argmin(~np.isnan(rewards)))
        if np.all(~np.isnan(rewards[:start_i + 1])):
            start_i = len(rewards[~np.isnan(rewards)])
        print(f"Resuming at index {start_i}/{total}", flush=True)
    else:
        rewards = np.full(total, np.nan, dtype=np.float32)
        start_i = 0

    # Top-K tracking
    top_k = []  # list of (reward, mask_int, image_tensor_cpu)

    t0 = time.perf_counter()
    for i in range(start_i, total, args.batch_size):
        b = min(args.batch_size, total - i)
        masks = torch.stack([int_to_bits(i + j, args.n_bits) for j in range(b)]).to(device)
        with torch.no_grad():
            images = generator.generate(masks)
            r, _, _ = rc.compute_rewards(images, alpha=args.alpha)
        r_np = r.cpu().numpy()
        rewards[i:i + b] = r_np

        # Save all images to NFS3 if requested (async via threadpool)
        if args.save_all_images:
            imgs_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            for j in range(b):
                mask_int = i + j
                fn = f"path_{mask_int:05d}_b{mask_int}.jpg"
                # Copy the array since np is about to be overwritten next batch
                save_pool.submit(async_save, imgs_np[j].copy(),
                                 os.path.join(args.images_dir, fn), args.jpg_quality)

        # Top-K update
        for j in range(b):
            mask_int = i + j
            reward = float(r_np[j])
            if len(top_k) < args.top_k:
                top_k.append((reward, mask_int, images[j].detach().cpu()))
                top_k.sort(key=lambda x: -x[0])
            elif reward > top_k[-1][0]:
                top_k[-1] = (reward, mask_int, images[j].detach().cpu())
                top_k.sort(key=lambda x: -x[0])

        # Periodic save + log
        if (i // args.batch_size) % 50 == 0:
            np.save(rewards_path, rewards)
            best = np.nanmax(rewards)
            elapsed = time.perf_counter() - t0
            done = i + b
            eta = elapsed * (total - done) / max(done - start_i, 1)
            print(f"  [{done:5d}/{total}] best={best:.4f} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    # Final save
    if save_pool is not None:
        save_pool.shutdown(wait=True)  # finish any pending JPG writes
    np.save(rewards_path, rewards)
    print(f"\nSaved rewards: {rewards_path}", flush=True)

    # Save top-K images
    for rank, (reward, mask_int, img_t) in enumerate(top_k):
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(
            os.path.join(out_dir, f"exhaustive_top{rank}_r{reward:.4f}_b{mask_int}.jpg"), quality=95)

    # Summary
    best_i = int(np.nanargmax(rewards))
    best_r = float(rewards[best_i])
    print(f"\n=== Exhaustive done ===")
    print(f"  Total images: {total}")
    print(f"  Best reward:  {best_r:.4f}  (mask_int={best_i})")
    print(f"  Mean reward:  {np.nanmean(rewards):.4f}")
    print(f"  Time:         {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
