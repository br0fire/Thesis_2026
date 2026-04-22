"""Exhaustive 2^n_bits enumeration using CANONICAL source + mask (no CUDA drift).

Saves all images as a single contiguous uint8 array (memmap-friendly).
Saves per-mask bg_ssim and fg_clip arrays — rewards for any α are analytical.

Output:
  <out_dir>/
    all_images.npy     # (2^n_bits, 3, H, W) uint8 — big, on NFS3 if --image_root given
    bg_ssim.npy        # (2^n_bits,) float32
    fg_clip.npy        # (2^n_bits,) float32
    top_k/             # top-K previews as PNG (lossless)
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generation"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reinforce_search import DiffusionGenerator, RewardComputer


def int_to_bits(n, n_bits):
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
    ap.add_argument("--canonical_dir", required=True,
                    help="Dir with source.pt, target.pt, bg_mask.npy, prompts.txt")
    ap.add_argument("--out_dir",       required=True, help="Dir for bg_ssim/fg_clip/top_k")
    ap.add_argument("--images_npy",    required=True, help="Path for all_images.npy (on NFS3)")
    ap.add_argument("--gpu",           type=int, default=0)
    ap.add_argument("--batch_size",    type=int, default=8)
    ap.add_argument("--top_k",         type=int, default=32)
    ap.add_argument("--alpha",         type=float, default=0.5, help="Only used for top-K sort")
    ap.add_argument("--vision_model",  default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--resume",        action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.images_npy) or ".", exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    prompts = parse_prompts(os.path.join(args.canonical_dir, "prompts.txt"))
    n_bits = int(prompts.get("n_bits", 14))
    steps = int(prompts.get("steps", n_bits * 2))
    total = 1 << n_bits
    H = W = 512

    # ── Load canonical ──
    print(f"Loading canonical from {args.canonical_dir}...", flush=True)
    source_img = torch.load(os.path.join(args.canonical_dir, "source.pt"),
                             map_location=device, weights_only=False).to(device).float()
    if source_img.dim() == 3:
        source_img = source_img.unsqueeze(0)
    bg_mask = np.load(os.path.join(args.canonical_dir, "bg_mask.npy")).astype(np.float32)

    # ── Load FLUX ──
    print("Loading FLUX...", flush=True)
    gen = DiffusionGenerator(
        device=device,
        source_prompt=prompts["source"], target_prompt=prompts["target"],
        height=H, width=W, guidance_scale=4.0,
        seed=int(prompts.get("seed", 42)), n_bits=n_bits, steps=steps,
    )

    # ── Build reward computer (uses canonical source + mask) ──
    print("Loading SigLIP (RewardComputer)...", flush=True)
    rc = RewardComputer(
        device=device, source_image=source_img, bg_mask=bg_mask,
        source_prompt=prompts["source"], target_prompt=prompts["target"],
        img_size=H, vision_model=args.vision_model,
    )

    # ── Prepare storage ──
    bg_path = os.path.join(args.out_dir, "bg_ssim.npy")
    fg_path = os.path.join(args.out_dir, "fg_clip.npy")
    imgs_path = args.images_npy

    if args.resume and os.path.isfile(bg_path) and os.path.isfile(fg_path) and os.path.isfile(imgs_path):
        bg_all = np.load(bg_path, mmap_mode="r+")
        fg_all = np.load(fg_path, mmap_mode="r+")
        imgs_all = np.load(imgs_path, mmap_mode="r+")
        start_i = int(np.argmin(np.isfinite(bg_all)))
        if np.isfinite(bg_all).all():
            start_i = total
        print(f"Resuming at {start_i}/{total}", flush=True)
    else:
        # Allocate memmap for all outputs — avoids concurrent np.save failures on NFS.
        print(f"Allocating {imgs_path}  shape=({total},3,{H},{W}) uint8 ~{total * 3 * H * W / 1e9:.1f} GB",
              flush=True)
        imgs_all = np.lib.format.open_memmap(imgs_path, mode="w+",
                                              dtype=np.uint8, shape=(total, 3, H, W))
        bg_all = np.lib.format.open_memmap(bg_path, mode="w+",
                                            dtype=np.float32, shape=(total,))
        fg_all = np.lib.format.open_memmap(fg_path, mode="w+",
                                            dtype=np.float32, shape=(total,))
        bg_all[:] = np.nan
        fg_all[:] = np.nan
        start_i = 0

    # ── Iterate through all masks ──
    top_k = []  # list of (reward_alpha, mask_int)
    t0 = time.perf_counter()
    for i in range(start_i, total, args.batch_size):
        b = min(args.batch_size, total - i)
        masks = torch.stack([int_to_bits(i + j, n_bits) for j in range(b)]).to(device)
        with torch.no_grad():
            images = gen.generate(masks)  # (b, 3, H, W) float32 [0,1]
            _, bg_v, fg_v = rc.compute_rewards(images, alpha=args.alpha)

        # Write to memmap
        imgs_u8 = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        imgs_u8 = imgs_u8.transpose(0, 3, 1, 2)  # (b, 3, H, W)
        imgs_all[i:i + b] = imgs_u8
        bg_all[i:i + b] = bg_v.cpu().numpy()
        fg_all[i:i + b] = fg_v.cpu().numpy()

        # Track top-K by reward at args.alpha
        fg_sig = 1.0 / (1.0 + np.exp(-fg_v.cpu().numpy() * 10.0))
        rewards_b = (np.clip(bg_v.cpu().numpy(), 1e-6, None) ** args.alpha
                     * np.clip(fg_sig, 1e-6, None) ** (1 - args.alpha))
        for j in range(b):
            mask_int = i + j
            r = float(rewards_b[j])
            if len(top_k) < args.top_k:
                top_k.append((r, mask_int))
                top_k.sort(key=lambda x: -x[0])
            elif r > top_k[-1][0]:
                top_k[-1] = (r, mask_int)
                top_k.sort(key=lambda x: -x[0])

        if (i // args.batch_size) % 50 == 0:
            elapsed = time.perf_counter() - t0
            done = i + b
            eta = elapsed * (total - done) / max(done - start_i, 1)
            print(f"  [{done:5d}/{total}]  best={top_k[0][0]:.4f}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        # Periodic flush — only memmap (images) to avoid concurrent np.save failures on NFS
        if (i // args.batch_size) % 100 == 0:
            if hasattr(imgs_all, "flush"):
                imgs_all.flush()

    # Final flush — memmap writes are already persistent, just ensure data on disk
    if hasattr(bg_all, "flush"): bg_all.flush()
    if hasattr(fg_all, "flush"): fg_all.flush()
    if hasattr(imgs_all, "flush"): imgs_all.flush()

    # Save top-K as PNG
    topk_dir = os.path.join(args.out_dir, "top_k")
    os.makedirs(topk_dir, exist_ok=True)
    for rank, (r, mask_int) in enumerate(top_k[:args.top_k]):
        img_u8 = imgs_all[mask_int]
        Image.fromarray(img_u8.transpose(1, 2, 0)).save(
            os.path.join(topk_dir, f"top_{rank:02d}_r{r:.4f}_b{mask_int}.png"))

    print(f"\n=== Exhaustive done ===", flush=True)
    print(f"  Images: {imgs_path}  ({total * 3 * H * W / 1e9:.1f} GB)")
    print(f"  bg_ssim: {bg_path}  mean={np.nanmean(bg_all):.4f} max={np.nanmax(bg_all):.4f}")
    print(f"  fg_clip: {fg_path}  mean={np.nanmean(fg_all):.4f} max={np.nanmax(fg_all):.4f}")
    print(f"  Top-K: {topk_dir}  best={top_k[0][0]:.4f} (mask={top_k[0][1]})")
    print(f"  Total time: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
