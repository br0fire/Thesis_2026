"""Recompute bg_ssim and fg_clip for every image in an exhaustive run.

Reads the 16384 JPGs from NFS3 (saved earlier during exhaustive_search.py),
reconstructs the RewardComputer, and saves per-image bg_ssim and fg_clip arrays.
With these, reward at ANY alpha is an analytical formula:
  R(α) = bg_ssim^α · σ(fg_clip·10)^(1-α)
No FLUX is needed — only SigLIP (~1GB VRAM) + SSIM (tiny).
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
from reinforce_search import RewardComputer, compute_segmentation


def parse_prompts(p):
    out = {}
    with open(p) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def load_jpg_batch(paths, executor):
    """Load N JPGs in parallel, return (N, 3, H, W) tensor in [0, 1] on CPU."""
    def _load(p):
        img = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        return img
    arrs = list(executor.map(_load, paths))
    batch = np.stack(arrs).astype(np.float32) / 255.0  # (N, H, W, 3)
    return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--images_root", default="/home/jovyan/shares/SR006.nfs3/svgrozny/exhaustive_new_bgrich")
    ap.add_argument("--meta_root",   default=None,
                    help="Where source_b0.jpg + bg_mask.npy + prompts.txt live. "
                         "Default: analysis/reinforce_analysis/new_bgrich/<exp_name>")
    ap.add_argument("--out_root",    default=None,
                    help="Where to save bg_ssim.npy + fg_clip.npy arrays. "
                         "Default: analysis/reinforce_analysis/exhaustive/<exp_name>")
    ap.add_argument("--n_bits",      type=int, default=14)
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--io_workers",  type=int, default=16)
    ap.add_argument("--vision_model", default="google/siglip2-so400m-patch14-384")
    args = ap.parse_args()

    exp = args.exp_name
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    meta_dir = args.meta_root or os.path.join(
        PROJECT_ROOT, "analysis/reinforce_analysis/new_bgrich", exp)
    out_dir  = args.out_root  or os.path.join(
        PROJECT_ROOT, "analysis/reinforce_analysis/exhaustive", exp)
    os.makedirs(out_dir, exist_ok=True)

    imgs_dir = os.path.join(args.images_root, exp)
    total = 1 << args.n_bits
    if len(os.listdir(imgs_dir)) < total:
        raise SystemExit(f"{imgs_dir}: only {len(os.listdir(imgs_dir))} files, need {total}")

    # Load source + bg_mask + prompts
    prompts = parse_prompts(os.path.join(meta_dir, "prompts.txt"))
    src_pil = Image.open(os.path.join(meta_dir, "source_b0.jpg")).convert("RGB")
    src_arr = np.array(src_pil).astype(np.float32) / 255.0
    src_t = torch.from_numpy(src_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    mask_path = os.path.join(meta_dir, "bg_mask.npy")
    if os.path.isfile(mask_path):
        bg_mask = np.load(mask_path).astype(np.float32)
    else:
        bg_mask = compute_segmentation(src_t, prompts.get("seg", prompts["source"]), device)
        np.save(mask_path, (bg_mask > 0.5).astype(np.uint8))

    print(f"Loading SigLIP + building RewardComputer...", flush=True)
    rc = RewardComputer(
        device=device, source_image=src_t, bg_mask=bg_mask,
        source_prompt=prompts["source"], target_prompt=prompts["target"],
        img_size=512, vision_model=args.vision_model,
    )

    bg_ssim_all = np.empty(total, dtype=np.float32)
    fg_clip_all = np.empty(total, dtype=np.float32)

    io_pool = ThreadPoolExecutor(max_workers=args.io_workers)
    t0 = time.perf_counter()
    for i in range(0, total, args.batch_size):
        b = min(args.batch_size, total - i)
        paths = [os.path.join(imgs_dir, f"path_{i + j:05d}_b{i + j}.jpg") for j in range(b)]
        imgs = load_jpg_batch(paths, io_pool).to(device, non_blocking=True)
        with torch.no_grad():
            _, bg, fg = rc.compute_rewards(imgs, alpha=0.5)  # alpha irrelevant for bg/fg
        bg_ssim_all[i:i + b] = bg.cpu().numpy()
        fg_clip_all[i:i + b] = fg.cpu().numpy()

        if (i // args.batch_size) % 32 == 0:
            elapsed = time.perf_counter() - t0
            done = i + b
            eta = elapsed * (total - done) / max(done, 1)
            print(f"  [{done:5d}/{total}] ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    io_pool.shutdown()

    np.save(os.path.join(out_dir, "exhaustive_bg_ssim.npy"), bg_ssim_all)
    np.save(os.path.join(out_dir, "exhaustive_fg_clip.npy"), fg_clip_all)

    # Also derive rewards at α=0.5 and α=0.7 for convenience
    for alpha in (0.5, 0.7):
        fg_sig = 1.0 / (1.0 + np.exp(-fg_clip_all * 10.0))
        rewards = (np.clip(bg_ssim_all, 1e-6, None) ** alpha) * (np.clip(fg_sig, 1e-6, None) ** (1 - alpha))
        np.save(os.path.join(out_dir, f"exhaustive_rewards_alpha{int(alpha*100):02d}.npy"), rewards)
        print(f"  α={alpha}: best={rewards.max():.4f}  mean={rewards.mean():.4f}", flush=True)

    total_time = time.perf_counter() - t0
    print(f"\nSaved components for {exp}  (time={total_time:.0f}s)")


if __name__ == "__main__":
    main()
