"""Generate canonical source + target + bg_mask for an experiment, ONCE.

Produces a stable reference that all downstream methods (exhaustive, REINFORCE,
random) load verbatim — eliminates CUDA non-determinism across runs.

Output layout:
  <output_dir>/
    source.pt          # torch (1,3,H,W) float32 in [0,1]
    target.pt          # torch (1,3,H,W) float32 — generated with all-ones mask
    source.png         # lossless preview
    target.png
    bg_mask.npy        # uint8 (H,W) {0,1}
    bg_mask_vis.png
    prompts.txt
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reinforce_search import DiffusionGenerator, compute_segmentation


def tensor_to_uint8(t):
    arr = (t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_prompt", required=True)
    ap.add_argument("--target_prompt", required=True)
    ap.add_argument("--seg_prompt",    required=True)
    ap.add_argument("--output_dir",    required=True)
    ap.add_argument("--gpu",           type=int, default=0)
    ap.add_argument("--n_bits",        type=int, default=14)
    ap.add_argument("--steps",         type=int, default=None)
    ap.add_argument("--height",        type=int, default=512)
    ap.add_argument("--width",         type=int, default=512)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--seg_method",    choices=["sam3", "gdino_sam", "clipseg"], default="sam3")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    n_bits = args.n_bits
    steps = args.steps if args.steps else n_bits * 2

    print(f"Loading FLUX on cuda:{args.gpu}...", flush=True)
    gen = DiffusionGenerator(
        device=device, source_prompt=args.source_prompt, target_prompt=args.target_prompt,
        height=args.height, width=args.width, guidance_scale=args.guidance_scale,
        seed=args.seed, n_bits=n_bits, steps=steps,
    )

    print("Generating source (all-zeros mask)...", flush=True)
    source_img = gen.generate(torch.zeros(1, n_bits, device=device))  # (1,3,H,W) [0,1]
    print(f"  source dtype={source_img.dtype}  shape={tuple(source_img.shape)}", flush=True)

    print("Generating target (all-ones mask)...", flush=True)
    target_img = gen.generate(torch.ones(1, n_bits, device=device))

    print(f"Running segmentation ({args.seg_method}) with prompt '{args.seg_prompt}'...", flush=True)
    bg_mask = compute_segmentation(source_img, args.seg_prompt, device, method=args.seg_method)
    bg_mask_u8 = (bg_mask > 0.5).astype(np.uint8)
    print(f"  bg={bg_mask_u8.mean() * 100:.1f}%  fg={(1 - bg_mask_u8.mean()) * 100:.1f}%", flush=True)

    # Save tensors on CPU so downstream scripts can load on any device
    torch.save(source_img.detach().cpu(), os.path.join(args.output_dir, "source.pt"))
    torch.save(target_img.detach().cpu(), os.path.join(args.output_dir, "target.pt"))
    np.save(os.path.join(args.output_dir, "bg_mask.npy"), bg_mask_u8)

    # Human-readable previews (PNG for lossless)
    Image.fromarray(tensor_to_uint8(source_img)).save(os.path.join(args.output_dir, "source.png"))
    Image.fromarray(tensor_to_uint8(target_img)).save(os.path.join(args.output_dir, "target.png"))

    # Segmentation visualization
    src_np = tensor_to_uint8(source_img)
    H, W = src_np.shape[:2]
    m_resized = np.array(Image.fromarray((bg_mask_u8 * 255)).resize((W, H), Image.NEAREST))
    fg = m_resized < 127
    overlay = src_np.copy()
    overlay[fg, 0] = np.clip(overlay[fg, 0].astype(int) + 100, 0, 255)
    overlay[fg, 1] = (overlay[fg, 1] * 0.5).astype(np.uint8)
    overlay[fg, 2] = (overlay[fg, 2] * 0.5).astype(np.uint8)
    Image.fromarray(np.concatenate([src_np, overlay], axis=1)).save(
        os.path.join(args.output_dir, "bg_mask_vis.png"))

    with open(os.path.join(args.output_dir, "prompts.txt"), "w") as f:
        f.write(f"source: {args.source_prompt}\n")
        f.write(f"target: {args.target_prompt}\n")
        f.write(f"seg: {args.seg_prompt}\n")
        f.write(f"n_bits: {n_bits}\n")
        f.write(f"steps: {steps}\n")
        f.write(f"seed: {args.seed}\n")

    print(f"\nSaved canonical to {args.output_dir}")


if __name__ == "__main__":
    main()
