"""
Retroactively generate bg_mask_vis.jpg for experiments.

Uses the same segmentation pipeline as reinforce_search.py (SAM 3.1 by default,
with GDino+SAM2 and CLIPSeg as fallbacks). Re-imports those helpers directly
so the behavior stays in sync.

Usage:
  python analysis/backfill_mask_vis.py              # skips existing (SAM 3.1)
  python analysis/backfill_mask_vis.py --force      # regenerates all
  python analysis/backfill_mask_vis.py --method gdino_sam   # GDino+SAM2
  python analysis/backfill_mask_vis.py --method clipseg     # legacy CLIPSeg
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ANALYSIS_DIR = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"

# Import segmentation helpers from the main training script
PROJECT_ROOT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))
from reinforce_search import compute_segmentation  # noqa: E402


def parse_prompts_txt(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def build_vis(source_pil, bg_mask):
    src_np = np.array(source_pil.convert("RGB"))
    H, W = src_np.shape[:2]
    if bg_mask.shape != (H, W):
        bg_pil = Image.fromarray((bg_mask * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
        bg_mask = (np.array(bg_pil) > 127).astype(np.float32)

    bg_full = bg_mask > 0.5
    fg_full = ~bg_full
    overlay = src_np.copy()
    overlay[fg_full, 0] = np.clip(overlay[fg_full, 0].astype(int) + 100, 0, 255)
    overlay[fg_full, 1] = (overlay[fg_full, 1] * 0.5).astype(np.uint8)
    overlay[fg_full, 2] = (overlay[fg_full, 2] * 0.5).astype(np.uint8)
    side_by_side = np.concatenate([src_np, overlay], axis=1)
    return Image.fromarray(side_by_side)


def seg_prompt_from_name(exp_name):
    """Fallback for experiments whose prompts.txt is missing."""
    mapping = {
        "catdog": "cat", "car_taxi": "car", "sunflower": "sunflower",
        "chair_throne": "chair", "penguin_flamingo": "penguin",
        "cake_books": "cake", "lighthouse_castle": "lighthouse",
        "violin_guitar": "violin", "room": "wooden coffee table",
        "snow_volcano": "mountain peak", "butterfly_hummingbird": "butterfly",
        "sail_pirate": "sailboat", "bgrich_teapot_globe": "teapot",
        "bgrich_candle_crystal": "candle", "bgrich_typewriter_laptop": "typewriter",
    }
    for key, prompt in mapping.items():
        if key in exp_name:
            return prompt
    # standalone "horse" must come last so it doesn't match e.g. "horse_v3"
    if exp_name.startswith("reinforce_horse"):
        return "horse"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--gpu", type=int, default=-1, help="-1 = CPU")
    p.add_argument("--method", choices=["sam3", "gdino_sam", "clipseg"], default="sam3")
    args = p.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}  Method: {args.method}")

    total = 0
    done = 0
    skipped = 0
    failed = 0
    for version in sorted(os.listdir(ANALYSIS_DIR)):
        exp_root = os.path.join(ANALYSIS_DIR, version, "experiments")
        if not os.path.isdir(exp_root):
            continue
        for exp_name in sorted(os.listdir(exp_root)):
            exp_dir = os.path.join(exp_root, exp_name)
            if not os.path.isdir(exp_dir):
                continue
            total += 1
            vis_path = os.path.join(exp_dir, "bg_mask_vis.jpg")
            if os.path.isfile(vis_path) and not args.force:
                skipped += 1
                continue

            source_path = os.path.join(exp_dir, "source_b0.jpg")
            if not os.path.isfile(source_path):
                print(f"  {exp_name}: no source_b0.jpg")
                failed += 1
                continue

            prompts = parse_prompts_txt(os.path.join(exp_dir, "prompts.txt"))
            seg_prompt = prompts.get("seg", "") or seg_prompt_from_name(exp_name)
            if not seg_prompt:
                print(f"  {exp_name}: unknown seg_prompt")
                failed += 1
                continue

            try:
                source_pil = Image.open(source_path).convert("RGB")
                src_arr = np.array(source_pil).astype(np.float32) / 255.0
                src_tensor = torch.from_numpy(src_arr).permute(2, 0, 1).unsqueeze(0).to(device)
                bg_mask = compute_segmentation(src_tensor, seg_prompt, device, method=args.method)
                vis = build_vis(source_pil, bg_mask)
                vis.save(vis_path, quality=92)
                fg_pct = 100 * (bg_mask < 0.5).mean()
                print(f"  {exp_name}: seg='{seg_prompt}' fg={fg_pct:.1f}%")
                done += 1
            except Exception as e:
                print(f"  {exp_name}: FAILED — {e}")
                failed += 1

    print(f"\nTotal: {total}, generated: {done}, skipped: {skipped}, failed: {failed}")


if __name__ == "__main__":
    main()
