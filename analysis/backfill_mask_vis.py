"""
Retroactively generate bg_mask_vis.jpg for experiments that were run before
the vis-save feature was added. Reads source_b0.jpg + prompts.txt from each
experiment dir, re-runs CLIPSeg to recover the mask (with the same
auto-lowering fallback as reinforce_search.py), and writes a side-by-side
visualization.

Usage:
  python analysis/backfill_mask_vis.py [--force]

  --force  overwrite existing bg_mask_vis.jpg files
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ANALYSIS_DIR = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"


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


def compute_mask_from_source(source_pil, seg_prompt, device, model, processor,
                              dilate_px=10, threshold=0.5):
    """Same logic as reinforce_search.py compute_segmentation, but takes a PIL img."""
    H, W = source_pil.size[1], source_pil.size[0]
    inputs = processor(text=[seg_prompt], images=[source_pil], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    upsampled = F.interpolate(
        logits.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze()
    prob = torch.sigmoid(upsampled).cpu().numpy()

    foreground = (prob > threshold).astype(np.uint8)
    if foreground.sum() == 0:
        for fallback in (0.3, 0.15, 0.05):
            if fallback >= threshold:
                continue
            foreground = (prob > fallback).astype(np.uint8)
            if foreground.sum() > 0:
                break
    if foreground.sum() == 0:
        q = np.quantile(prob, 0.90)
        foreground = (prob >= q).astype(np.uint8)

    if dilate_px > 0:
        from scipy.ndimage import binary_dilation
        struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1))
        foreground = binary_dilation(foreground, structure=struct).astype(np.uint8)

    return 1 - foreground  # bg_mask


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--gpu", type=int, default=-1, help="-1 = CPU")
    args = p.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Loading CLIPSeg on {device}...")
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    model.eval()

    # Find all experiment dirs under analysis/reinforce_analysis/*/experiments/
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
            prompts_path = os.path.join(exp_dir, "prompts.txt")
            if not os.path.isfile(source_path):
                print(f"  {exp_name}: no source_b0.jpg")
                failed += 1
                continue
            prompts = parse_prompts_txt(prompts_path)
            seg_prompt = prompts.get("seg", "")
            if not seg_prompt:
                # Fallback: derive from experiment name
                if "catdog" in exp_name or exp_name.startswith("catdog"):
                    seg_prompt = "cat"
                elif "car_taxi" in exp_name:
                    seg_prompt = "car"
                elif "sunflower" in exp_name:
                    seg_prompt = "sunflower"
                elif "chair_throne" in exp_name:
                    seg_prompt = "chair"
                elif "penguin" in exp_name:
                    seg_prompt = "penguin"
                elif "cake_books" in exp_name:
                    seg_prompt = "cake"
                elif "lighthouse" in exp_name:
                    seg_prompt = "lighthouse"
                elif "violin" in exp_name:
                    seg_prompt = "violin"
                elif exp_name.startswith("reinforce_horse"):
                    seg_prompt = "horse"
                elif "room" in exp_name:
                    seg_prompt = "wooden coffee table"
                elif "snow_volcano" in exp_name:
                    seg_prompt = "mountain peak"
                elif "butterfly" in exp_name:
                    seg_prompt = "butterfly"
                elif "sail_pirate" in exp_name:
                    seg_prompt = "sailboat"
                elif "teapot_globe" in exp_name:
                    seg_prompt = "teapot"
                elif "candle_crystal" in exp_name:
                    seg_prompt = "candle"
                elif "typewriter_laptop" in exp_name:
                    seg_prompt = "typewriter"

            if not seg_prompt:
                print(f"  {exp_name}: unknown seg_prompt")
                failed += 1
                continue

            try:
                source_pil = Image.open(source_path).convert("RGB")
                bg_mask = compute_mask_from_source(source_pil, seg_prompt, device, model, processor)
                vis = build_vis(source_pil, bg_mask)
                vis.save(vis_path, quality=92)
                fg_pct = 100 * (bg_mask < 0.5).mean()
                print(f"  {exp_name}: seg='{seg_prompt}' fg={fg_pct:.1f}%")
                done += 1
            except Exception as e:
                print(f"  {exp_name}: FAILED — {e}")
                failed += 1

    print(f"\nTotal: {total}, generated: {done}, skipped (already have): {skipped}, failed: {failed}")


if __name__ == "__main__":
    main()
