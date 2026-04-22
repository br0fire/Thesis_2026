"""Compare segmentation: GDino+SAM2 (current) vs SAM 3.1 across ALL source images.

For every unique source_b0.jpg under analysis/reinforce_analysis (deduplicating by
experiment name, keeping the most recent version), runs both segmenters, measures
time, and builds a side-by-side visualization + stats CSV.

Usage:
  HF_TOKEN=... python analysis/compare_sam3.py --gpu 0
"""
import argparse
import csv
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))
from reinforce_search import _segment_gdino_sam

ANALYSIS = os.path.join(PROJECT_ROOT, "analysis/reinforce_analysis")


def parse_prompts(p):
    out = {}
    with open(p) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def find_prompts_txt(exp_dir):
    """Look for prompts.txt in the dir itself or in configs/*/prompts.txt."""
    p = os.path.join(exp_dir, "prompts.txt")
    if os.path.isfile(p):
        return p
    configs_dir = os.path.join(exp_dir, "configs")
    if os.path.isdir(configs_dir):
        for cfg in sorted(os.listdir(configs_dir)):
            cp = os.path.join(configs_dir, cfg, "prompts.txt")
            if os.path.isfile(cp):
                return cp
    return None


def canonical_name(root):
    """Derive the experiment name from a directory path, stripping sweep/config/version layers."""
    rel = os.path.relpath(root, ANALYSIS)
    parts = rel.split(os.sep)

    # sweep_<exp>/configs/<cfg>  → use <exp>
    # sweep_<exp>                → use <exp>
    if parts[0].startswith("sweep_"):
        return parts[0][len("sweep_"):]
    # nbits20/<exp>[/...]        → use <exp>
    if parts[0] == "nbits20" and len(parts) >= 2:
        return parts[1]
    # v<N>/experiments/reinforce_<exp>_v<N> → strip prefixes/suffix
    base = parts[-1]
    base = re.sub(r"^reinforce_", "", base)
    base = re.sub(r"_v\d+[a-z]*$", "", base)
    return base


def collect_source_images():
    """Find every source_b0.jpg, dedup by canonical experiment name, keep newest mtime."""
    tasks = {}
    for root, _, files in os.walk(ANALYSIS):
        if "source_b0.jpg" not in files:
            continue
        src = os.path.join(root, "source_b0.jpg")
        prompts_path = find_prompts_txt(root)
        if not prompts_path:
            continue
        prompts = parse_prompts(prompts_path)
        seg = prompts.get("seg")
        if not seg:
            continue

        base = canonical_name(root)
        mtime = os.path.getmtime(src)
        if base not in tasks or mtime > tasks[base]["mtime"]:
            tasks[base] = {
                "name": base,
                "src": src,
                "seg": seg,
                "mtime": mtime,
                "src_root": root,
            }
    return sorted(tasks.values(), key=lambda t: t["name"])


def segment_sam3(processor, pil_image, text_prompt):
    state = processor.set_image(pil_image)
    output = processor.set_text_prompt(state=state, prompt=text_prompt)
    masks = output.get("masks")
    if masks is None or len(masks) == 0:
        H, W = pil_image.size[1], pil_image.size[0]
        return np.zeros((H, W), dtype=np.float32)
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks).squeeze()
    if masks.ndim == 2:
        return (masks > 0.5).astype(np.float32)
    combined = np.zeros(masks.shape[-2:], dtype=np.float32)
    for m in masks:
        combined = np.maximum(combined, (m > 0.5).astype(np.float32))
    return combined


def overlay_mask(pil_image, mask, color=(255, 0, 0), alpha=0.45):
    arr = np.array(pil_image.convert("RGB"))
    H, W = arr.shape[:2]
    if mask.shape != (H, W):
        m_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
        mask = (np.array(m_pil) > 127).astype(np.float32)
    fg = mask > 0.5
    out = arr.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    out[fg] = out[fg] * (1 - alpha) + color_arr * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def add_caption(pil_img, text, height=40):
    W = pil_img.width
    canvas = Image.new("RGB", (W, pil_img.height + height), (255, 255, 255))
    canvas.paste(pil_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.multiline_text((6, pil_img.height + 4), text, fill=(0, 0, 0), font=font)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ANALYSIS, "sam3_vs_gdino_full.png"))
    p.add_argument("--stats", default=os.path.join(ANALYSIS, "sam3_vs_gdino_stats.csv"))
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    tasks = collect_source_images()
    print(f"Found {len(tasks)} unique experiments to compare")

    # Load SAM3 once
    print("Loading SAM 3.1...", flush=True)
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be set in the environment before loading SAM 3.")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sam3_model = build_sam3_image_model(device=str(device))
    sam3_model = sam3_model.to(device).eval()
    sam3_processor = Sam3Processor(sam3_model)

    stats = []
    rows = []
    for t in tasks:
        print(f"\n=== {t['name']}  (seg='{t['seg']}') ===", flush=True)
        src_pil = Image.open(t["src"]).convert("RGB")

        # GDino+SAM2 timing
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        try:
            fg_gdino = _segment_gdino_sam(src_pil, t["seg"], device)
            if torch.is_tensor(fg_gdino):
                fg_gdino = fg_gdino.cpu().numpy()
            fg_gdino = np.asarray(fg_gdino)
        except Exception as e:
            print(f"  GDino+SAM2 FAILED: {e}")
            fg_gdino = np.zeros_like(np.array(src_pil)[:, :, 0], dtype=np.float32)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        dt_gdino = time.perf_counter() - t0
        fg_gdino_pct = 100.0 * (fg_gdino > 0.5).mean()

        # SAM3 timing
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        try:
            fg_sam3 = segment_sam3(sam3_processor, src_pil, t["seg"])
        except Exception as e:
            print(f"  SAM3 FAILED: {e}")
            fg_sam3 = np.zeros_like(np.array(src_pil)[:, :, 0], dtype=np.float32)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        dt_sam3 = time.perf_counter() - t0
        fg_sam3_pct = 100.0 * (fg_sam3 > 0.5).mean()

        print(f"  GDino+SAM2: {fg_gdino_pct:5.1f}% fg  ({dt_gdino:.2f}s)")
        print(f"  SAM 3.1:    {fg_sam3_pct:5.1f}% fg  ({dt_sam3:.2f}s)")
        stats.append({
            "name": t["name"], "seg_prompt": t["seg"],
            "gdino_fg_pct": round(fg_gdino_pct, 2), "gdino_time_s": round(dt_gdino, 3),
            "sam3_fg_pct": round(fg_sam3_pct, 2),  "sam3_time_s": round(dt_sam3, 3),
        })

        src_thumb = src_pil.resize((256, 256), Image.LANCZOS)
        ov_gdino = overlay_mask(src_thumb, fg_gdino).resize((256, 256))
        ov_sam3 = overlay_mask(src_thumb, fg_sam3).resize((256, 256))

        src_cap = add_caption(src_thumb, f"{t['name']}\nseg='{t['seg']}'", height=40)
        g_cap = add_caption(ov_gdino, f"GDino+SAM2 fg={fg_gdino_pct:.1f}%\n{dt_gdino:.2f}s", height=40)
        s_cap = add_caption(ov_sam3, f"SAM 3.1 fg={fg_sam3_pct:.1f}%\n{dt_sam3:.2f}s", height=40)

        row_w = src_cap.width + g_cap.width + s_cap.width + 8
        row_h = src_cap.height
        row = Image.new("RGB", (row_w, row_h), (240, 240, 240))
        row.paste(src_cap, (0, 0))
        row.paste(g_cap, (src_cap.width + 4, 0))
        row.paste(s_cap, (src_cap.width + g_cap.width + 8, 0))
        rows.append(row)

    # Save stats CSV
    with open(args.stats, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    print(f"\nStats: {args.stats}")

    # Totals
    total_gdino = sum(s["gdino_time_s"] for s in stats)
    total_sam3 = sum(s["sam3_time_s"] for s in stats)
    mean_gdino = total_gdino / len(stats)
    mean_sam3 = total_sam3 / len(stats)
    print(f"\n=== SPEED ===")
    print(f"GDino+SAM2: total {total_gdino:.1f}s  mean {mean_gdino:.2f}s/image")
    print(f"SAM 3.1:    total {total_sam3:.1f}s  mean {mean_sam3:.2f}s/image")
    print(f"Speedup: {total_gdino/total_sam3:.2f}x ({'SAM3 faster' if total_sam3<total_gdino else 'GDino faster'})")

    # Stack grid
    W = max(r.width for r in rows)
    # Add header strip with speed summary
    header_h = 80
    total_h = header_h + sum(r.height + 4 for r in rows)
    canvas = Image.new("RGB", (W, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        hfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        hfont = ImageFont.load_default()
    header_txt = (f"SAM 3.1 vs GDino+SAM2  ({len(stats)} experiments)\n"
                  f"GDino+SAM2: mean {mean_gdino:.2f}s   |   "
                  f"SAM 3.1: mean {mean_sam3:.2f}s   |   "
                  f"Speedup: {total_gdino/total_sam3:.2f}x")
    draw.multiline_text((10, 10), header_txt, fill=(0, 0, 0), font=hfont)
    y = header_h
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + 4
    canvas.save(args.out, quality=90)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
