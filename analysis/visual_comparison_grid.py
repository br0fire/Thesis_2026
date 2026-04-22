"""Visual grid: for each experiment, show source + target + best images from each method.

Helps sanity-check whether reward correlates with perceived editing quality.
"""
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ANALYSIS = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"
CELL = 220
LABEL_H = 36

COLS = [
    ("source",         "Source"),
    ("target",         "Target (all-1s)"),
    ("reinforce",      "REINFORCE@640"),
    ("cem160",         "CEM@160"),
    ("cem80",          "CEM@80"),
    ("cem40",          "CEM@40"),
    ("amort_oracle",   "Oracle probs @1"),
    ("amort_mlp",      "MLP @1"),
    ("amort_popmean",  "PopMean @1"),
]


def find_reinforce_best(sweep_dir):
    """Look for reinforce_top0_*.jpg in sweep alpha_high config or new_bgrich."""
    cfg_dir = os.path.join(sweep_dir, "configs", "alpha_high")
    if os.path.isdir(cfg_dir):
        tops = sorted(f for f in os.listdir(cfg_dir) if f.startswith("reinforce_top0_"))
        if tops:
            return os.path.join(cfg_dir, tops[0])
    return None


def find_new_bgrich_best(exp_name):
    exp_dir = os.path.join(ANALYSIS, "new_bgrich", exp_name)
    if os.path.isdir(exp_dir):
        tops = sorted(f for f in os.listdir(exp_dir) if f.startswith("reinforce_top0_"))
        if tops:
            return os.path.join(exp_dir, tops[0])
    return None


def find_v_series_dir(exp_name):
    """Return latest vN experiments/reinforce_<name>_vN dir for this experiment."""
    for version in sorted(os.listdir(ANALYSIS), reverse=True):
        if not re.match(r"v\d+[a-z]*$", version):
            continue
        d = os.path.join(ANALYSIS, version, "experiments", f"reinforce_{exp_name}_{version}")
        if os.path.isdir(d):
            return d
    return None


def find_v_series_best(exp_name):
    d = find_v_series_dir(exp_name)
    if not d:
        return None
    tops = sorted(f for f in os.listdir(d) if f.startswith("reinforce_top0_"))
    return os.path.join(d, tops[0]) if tops else None


def find_cem_best(exp_name, budget):
    d = os.path.join(ANALYSIS, "cem", f"{exp_name}_budget{budget}")
    if not os.path.isdir(d):
        return None
    tops = sorted(f for f in os.listdir(d) if f.startswith("cem_top0_"))
    return os.path.join(d, tops[0]) if tops else None


def find_amortized(exp_name, strategy):
    d = os.path.join(ANALYSIS, "amortized", exp_name)
    if not os.path.isdir(d):
        return None
    files = [f for f in os.listdir(d) if f.startswith(f"{strategy}_r") and f.endswith(".jpg")]
    return os.path.join(d, files[0]) if files else None


def extract_reward(fname):
    m = re.search(r"_r([\d.-]+)_", fname)
    return float(m.group(1)) if m else None


def label_image(pil_img, label_top, label_bot="", size=CELL):
    img = pil_img.resize((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size + LABEL_H), (245, 245, 245))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        f = ImageFont.load_default()
    draw.text((4, size + 2), label_top, fill=(0, 0, 0), font=f)
    if label_bot:
        draw.text((4, size + 16), label_bot, fill=(80, 80, 80), font=f)
    return canvas


def build_row(exp_name, height=CELL + LABEL_H):
    cells = []
    # Figure out source dir: sweep → new_bgrich → v-series
    sweep_dir = os.path.join(ANALYSIS, f"sweep_{exp_name}")
    new_bgrich_dir = os.path.join(ANALYSIS, "new_bgrich", exp_name)
    v_dir = find_v_series_dir(exp_name)
    for cand in (sweep_dir, new_bgrich_dir, v_dir):
        if cand and os.path.isfile(os.path.join(cand, "source_b0.jpg")):
            primary_dir = cand
            break
    else:
        primary_dir = None

    # Source
    if primary_dir:
        src_pil = Image.open(os.path.join(primary_dir, "source_b0.jpg")).convert("RGB")
    else:
        src_pil = Image.new("RGB", (CELL, CELL), (200, 200, 200))
    cells.append(label_image(src_pil, exp_name, "source"))

    # Target all-ones
    tgt_files = [f for f in os.listdir(primary_dir) if f.startswith("target_b") and f.endswith(".jpg")] \
                if primary_dir else []
    if tgt_files:
        tgt_pil = Image.open(os.path.join(primary_dir, tgt_files[0])).convert("RGB")
    else:
        tgt_pil = Image.new("RGB", (CELL, CELL), (200, 200, 200))
    cells.append(label_image(tgt_pil, "target (all-1s)", ""))

    # REINFORCE best: sweep → new_bgrich → v-series
    rein = find_reinforce_best(sweep_dir) or find_new_bgrich_best(exp_name) or find_v_series_best(exp_name)
    if rein:
        r = extract_reward(os.path.basename(rein))
        cells.append(label_image(Image.open(rein).convert("RGB"),
                                 "REINFORCE@640", f"R={r:.3f}" if r else ""))
    else:
        cells.append(label_image(Image.new("RGB", (CELL, CELL), (230, 230, 230)), "REINFORCE@640", "—"))

    # CEM at 160, 80, 40
    for b in (160, 80, 40):
        cem = find_cem_best(exp_name, b)
        if cem:
            r = extract_reward(os.path.basename(cem))
            cells.append(label_image(Image.open(cem).convert("RGB"),
                                     f"CEM@{b}", f"R={r:.3f}" if r else ""))
        else:
            cells.append(label_image(Image.new("RGB", (CELL, CELL), (230, 230, 230)), f"CEM@{b}", "—"))

    # Amortized: oracle, mlp, popmean
    for strategy, label in [("oracle", "Oracle@1"), ("mlp", "MLP@1"), ("popmean", "PopMean@1")]:
        a = find_amortized(exp_name, strategy)
        if a:
            r = extract_reward(os.path.basename(a))
            cells.append(label_image(Image.open(a).convert("RGB"),
                                     label, f"R={r:.3f}" if r else ""))
        else:
            cells.append(label_image(Image.new("RGB", (CELL, CELL), (230, 230, 230)), label, "—"))

    # Compose row
    row_w = sum(c.width for c in cells) + (len(cells) - 1) * 4
    row_h = cells[0].height
    row = Image.new("RGB", (row_w, row_h), (220, 220, 220))
    x = 0
    for c in cells:
        row.paste(c, (x, 0))
        x += c.width + 4
    return row


def prompts_agree(exp_name):
    """Check that CEM and REINFORCE-source prompts are identical for this experiment."""
    sweep_prompts = os.path.join(ANALYSIS, f"sweep_{exp_name}", "configs", "alpha_high", "prompts.txt")
    new_prompts = os.path.join(ANALYSIS, "new_bgrich", exp_name, "prompts.txt")
    cem_prompts = os.path.join(ANALYSIS, "cem", f"{exp_name}_budget160", "prompts.txt")
    if not os.path.isfile(cem_prompts):
        return False
    # Read CEM source
    def read_source(p):
        if not os.path.isfile(p):
            return None
        with open(p) as f:
            for line in f:
                if line.startswith("source:"):
                    return line.split(":", 1)[1].strip()
        return None
    cem_src = read_source(cem_prompts)
    ref_src = read_source(sweep_prompts) or read_source(new_prompts)
    if ref_src is None:
        # Fall back to v-series
        for version in sorted(os.listdir(ANALYSIS), reverse=True):
            if not re.match(r"v\d+[a-z]*$", version):
                continue
            vp = os.path.join(ANALYSIS, version, "experiments", f"reinforce_{exp_name}_{version}", "prompts.txt")
            if os.path.isfile(vp):
                ref_src = read_source(vp)
                break
    return cem_src is not None and ref_src is not None and cem_src == ref_src


def main():
    # Select experiments where CEM and REINFORCE prompts match
    candidates = sorted(os.listdir(os.path.join(ANALYSIS, "amortized")))
    candidates = [c for c in candidates if os.path.isfile(
        os.path.join(ANALYSIS, "amortized", c, "eval.json"))]
    before = len(candidates)
    candidates = [c for c in candidates if prompts_agree(c)]
    print(f"Building grid for {len(candidates)}/{before} experiments (excluded {before - len(candidates)} with prompt mismatch)")

    rows = [build_row(name) for name in candidates]
    W = max(r.width for r in rows)
    total_h = sum(r.height + 6 for r in rows)
    canvas = Image.new("RGB", (W, total_h), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + 6

    out_path = os.path.join(ANALYSIS, "visual_comparison_grid.png")
    canvas.save(out_path, quality=88)
    print(f"Saved: {out_path}  ({W}x{total_h})")


if __name__ == "__main__":
    main()
