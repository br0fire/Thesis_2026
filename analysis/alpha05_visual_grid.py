"""Visual comparison grid for α=0.5 only: shows best edits from each method.

Columns: source | target (all-1s) | REINFORCE 14-bit | REINFORCE 28-bit |
         CEM@40 | CEM@160 | Exhaustive 14-bit top-0 (raw float)
Rows: 15 new_bgrich experiments.
"""
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ANALYSIS = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"
CELL = 220
LABEL_H = 40

_SRC = os.path.join(ANALYSIS, "new_bgrich_alpha05")
EXPS = sorted(d for d in os.listdir(_SRC) if os.path.isdir(os.path.join(_SRC, d)))


def find_file(dir_path, pattern):
    if not os.path.isdir(dir_path):
        return None
    files = sorted(f for f in os.listdir(dir_path) if re.match(pattern, f))
    return os.path.join(dir_path, files[0]) if files else None


def extract_reward(fname):
    m = re.search(r"_r([\d.-]+)_", fname)
    return float(m.group(1)) if m else None


def label_image(pil_img, top, bottom=""):
    img = pil_img.resize((CELL, CELL), Image.LANCZOS)
    canvas = Image.new("RGB", (CELL, CELL + LABEL_H), (245, 245, 245))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        f = ImageFont.load_default()
    draw.text((4, CELL + 2), top, fill=(0, 0, 0), font=f)
    if bottom:
        draw.text((4, CELL + 18), bottom, fill=(80, 80, 80), font=f)
    return canvas


def placeholder(text):
    p = Image.new("RGB", (CELL, CELL), (230, 230, 230))
    return label_image(p, text, "—")


def build_row(exp):
    cells = []
    d14 = os.path.join(ANALYSIS, "new_bgrich_alpha05", exp)
    d28 = os.path.join(ANALYSIS, "new_bgrich_28bit_alpha05", exp)
    cem_dirs = {b: os.path.join(ANALYSIS, "cem", f"{exp}_budget{b}") for b in (40, 160)}
    exh_dir = os.path.join(ANALYSIS, "exhaustive", exp)

    # 1) Source (take from new_bgrich_alpha05)
    src = os.path.join(d14, "source_b0.jpg")
    if os.path.isfile(src):
        cells.append(label_image(Image.open(src).convert("RGB"), exp, "source"))
    else:
        cells.append(placeholder("source"))

    # 2) Target (all-1s)
    tgt_files = [f for f in os.listdir(d14) if f.startswith("target_b")] if os.path.isdir(d14) else []
    if tgt_files:
        cells.append(label_image(Image.open(os.path.join(d14, tgt_files[0])).convert("RGB"),
                                 "target (all-1s)", ""))
    else:
        cells.append(placeholder("target"))

    # 3) REINFORCE 14-bit α=0.5 best
    r14 = find_file(d14, r"reinforce_top0_")
    if r14:
        r = extract_reward(os.path.basename(r14))
        cells.append(label_image(Image.open(r14).convert("RGB"),
                                 "REINFORCE 14-bit α=0.5", f"R={r:.3f}" if r else ""))
    else:
        cells.append(placeholder("REINFORCE 14-bit"))

    # 4) REINFORCE 28-bit α=0.5 best
    r28 = find_file(d28, r"reinforce_top0_")
    if r28:
        r = extract_reward(os.path.basename(r28))
        cells.append(label_image(Image.open(r28).convert("RGB"),
                                 "REINFORCE 28-bit α=0.5", f"R={r:.3f}" if r else ""))
    else:
        cells.append(placeholder("REINFORCE 28-bit"))

    # 5) CEM@40 α=0.5
    c40 = find_file(cem_dirs[40], r"cem_top0_")
    if c40:
        r = extract_reward(os.path.basename(c40))
        cells.append(label_image(Image.open(c40).convert("RGB"),
                                 "CEM@40 α=0.5", f"R={r:.3f}" if r else ""))
    else:
        cells.append(placeholder("CEM@40"))

    # 6) CEM@160 α=0.5
    c160 = find_file(cem_dirs[160], r"cem_top0_")
    if c160:
        r = extract_reward(os.path.basename(c160))
        cells.append(label_image(Image.open(c160).convert("RGB"),
                                 "CEM@160 α=0.5", f"R={r:.3f}" if r else ""))
    else:
        cells.append(placeholder("CEM@160"))

    # 7) Exhaustive 14-bit α=0.5 best
    exh = find_file(exh_dir, r"exhaustive_top0_")
    if exh:
        r = extract_reward(os.path.basename(exh))
        cells.append(label_image(Image.open(exh).convert("RGB"),
                                 "Exhaustive 14-bit α=0.5", f"R={r:.3f}" if r else ""))
    else:
        cells.append(placeholder("Exhaustive 14-bit"))

    # Compose row
    row_w = sum(c.width for c in cells) + (len(cells) - 1) * 4
    row_h = cells[0].height
    row = Image.new("RGB", (row_w, row_h), (220, 220, 220))
    x = 0
    for c in cells:
        row.paste(c, (x, 0))
        x += c.width + 4
    return row


def main():
    rows = [build_row(exp) for exp in EXPS]
    W = max(r.width for r in rows)
    H = sum(r.height + 6 for r in rows)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + 6
    out_path = os.path.join(ANALYSIS, "alpha05_visual_grid.png")
    canvas.save(out_path, quality=88)
    print(f"Saved {out_path}  ({W}x{H})  {len(rows)} rows")


if __name__ == "__main__":
    main()
