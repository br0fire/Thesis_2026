"""Compare v1 (delta reward, collapsed) vs v2 (relative reward) for the 4 failed experiments.

Produces:
  - v1_vs_v2.csv: side-by-side metrics
  - v1_vs_v2_grid.jpg: per-experiment side-by-side top-K grid
  - v1_vs_v2_probs.png: overlaid bit probability bar chart
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

NFS3 = "/home/jovyan/shares/SR006.nfs3/svgrozny"
OUT_DIR = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"

PAIRS = [
    # (v1_name, v2_name, human_label)
    ("test_catdog", "catdog_v2", "catdog"),
    ("car_taxi", "car_taxi_v2", "car_taxi"),
    ("sunflower_lavender", "sunflower_lavender_v2", "sunflower_lavender"),
    ("chair_throne", "chair_throne_v2", "chair_throne"),
    ("penguin_flamingo", "penguin_flamingo_v2", "penguin_flamingo"),
    ("cake_books", "cake_books_v2", "cake_books"),
    ("lighthouse_castle", "lighthouse_castle_v2", "lighthouse_castle"),
    ("violin_guitar", "violin_guitar_v2", "violin_guitar"),
    ("horse", "horse_v2", "horse"),
    ("room", "room_v2", "room"),
    ("snow_volcano", "snow_volcano_v2", "snow_volcano"),
    ("butterfly_hummingbird", "butterfly_hummingbird_v2", "butterfly_hummingbird"),
    ("sail_pirate", "sail_pirate_v2", "sail_pirate"),
]


def load(name):
    d = os.path.join(NFS3, f"reinforce_{name}")
    ckpt_path = os.path.join(d, "reinforce_result.pt")
    csv_path = os.path.join(d, "reinforce_log.csv")
    if not os.path.isfile(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    df = pd.read_csv(csv_path) if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0 else None
    return {"dir": d, "ckpt": ckpt, "df": df}


def font(size, bold=False):
    path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def build_comparison_csv(out_path):
    rows = []
    for v1, v2, label in PAIRS:
        a = load(v1)
        b = load(v2)
        row = {"experiment": label}
        for tag, exp in [("v1", a), ("v2", b)]:
            if exp is None:
                row[f"{tag}_best_reward"] = np.nan
                row[f"{tag}_final_bg"] = np.nan
                row[f"{tag}_final_fg"] = np.nan
                row[f"{tag}_total_images"] = np.nan
                row[f"{tag}_final_entropy"] = np.nan
                row[f"{tag}_status"] = "missing"
                continue
            ckpt = exp["ckpt"]
            df = exp["df"]
            row[f"{tag}_best_reward"] = ckpt.get("best_reward", np.nan)
            row[f"{tag}_total_images"] = ckpt.get("total_images", np.nan)
            if df is not None and len(df) > 0:
                row[f"{tag}_final_bg"] = df["mean_bg_ssim"].iloc[-10:].mean()
                row[f"{tag}_final_fg"] = df["mean_fg_clip"].iloc[-10:].mean()
                row[f"{tag}_final_entropy"] = df["entropy"].iloc[-1]
            row[f"{tag}_status"] = "done"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(df.to_string(index=False))
    return df


def build_probs_plot(out_path):
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(12, 3 * len(PAIRS)))
    if len(PAIRS) == 1:
        axes = [axes]
    n_bits = 14
    x = np.arange(n_bits)

    for ax, (v1, v2, label) in zip(axes, PAIRS):
        a = load(v1)
        b = load(v2)
        if a is not None:
            p1 = a["ckpt"]["probs"]
            if torch.is_tensor(p1):
                p1 = p1.numpy()
            ax.bar(x - 0.2, p1, width=0.4, label=f"v1 (delta reward)",
                   color="#cc5555", alpha=0.8)
        if b is not None:
            p2 = b["ckpt"]["probs"]
            if torch.is_tensor(p2):
                p2 = p2.numpy()
            ax.bar(x + 0.2, p2, width=0.4, label=f"v2 (relative reward)",
                   color="#5599cc", alpha=0.8)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"b{i}" for i in range(n_bits)])
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(target)")
        ax.set_title(f"{label} — learned bit probabilities")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _diff_words(a, b):
    import difflib
    aw = (a or "").split()
    bw = (b or "").split()
    at = [(w, True) for w in aw]
    bt = [(w, True) for w in bw]
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                at[k] = (aw[k], False)
            for k in range(j1, j2):
                bt[k] = (bw[k], False)
    return at, bt


def _text_w(draw, text, fnt):
    try:
        return draw.textlength(text, font=fnt)
    except Exception:
        bb = draw.textbbox((0, 0), text, font=fnt)
        return bb[2] - bb[0]


def _draw_colored(draw, x, y, tagged, fnt, max_w, line_h):
    if not tagged:
        return y
    cur_x, cur_y = x, y
    sp = _text_w(draw, " ", fnt) or 4
    for word, is_diff in tagged:
        ww = _text_w(draw, word, fnt)
        if cur_x + ww > x + max_w and cur_x > x:
            cur_x = x
            cur_y += line_h
        draw.text((cur_x, cur_y), word,
                  fill=(200, 0, 0) if is_diff else (60, 60, 60), font=fnt)
        cur_x += ww + sp
    return cur_y + line_h


def build_side_by_side_grid(out_path, top_k=5):
    """For each pair, show source + target + top-K from v1 + top-K from v2 in one strip.
    Includes a prompt-diff row (source / target in red-highlighted text)."""
    thumb = 180
    gap = 8
    strip_h = 30
    label_h = 32
    header_h = 40
    prompt_h = 60  # room for source+target prompt with colored diff
    row_h = header_h + prompt_h + thumb + strip_h + label_h + 20
    # Columns: SRC + TGT + 1 separator + top_k v1 + 1 separator + top_k v2
    cols = 2 + 1 + top_k + 1 + top_k
    canvas_w = cols * (thumb + gap) + 20
    canvas_h = len(PAIRS) * row_h + 30

    canvas = Image.new("RGB", (canvas_w, canvas_h), (252, 252, 252))
    draw = ImageDraw.Draw(canvas)
    title_font = font(16, bold=True)
    label_font = font(11, bold=True)
    meta_font = font(10)
    prompt_font = font(11)

    for row, (v1, v2, label) in enumerate(PAIRS):
        y0 = row * row_h + 20
        # Title row
        a = load(v1)
        b = load(v2)
        v1_r = a["ckpt"].get("best_reward", 0) if a else 0
        v2_r = b["ckpt"].get("best_reward", 0) if b else 0
        v2_status = "✓" if b else "waiting"
        title = f"{label}   v1 best={v1_r:.4f}   v2 ({v2_status}) best={v2_r:.4f}"
        draw.text((20, y0), title, fill="black", font=title_font)

        # Prompt diff row (source / target with red-highlighted unique words)
        src_prompt = ""
        tgt_prompt = ""
        for exp in (b, a):
            if exp and "prompts" not in exp:
                exp_dir = exp["dir"]
                pf = os.path.join(exp_dir, "prompts.txt")
                if os.path.isfile(pf):
                    with open(pf) as f:
                        for line in f:
                            if line.startswith("source:") and not src_prompt:
                                src_prompt = line.split(":", 1)[1].strip()
                            elif line.startswith("target:") and not tgt_prompt:
                                tgt_prompt = line.split(":", 1)[1].strip()
            if exp and "ckpt" in exp and isinstance(exp["ckpt"].get("args"), dict):
                args = exp["ckpt"]["args"]
                if not src_prompt:
                    src_prompt = args.get("source_prompt", "")
                if not tgt_prompt:
                    tgt_prompt = args.get("target_prompt", "")
        src_tag, tgt_tag = _diff_words(src_prompt, tgt_prompt)
        col_w = (canvas_w - 40) // 2
        y_prompts = y0 + 24
        draw.text((20, y_prompts - 14), "SRC:", fill="black", font=label_font)
        _draw_colored(draw, 60, y_prompts - 14, src_tag, prompt_font, col_w - 60, 14)
        draw.text((20 + col_w, y_prompts - 14), "TGT:", fill="black", font=label_font)
        _draw_colored(draw, 60 + col_w, y_prompts - 14, tgt_tag, prompt_font, col_w - 60, 14)

        y_img = y0 + header_h + prompt_h
        x = 10

        def paste(path, sub_label, x):
            if path and os.path.isfile(path):
                try:
                    im = Image.open(path).convert("RGB")
                    im.thumbnail((thumb, thumb))
                    canvas.paste(im, (x + (thumb - im.width) // 2, y_img))
                except Exception:
                    pass
            else:
                draw.rectangle([x, y_img, x + thumb, y_img + thumb], outline="lightgray")
            draw.text((x + 4, y_img + thumb + 4), sub_label, fill="black", font=label_font)

        # SRC & TGT from v2 if available, else v1
        src_src = os.path.join((b or a or {"dir": ""})["dir"], "source_b0.jpg") if (a or b) else None
        n_bits = 14
        tgt_path = os.path.join((b or a or {"dir": ""})["dir"], f"target_b{(1 << n_bits) - 1}.jpg") if (a or b) else None
        if not (tgt_path and os.path.isfile(tgt_path)):
            tgt_path = None
        paste(src_src, "SOURCE", x); x += thumb + gap
        paste(tgt_path, "TARGET", x); x += thumb + gap

        # Separator
        x += 8

        # v1 top-K
        if a:
            v1_files = sorted([f for f in os.listdir(a["dir"])
                               if f.startswith("reinforce_top") and f.endswith(".jpg")])[:top_k]
            for i, f in enumerate(v1_files):
                try:
                    r_str = f.split("_r")[1].split("_")[0]
                except Exception:
                    r_str = "?"
                paste(os.path.join(a["dir"], f), f"v1 #{i}\nr={r_str}", x)
                x += thumb + gap
        else:
            x += (thumb + gap) * top_k

        # Separator
        x += 8

        # v2 top-K
        if b:
            v2_files = sorted([f for f in os.listdir(b["dir"])
                               if f.startswith("reinforce_top") and f.endswith(".jpg")])[:top_k]
            for i, f in enumerate(v2_files):
                try:
                    r_str = f.split("_r")[1].split("_")[0]
                except Exception:
                    r_str = "?"
                paste(os.path.join(b["dir"], f), f"v2 #{i}\nr={r_str}", x)
                x += thumb + gap

    canvas.save(out_path, quality=92)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    build_comparison_csv(os.path.join(OUT_DIR, "v1_vs_v2.csv"))
    build_probs_plot(os.path.join(OUT_DIR, "v1_vs_v2_probs.png"))
    build_side_by_side_grid(os.path.join(OUT_DIR, "v1_vs_v2_grid.jpg"))


if __name__ == "__main__":
    main()
