"""
Summarize and visualize REINFORCE experiment results.

Produces:
  - summary.csv: one row per experiment with key metrics
  - training_curves.png: reward/bg/fg/entropy over episodes for all experiments
  - learned_probs.png: heatmap of learned bit probabilities per experiment
  - top_images_grids/: per-experiment grids of top-K generated images with source
"""
import os
import csv
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

NFS3 = "/home/jovyan/shares/SR006.nfs3/svgrozny"
LOGS_DIR = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/logs"
OUT_DIR = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/analysis/reinforce_analysis"

EXPERIMENTS = [
    "test_catdog", "car_taxi", "sunflower_lavender", "chair_throne",
    "penguin_flamingo", "cake_books", "lighthouse_castle", "violin_guitar",
    "horse", "room", "snow_volcano", "butterfly_hummingbird", "sail_pirate",
    # v2 reruns with fixed reward (all 13 pairs)
    "catdog_v2", "car_taxi_v2", "sunflower_lavender_v2", "chair_throne_v2",
    "penguin_flamingo_v2", "cake_books_v2", "lighthouse_castle_v2", "violin_guitar_v2",
    "horse_v2", "room_v2", "snow_volcano_v2", "butterfly_hummingbird_v2", "sail_pirate_v2",
]


def load_experiment(name):
    """Load experiment data if it exists. Returns dict or None."""
    exp_dir = os.path.join(NFS3, f"reinforce_{name}")
    csv_path = os.path.join(exp_dir, "reinforce_log.csv")
    ckpt_path = os.path.join(exp_dir, "reinforce_result.pt")

    if not os.path.isfile(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    df = pd.read_csv(csv_path) if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0 else None

    # Load prompts from args in checkpoint (or from prompts.txt if present)
    prompts = {"source": "", "target": ""}
    args = ckpt.get("args", {})
    if isinstance(args, dict):
        prompts["source"] = args.get("source_prompt", "")
        prompts["target"] = args.get("target_prompt", "")
    # Fallback / override from prompts.txt
    txt_path = os.path.join(exp_dir, "prompts.txt")
    if os.path.isfile(txt_path):
        with open(txt_path) as f:
            for line in f:
                if line.startswith("source:"):
                    prompts["source"] = line.split(":", 1)[1].strip()
                elif line.startswith("target:"):
                    prompts["target"] = line.split(":", 1)[1].strip()

    # Find source and target image files
    source_img = os.path.join(exp_dir, "source_b0.jpg")
    n_bits = ckpt.get("n_bits", 14)
    target_b = (1 << n_bits) - 1
    target_img = os.path.join(exp_dir, f"target_b{target_b}.jpg")
    if not os.path.isfile(target_img):
        # Legacy: no target image was saved, leave empty
        target_img = None

    return {
        "name": name,
        "dir": exp_dir,
        "ckpt": ckpt,
        "df": df,
        "prompts": prompts,
        "source_img": source_img if os.path.isfile(source_img) else None,
        "target_img": target_img,
    }


def build_summary(experiments):
    rows = []
    for exp in experiments:
        if exp is None:
            continue
        ckpt = exp["ckpt"]
        df = exp["df"]
        probs = ckpt["probs"].numpy() if torch.is_tensor(ckpt["probs"]) else ckpt["probs"]
        best_mask = ckpt.get("best_mask")
        best_mask_str = ""
        if best_mask is not None:
            best_mask_str = "".join(str(int(b)) for b in (best_mask.numpy() if torch.is_tensor(best_mask) else best_mask))

        row = {
            "name": exp["name"],
            "best_reward": ckpt.get("best_reward", np.nan),
            "total_images": ckpt.get("total_images", np.nan),
            "best_mask": best_mask_str,
        }
        if df is not None and len(df) > 0:
            row["n_episodes"] = len(df)
            row["final_mean_reward"] = df["mean_reward"].iloc[-1]
            row["final_bg_ssim"] = df["mean_bg_ssim"].iloc[-1]
            row["final_fg_clip"] = df["mean_fg_clip"].iloc[-1]
            row["final_entropy"] = df["entropy"].iloc[-1]
            row["initial_mean_reward"] = df["mean_reward"].iloc[0]
            row["reward_improvement"] = row["final_mean_reward"] - row["initial_mean_reward"]
        for i, p in enumerate(probs):
            row[f"prob_{i:02d}"] = float(p)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_training_curves(experiments, out_path):
    exps = [e for e in experiments if e is not None and e["df"] is not None]
    n = len(exps)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    colors = plt.cm.tab20(np.linspace(0, 1, n))

    for ax, col, title in zip(
        axes,
        ["mean_reward", "mean_bg_ssim", "mean_fg_clip", "entropy"],
        ["Mean reward per episode", "Mean bg_ssim per episode",
         "Mean fg_clip_score per episode", "Policy entropy"],
    ):
        for exp, c in zip(exps, colors):
            df = exp["df"]
            ax.plot(df["episode"], df[col], label=exp["name"], color=c, alpha=0.8, linewidth=1.2)
        ax.set_xlabel("Episode")
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_learned_probs(experiments, out_path):
    exps = [e for e in experiments if e is not None]
    n = len(exps)
    if n == 0:
        return

    n_bits = len(exps[0]["ckpt"]["probs"])
    matrix = np.zeros((n, n_bits))
    names = []
    for i, exp in enumerate(exps):
        probs = exp["ckpt"]["probs"]
        if torch.is_tensor(probs):
            probs = probs.numpy()
        matrix[i] = probs
        names.append(exp["name"])

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(n_bits))
    ax.set_xticklabels([f"b{i}" for i in range(n_bits)])
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)
    ax.set_xlabel("Bit position (0 = earliest diffusion step)")
    ax.set_title(f"Learned Bernoulli probabilities\n(blue = source, red = target)")

    for i in range(n):
        for j in range(n_bits):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(matrix[i, j] - 0.5) > 0.25 else "black",
                    fontsize=8)

    plt.colorbar(im, ax=ax, label="P(target)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _wrap_text(text, max_chars):
    """Simple greedy word wrap."""
    if not text:
        return [""]
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def _get_font(size, bold=False):
    font_path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                 else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def _diff_words(src_prompt, tgt_prompt):
    """Word-level diff via difflib. Returns (src_tagged, tgt_tagged) where each
    element is (word, is_different_bool). Equal blocks get False, insert/replace/delete True."""
    import difflib
    src_words = src_prompt.split() if src_prompt else []
    tgt_words = tgt_prompt.split() if tgt_prompt else []
    if not src_words and not tgt_words:
        return [], []

    matcher = difflib.SequenceMatcher(None, src_words, tgt_words, autojunk=False)
    src_tagged = [(w, True) for w in src_words]
    tgt_tagged = [(w, True) for w in tgt_words]
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                src_tagged[k] = (src_words[k], False)
            for k in range(j1, j2):
                tgt_tagged[k] = (tgt_words[k], False)
    return src_tagged, tgt_tagged


def _text_width(draw, text, font):
    try:
        return draw.textlength(text, font=font)
    except Exception:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]


def _draw_colored_wrapped(draw, x, y, words_tagged, font, max_width, line_h,
                          diff_color=(200, 0, 0), normal_color=(60, 60, 60)):
    """Render a tagged word list with word-wrap. Returns end_y (bottom of last line).
    Diff words drawn in bold red via `diff_color`, equal words in normal_color."""
    if not words_tagged:
        return y
    cur_x = x
    cur_y = y
    space_w = _text_width(draw, " ", font) or 4
    for word, is_diff in words_tagged:
        w_w = _text_width(draw, word, font)
        if cur_x + w_w > x + max_width and cur_x > x:
            cur_x = x
            cur_y += line_h
        color = diff_color if is_diff else normal_color
        draw.text((cur_x, cur_y), word, fill=color, font=font)
        cur_x += w_w + space_w
    return cur_y + line_h


def build_top_images_grid(exp, out_path, top_k=10):
    """Create a grid showing source/target images + prompts + top-K generated images."""
    exp_dir = exp["dir"]
    source_path = exp.get("source_img")
    target_path = exp.get("target_img")
    if not source_path:
        return

    src_prompt = exp["prompts"].get("source", "")
    tgt_prompt = exp["prompts"].get("target", "")
    src_tagged, tgt_tagged = _diff_words(src_prompt, tgt_prompt)

    # Find top-K reinforce images
    top_files = sorted([
        f for f in os.listdir(exp_dir) if f.startswith("reinforce_top") and f.endswith(".jpg")
    ])[:top_k + 1]
    if not top_files:
        return

    # Layout:
    #   [big title bar]
    #   [SOURCE | TARGET]  each with prompt text below, big thumbnails
    #   [top0 ... topN grid]
    big_thumb = 384
    thumb = 224
    cols = 6
    rows = (len(top_files) + cols - 1) // cols
    strip_h = 36
    prompt_h = 160  # more room for colored wrapped prompt text
    header_h = 56

    canvas_w = cols * thumb
    ref_row_h = big_thumb + prompt_h + 16
    top_grid_h = rows * (thumb + strip_h) + 30
    canvas_h = header_h + ref_row_h + top_grid_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (252, 252, 252))
    draw = ImageDraw.Draw(canvas)

    title_font = _get_font(22, bold=True)
    label_font = _get_font(18, bold=True)
    prompt_font = _get_font(12)
    meta_font = _get_font(11)

    # Header
    best_r = exp["ckpt"].get("best_reward", 0)
    total = exp["ckpt"].get("total_images", 0)
    args = exp["ckpt"].get("args", {})
    reward_type = args.get("reward_type", "delta") if isinstance(args, dict) else "delta"
    title = f"{exp['name']}  |  best_reward={best_r:.4f}  |  {total} imgs  |  reward={reward_type}"
    draw.text((12, 14), title, fill="black", font=title_font)
    # Legend for colored diff words
    legend_x = canvas_w - 270
    draw.text((legend_x, 20), "prompt diff:", fill=(60, 60, 60), font=prompt_font)
    draw.text((legend_x + 78, 20), "source-only / target-only words in red",
              fill=(200, 0, 0), font=prompt_font)

    # Reference row: SOURCE | TARGET
    ref_pad = (canvas_w - 2 * big_thumb) // 3
    y_ref = header_h
    for i, (label, path, tagged) in enumerate([
        ("SOURCE (b=0)", source_path, src_tagged),
        ("TARGET (b=all-ones)", target_path, tgt_tagged),
    ]):
        x_ref = ref_pad + i * (big_thumb + ref_pad)
        if path and os.path.isfile(path):
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((big_thumb, big_thumb))
                x_img = x_ref + (big_thumb - img.width) // 2
                y_img = y_ref + (big_thumb - img.height) // 2
                canvas.paste(img, (x_img, y_img))
            except Exception as e:
                print(f"  failed {path}: {e}")
        else:
            draw.rectangle([x_ref, y_ref, x_ref + big_thumb, y_ref + big_thumb],
                           outline="gray", width=2)
            draw.text((x_ref + 10, y_ref + 10), "(not saved — legacy run)",
                      fill="gray", font=prompt_font)
        # Label
        draw.text((x_ref, y_ref - 22), label, fill="black", font=label_font)
        # Prompt text below — words that differ from the other prompt are shown in red
        _draw_colored_wrapped(
            draw,
            x=x_ref,
            y=y_ref + big_thumb + 8,
            words_tagged=tagged,
            font=prompt_font,
            max_width=big_thumb,
            line_h=15,
        )

    # Top-K grid
    y_grid = header_h + ref_row_h
    draw.text((12, y_grid - 4), "Top-K REINFORCE outputs", fill="black", font=label_font)
    y_grid += 22

    for i, f in enumerate(top_files):
        r, c = divmod(i, cols)
        # Parse reward and bit string from filename
        try:
            parts = f.split("_")
            r_str = [p for p in parts if p.startswith("r")][0][1:]
            b_str = [p for p in parts if p.startswith("b")][-1].split(".")[0][1:]
            label = f"#{i}  r={r_str}\nb={b_str}"
        except Exception:
            label = f"#{i}"

        try:
            img = Image.open(os.path.join(exp_dir, f)).convert("RGB")
            img.thumbnail((thumb, thumb))
            x = c * thumb + (thumb - img.width) // 2
            y = y_grid + r * (thumb + strip_h) + (thumb - img.height) // 2
            canvas.paste(img, (x, y))
            y_text = y_grid + r * (thumb + strip_h) + thumb + 2
            for line_idx, line in enumerate(label.split("\n")):
                draw.text((c * thumb + 6, y_text + line_idx * 14),
                          line, fill="black", font=meta_font)
        except Exception as e:
            print(f"  failed {f}: {e}")

    canvas.save(out_path, quality=95)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    grids_dir = os.path.join(OUT_DIR, "top_images_grids")
    os.makedirs(grids_dir, exist_ok=True)

    print("Loading experiments...")
    experiments = [load_experiment(n) for n in EXPERIMENTS]
    done = [e for e in experiments if e is not None]
    print(f"  {len(done)} / {len(EXPERIMENTS)} experiments have results\n")

    # 1. Summary CSV
    print("Building summary table...")
    summary = build_summary(done)
    csv_path = os.path.join(OUT_DIR, "summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print()

    # Print compact summary to stdout
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    cols_to_show = ["name", "best_reward", "total_images", "final_bg_ssim",
                    "final_fg_clip", "final_entropy", "best_mask"]
    avail = [c for c in cols_to_show if c in summary.columns]
    print(summary[avail].to_string(index=False))
    print()

    # 2. Training curves
    print("Plotting training curves...")
    plot_training_curves(done, os.path.join(OUT_DIR, "training_curves.png"))

    # 3. Learned probs heatmap
    print("Plotting learned probabilities...")
    plot_learned_probs(done, os.path.join(OUT_DIR, "learned_probs.png"))

    # 4. Top-K image grids per experiment
    print("Building top-image grids...")
    for exp in done:
        out_path = os.path.join(grids_dir, f"{exp['name']}_top_grid.jpg")
        build_top_images_grid(exp, out_path)

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
