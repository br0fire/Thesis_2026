"""Build a visual grid for a parallel sweep experiment.

For each config: best-ever image, greedy mask (prob>0.5), 3 policy samples.
Rows = configs, Columns = [best, greedy, sample1, sample2, sample3].
Top row shows source (all-zeros) and target (all-ones) for reference.

Usage:
  python analysis/sweep_grid.py --sweep_dir analysis/reinforce_analysis/sweep_X --gpu 0
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.distributions import Bernoulli

import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PROJECT_ROOT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))
from reinforce_search import DiffusionGenerator, RewardComputer, compute_segmentation

CONFIG_ORDER = [
    "default", "alpha_low", "alpha_high", "lr_low",
    "lr_high", "no_entropy", "high_entropy", "combined",
]

CELL_SIZE = 256
LABEL_H = 40
COLS = ["best", "greedy", "sample 1", "sample 2", "sample 3"]


def parse_prompts(sweep_dir):
    for name in CONFIG_ORDER:
        p = os.path.join(sweep_dir, "configs", name, "prompts.txt")
        if not os.path.isfile(p):
            continue
        out = {}
        with open(p) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    out[k.strip()] = v.strip()
        return out
    raise FileNotFoundError("No prompts.txt found")


def tensor_to_pil(t):
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def label_image(pil_img, text, size=CELL_SIZE):
    img = pil_img.resize((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size + LABEL_H), (255, 255, 255))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    draw.text((4, size + 2), text, fill=(0, 0, 0), font=font)
    return canvas


def mask_to_int(mask):
    bits = mask.int().tolist()
    return sum(b << (len(bits) - 1 - i) for i, b in enumerate(bits))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_bits", type=int, default=14)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vision_model", default="google/siglip2-so400m-patch14-384")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    n_bits = args.n_bits
    steps = n_bits * 2

    prompts = parse_prompts(args.sweep_dir)
    src_prompt = prompts["source"]
    tgt_prompt = prompts["target"]
    seg_prompt = prompts.get("seg", src_prompt)

    print("Loading generator...", flush=True)
    generator = DiffusionGenerator(
        device=device, source_prompt=src_prompt, target_prompt=tgt_prompt,
        height=args.height, width=args.width, guidance_scale=args.guidance_scale,
        seed=args.seed, n_bits=n_bits, steps=steps,
    )

    print("Generating source + target...", flush=True)
    with torch.no_grad():
        source_img = generator.generate(torch.zeros(1, n_bits, device=device))
        target_img = generator.generate(torch.ones(1, n_bits, device=device))

    bg_mask = compute_segmentation(source_img, seg_prompt, device)

    print("Loading reward computer...", flush=True)
    reward_computer = RewardComputer(
        device=device, source_image=source_img, bg_mask=bg_mask,
        source_prompt=src_prompt, target_prompt=tgt_prompt,
        img_size=args.height, vision_model=args.vision_model,
    )

    source_pil = label_image(tensor_to_pil(source_img[0]), "source (all-zeros)")
    target_pil = label_image(tensor_to_pil(target_img[0]), "target (all-ones)")

    n_cols = len(COLS)
    cell_w = CELL_SIZE
    cell_h = CELL_SIZE + LABEL_H

    # Header row: source, target, then empty
    header_h = cell_h + 10
    grid_w = n_cols * cell_w + (n_cols - 1) * 4
    grid_h = header_h + len(CONFIG_ORDER) * (cell_h + 4)
    grid = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
    grid.paste(source_pil, (0, 0))
    grid.paste(target_pil, (cell_w + 4, 0))

    # Column headers
    draw = ImageDraw.Draw(grid)
    try:
        hfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        hfont = ImageFont.load_default()

    configs_dir = os.path.join(args.sweep_dir, "configs")
    for row_idx, cfg_name in enumerate(CONFIG_ORDER):
        cfg_dir = os.path.join(configs_dir, cfg_name)
        ckpt_path = os.path.join(cfg_dir, "reinforce_result.pt")
        if not os.path.isfile(ckpt_path):
            print(f"  {cfg_name}: no checkpoint, skipping", flush=True)
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        probs = ckpt["probs"]
        if torch.is_tensor(probs):
            probs = probs.float()
        else:
            probs = torch.tensor(probs, dtype=torch.float32)

        print(f"  {cfg_name}: generating images...", flush=True)
        y = header_h + row_idx * (cell_h + 4)

        # Col 0: best-ever (load from disk)
        best_files = sorted(
            [f for f in os.listdir(cfg_dir) if f.startswith("reinforce_top0_")],
        )
        if best_files:
            best_pil = Image.open(os.path.join(cfg_dir, best_files[0])).convert("RGB")
            best_r = ckpt.get("best_reward", 0)
            best_mask_val = ckpt.get("best_mask")
            if best_mask_val is not None:
                if torch.is_tensor(best_mask_val):
                    best_mask_val = best_mask_val.int().tolist()
                bstr = "".join(str(int(b)) for b in best_mask_val)
            else:
                bstr = "?"
            cell = label_image(best_pil, f"{cfg_name} best R={best_r:.4f} {bstr}")
            grid.paste(cell, (0, y))

        # Col 1: greedy (prob > 0.5)
        greedy_mask = (probs > 0.5).float().unsqueeze(0).to(device)
        with torch.no_grad():
            greedy_img = generator.generate(greedy_mask)
            greedy_r, _, _ = reward_computer.compute_rewards(greedy_img, alpha=0.5)
        gint = mask_to_int(greedy_mask[0])
        gbits = "".join(str(int(b)) for b in greedy_mask[0].int().tolist())
        cell = label_image(tensor_to_pil(greedy_img[0]),
                           f"{cfg_name} greedy R={greedy_r.item():.4f} {gbits}")
        grid.paste(cell, (1 * (cell_w + 4), y))

        # Cols 2-4: 3 policy samples
        dist = Bernoulli(probs.to(device))
        sample_masks = dist.sample((3,))
        with torch.no_grad():
            sample_imgs = generator.generate(sample_masks)
            sample_rs, _, _ = reward_computer.compute_rewards(sample_imgs, alpha=0.5)
        for si in range(3):
            sbits = "".join(str(int(b)) for b in sample_masks[si].int().tolist())
            cell = label_image(tensor_to_pil(sample_imgs[si]),
                               f"{cfg_name} s{si+1} R={sample_rs[si].item():.4f} {sbits}")
            grid.paste(cell, ((2 + si) * (cell_w + 4), y))

    out_path = os.path.join(args.sweep_dir, "config_grid.png")
    grid.save(out_path, quality=95)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
