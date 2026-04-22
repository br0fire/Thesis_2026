"""Evaluate amortized policy: generate ONE image from predicted greedy mask and score it.

For each experiment in predictions.json, loads source_prompt/target_prompt/seg from its
most recent prompts.txt, builds the reward computer, then for each prediction strategy
(MLP, Ridge, Population mean, Oracle) generates one image and measures reward.

Usage:
  python analysis/eval_amortized.py --experiment <name> --gpu 0
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))
from reinforce_search import DiffusionGenerator, RewardComputer, compute_segmentation

ANALYSIS = os.path.join(PROJECT_ROOT, "analysis/reinforce_analysis")
OUT_DIR = os.path.join(ANALYSIS, "amortized")


def parse_prompts(p):
    out = {}
    with open(p) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def find_prompts_for(exp_name):
    # Priority 1: sweep_<exp>/configs/alpha_high/prompts.txt
    pp = os.path.join(ANALYSIS, f"sweep_{exp_name}", "configs", "alpha_high", "prompts.txt")
    if os.path.isfile(pp):
        return pp
    # Priority 2: new_bgrich/<exp>/prompts.txt
    pp = os.path.join(ANALYSIS, "new_bgrich", exp_name, "prompts.txt")
    if os.path.isfile(pp):
        return pp
    # Priority 3: latest v-series
    for version in sorted(os.listdir(ANALYSIS), reverse=True):
        if not re.match(r"v\d+[a-z]*$", version):
            continue
        pp = os.path.join(ANALYSIS, version, "experiments", f"reinforce_{exp_name}_{version}", "prompts.txt")
        if os.path.isfile(pp):
            return pp
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_bits", type=int, default=14)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--vision_model", default="google/siglip2-so400m-patch14-384")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    preds = json.load(open(os.path.join(OUT_DIR, "predictions.json")))
    if args.experiment not in preds:
        raise SystemExit(f"{args.experiment} not in predictions.json")
    exp_preds = preds[args.experiment]

    pp = find_prompts_for(args.experiment)
    if not pp:
        raise SystemExit(f"No prompts.txt found for {args.experiment}")
    prompts = parse_prompts(pp)
    print(f"Prompts: {pp}")
    print(f"  source: {prompts['source'][:70]}...")
    print(f"  target: {prompts['target'][:70]}...")
    print(f"  seg: {prompts.get('seg', '')}")

    out_exp = os.path.join(OUT_DIR, args.experiment)
    os.makedirs(out_exp, exist_ok=True)

    # Load generator
    generator = DiffusionGenerator(
        device=device, source_prompt=prompts["source"], target_prompt=prompts["target"],
        height=512, width=512, guidance_scale=4.0,
        seed=42, n_bits=args.n_bits, steps=args.n_bits * 2,
    )

    # Source + seg
    source_img = generator.generate(torch.zeros(1, args.n_bits, device=device))
    seg = prompts.get("seg", prompts["source"])
    bg_mask = compute_segmentation(source_img, seg, device)

    reward_computer = RewardComputer(
        device=device, source_image=source_img, bg_mask=bg_mask,
        source_prompt=prompts["source"], target_prompt=prompts["target"],
        img_size=512, vision_model=args.vision_model,
    )

    # Evaluate each strategy
    results = {}
    strategies = {
        "mlp":       np.asarray(exp_preds["probs_mlp"], dtype=np.float32),
        "ridge":     np.asarray(exp_preds["probs_ridge"], dtype=np.float32),
        "popmean":   np.asarray(exp_preds["probs_popmean"], dtype=np.float32),
        "oracle":    np.asarray(exp_preds["probs_true"], dtype=np.float32),
    }

    for name, probs in strategies.items():
        greedy = (probs > 0.5).astype(np.float32)
        mask = torch.from_numpy(greedy).unsqueeze(0).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            img = generator.generate(mask)
            reward, bg, fg = reward_computer.compute_rewards(img, alpha=args.alpha)
        dt = time.perf_counter() - t0

        bstr = "".join(str(int(b)) for b in greedy.tolist())
        print(f"  {name:<10} probs_mae={np.abs(probs - strategies['oracle']).mean():.3f}  "
              f"greedy={bstr}  R={reward.item():.4f}  ({dt:.2f}s)")

        img_pil = Image.fromarray(
            (img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
        img_pil.save(os.path.join(out_exp, f"{name}_r{reward.item():.4f}_b{int(''.join(bstr), 2)}.jpg"),
                     quality=92)
        results[name] = {
            "probs": probs.tolist(),
            "greedy_bits": bstr,
            "reward": float(reward.item()),
            "bg_ssim": float(bg.item()),
            "fg_clip": float(fg.item()),
            "time_s": dt,
        }

    results["oracle_best_reward"] = exp_preds["best_reward_true"]
    with open(os.path.join(out_exp, "eval.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_exp}/eval.json")


if __name__ == "__main__":
    main()
