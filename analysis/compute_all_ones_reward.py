"""Compute all-ones (target-only) reward for each experiment.

Uses existing source_b0.jpg and target_b*.jpg on disk — no FLUX needed.
Reuses SigLIP + SAM 3.1 across experiments (loaded once).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "generation"))
from reinforce_search import compute_segmentation, _gaussian_kernel_2d, ssim_map  # noqa: E402

ANALYSIS = os.path.join(PROJECT_ROOT, "analysis/reinforce_analysis")


def parse_prompts(p):
    out = {}
    with open(p) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def compute_all_ones(src_t, tgt_t, bg_mask_np, src_prompt, tgt_prompt, alpha,
                    vision_model, processor, img_size, device):
    """Compute α·bg_ssim^... · σ(fg_clip·10)^... for a single image pair (no class reload)."""
    H = W = img_size
    bg = torch.from_numpy((bg_mask_np > 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    if bg.shape[-2:] != (H, W):
        bg = F.interpolate(bg, size=(H, W), mode="nearest")
    fg = 1.0 - bg

    # Resize images to img_size
    if src_t.shape[-2:] != (H, W):
        src_t = F.interpolate(src_t, size=(H, W), mode="bilinear", align_corners=False)
    if tgt_t.shape[-2:] != (H, W):
        tgt_t = F.interpolate(tgt_t, size=(H, W), mode="bilinear", align_corners=False)

    # bg SSIM between source and target on bg region
    ksize = 11
    kernel = _gaussian_kernel_2d(ksize, 1.5, device=device)
    ssim = ssim_map(src_t, tgt_t, kernel, pad=ksize // 2)  # (B, 3, H, W)
    ssim_mean = ssim.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    ssim_val = (ssim_mean * bg).sum() / (bg.sum() + 1e-8)
    bg_ssim = float(ssim_val.item())

    # fg_clip: CLIP-delta vs fg-cropped target
    with torch.no_grad():
        # Crop to bounding box of fg
        mask_np = (fg > 0.5).squeeze().cpu().numpy()
        ys, xs = np.where(mask_np)
        if len(xs) == 0:
            return alpha * max(bg_ssim, 0) ** alpha, bg_ssim, 0.0
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        tgt_crop = tgt_t[:, :, y0:y1, x0:x1]
        # Upsample to processor input size
        proc_size = 384  # SigLIP2 SO400M at 384
        tgt_crop_resized = F.interpolate(tgt_crop, size=(proc_size, proc_size),
                                          mode="bilinear", align_corners=False)
        tgt_crop_pil = Image.fromarray(
            (tgt_crop_resized[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))

        # Image and delta text embeddings
        inputs = processor(images=tgt_crop_pil, return_tensors="pt").to(device)
        img_emb = vision_model.get_image_features(**inputs)
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)

        text_inputs = processor(text=[src_prompt, tgt_prompt], padding="max_length", truncation=True,
                                 max_length=64, return_tensors="pt").to(device)
        text_emb = vision_model.get_text_features(**text_inputs)
        text_emb = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-8)
        delta = text_emb[1] - text_emb[0]
        delta = delta / (delta.norm() + 1e-8)
        fg_clip = float((img_emb @ delta.unsqueeze(0).t()).squeeze().item())

    # Reward: geometric mean
    bg_term = max(bg_ssim, 1e-6) ** alpha
    fg_sigmoid = 1.0 / (1.0 + np.exp(-fg_clip * 10.0))
    fg_term = fg_sigmoid ** (1.0 - alpha)
    reward = bg_term * fg_term
    return reward, bg_ssim, fg_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ANALYSIS, "new_bgrich"))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--vision_model", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--img_size", type=int, default=512)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print("Loading SigLIP (once)...", flush=True)
    vision_model = AutoModel.from_pretrained(args.vision_model, torch_dtype=torch.float32).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.vision_model)

    results = {}
    for name in sorted(os.listdir(args.dir)):
        exp = os.path.join(args.dir, name)
        if not os.path.isdir(exp):
            continue
        src_p = os.path.join(exp, "source_b0.jpg")
        tgt_files = sorted(f for f in os.listdir(exp) if f.startswith("target_b") and f.endswith(".jpg"))
        prompts_p = os.path.join(exp, "prompts.txt")
        if not (os.path.isfile(src_p) and tgt_files and os.path.isfile(prompts_p)):
            continue

        prompts = parse_prompts(prompts_p)

        def pil_to_tensor(pil):
            arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        src_t = pil_to_tensor(Image.open(src_p))
        tgt_t = pil_to_tensor(Image.open(os.path.join(exp, tgt_files[0])))

        # Prefer saved bg_mask.npy; otherwise recompute via SAM3
        mask_path = os.path.join(exp, "bg_mask.npy")
        if os.path.isfile(mask_path):
            bg_mask_np = np.load(mask_path).astype(np.float32)
        else:
            bg_mask_np = compute_segmentation(src_t, prompts.get("seg", prompts["source"]), device)
            np.save(mask_path, (bg_mask_np > 0.5).astype(np.uint8))  # cache for next time

        r, bg, fg = compute_all_ones(src_t, tgt_t, bg_mask_np,
                                      prompts["source"], prompts["target"],
                                      args.alpha, vision_model, processor,
                                      args.img_size, device)
        results[name] = {"all_ones_reward": r, "bg_ssim": bg, "fg_clip": fg}
        print(f"  {name:<32} R={r:.4f} bg={bg:.3f} fg={fg:+.3f}")

    out_path = os.path.join(args.dir, "_all_ones_rewards.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
