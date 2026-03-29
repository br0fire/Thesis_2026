"""
Compute segmentation-based metrics for generated images.

Uses a pre-computed binary mask (from segment_source.py) to separately
evaluate background preservation and foreground prompt following.

Metrics:
  Background preservation (source object masked out on both images):
    - bg_clip_similarity: CLIP cosine sim between masked source & masked generated
    - bg_mse: Mean Squared Error on background pixels only
    - bg_ssim: Structural Similarity Index on background region

  Prompt following (background masked out, foreground only):
    - fg_clip_score: CLIP cosine sim between foreground-only generated image & target prompt

Usage:
  1. First run: python segment_source.py --save_vis  (creates background_mask.npy)
  2. Then run:  python calc_bg_metrics.py [--mask background_mask.npy]
"""
import argparse
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import math


# ────────────────────────────────────────────
# SSIM (pure torch, no extra packages needed)
# ────────────────────────────────────────────

def _gaussian_kernel_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size=11, sigma=1.5, channels=3):
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d[:, None] * k1d[None, :]
    kernel = k2d.expand(channels, 1, size, size).contiguous()
    return kernel


def ssim_map(img1, img2, window_size=11, sigma=1.5):
    """Compute per-pixel SSIM map between two (B,C,H,W) tensors in [0,1]."""
    C = img1.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma, C).to(img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=C) - mu12

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean(dim=1, keepdim=True)  # average over channels → (B,1,H,W)


# ────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────

class MaskedImageDataset(Dataset):
    """Load generated images and apply background mask for metrics."""

    def __init__(self, paths, target_size=(512, 512)):
        self.paths = paths
        self.target_size = target_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            img = Image.open(path).convert("RGB")
            if img.size != self.target_size:
                img = img.resize(self.target_size, Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0,1]
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
            return path, tensor
        except Exception:
            return path, None


def _collate(samples):
    valid = [(p, t) for p, t in samples if t is not None]
    if not valid:
        return [], None
    paths = [s[0] for s in valid]
    tensors = torch.stack([s[1] for s in valid])
    return paths, tensors


# ────────────────────────────────────────────
# CLIP on masked images
# ────────────────────────────────────────────

def _to_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        x = out.last_hidden_state
        return x[:, 0] if x.dim() == 3 else x
    return out


def load_clip(device, model_name="openai/clip-vit-base-patch32"):
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained(model_name).to(device)
    model = torch.compile(model)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def apply_bg_mask_to_pil(pil_img, bg_mask, fill_value=128):
    """Fill foreground (bg_mask==0) with gray → keeps background only."""
    arr = np.array(pil_img)
    arr[bg_mask == 0] = fill_value
    return Image.fromarray(arr)


def apply_fg_mask_to_pil(pil_img, bg_mask, fill_value=128):
    """Fill background (bg_mask==1) with gray → keeps foreground only."""
    arr = np.array(pil_img)
    arr[bg_mask == 1] = fill_value
    return Image.fromarray(arr)


# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────

def get_image_list(images_dir):
    return sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_image", default="istanbul-cats-history.jpg")
    p.add_argument("--mask", default="background_mask.npy", help="Background mask from segment_source.py")
    p.add_argument("--target_prompt", default="tabby dog walking confidently across a stone pavement.")
    p.add_argument("--images_dir", default="/home/jovyan/shares/SR006.nfs3/svgrozny/generated_samples")
    p.add_argument("--output_csv", default="bg_metrics.csv")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--img_size", type=int, default=512)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve paths
    source_path = args.source_image if os.path.isabs(args.source_image) else os.path.join(script_dir, args.source_image)
    mask_path = args.mask if os.path.isabs(args.mask) else os.path.join(script_dir, args.mask)
    images_dir = args.images_dir if os.path.isabs(args.images_dir) else os.path.join(script_dir, args.images_dir)
    out_path = args.output_csv if os.path.isabs(args.output_csv) else os.path.join(script_dir, args.output_csv)

    for fp, name in [(source_path, "Source image"), (mask_path, "Background mask")]:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"{name} not found: {fp}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError("Images directory not found: " + images_dir)

    # ── Load mask ──
    bg_mask_full = np.load(mask_path)  # (H_orig, W_orig), 1=bg, 0=fg
    # Resize mask to match generation resolution
    sz = (args.img_size, args.img_size)
    bg_mask_pil = Image.fromarray((bg_mask_full * 255).astype(np.uint8)).resize(sz, Image.NEAREST)
    bg_mask = (np.array(bg_mask_pil) > 127).astype(np.float32)  # (H, W)
    bg_mask_torch = torch.from_numpy(bg_mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    bg_pixels = bg_mask.sum()
    print(f"Background mask: {bg_mask.shape}, bg pixels: {int(bg_pixels)} / {bg_mask.size} "
          f"({100 * bg_mask.mean():.1f}%)")

    # ── Load source image tensor ──
    source_pil = Image.open(source_path).convert("RGB")
    if source_pil.size != sz:
        source_pil = source_pil.resize(sz, Image.LANCZOS)
    source_arr = np.array(source_pil, dtype=np.float32) / 255.0
    source_tensor = torch.from_numpy(source_arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    # ── Create masked source for CLIP (background only — foreground grayed out) ──
    masked_source_pil = apply_bg_mask_to_pil(source_pil, bg_mask.astype(np.uint8))

    # ── Load CLIP ──
    print("Loading CLIP...")
    clip_model, clip_processor = load_clip(device)

    # Precompute masked source CLIP embedding
    src_in = clip_processor(images=[masked_source_pil], return_tensors="pt")
    src_in = {k: v.to(device) for k, v in src_in.items()}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        masked_source_emb = _to_tensor(clip_model.get_image_features(**src_in))
    masked_source_emb = F.normalize(masked_source_emb.float(), p=2, dim=-1)

    # Also precompute text embedding for target CLIP score
    text_in = clip_processor(text=[args.target_prompt], return_tensors="pt")
    text_in = {k: v.to(device) for k, v in text_in.items()}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        text_emb = _to_tensor(clip_model.get_text_features(**text_in))
    text_emb = F.normalize(text_emb.float(), p=2, dim=-1)

    del src_in, text_in

    # ── Prepare data ──
    image_paths = get_image_list(images_dir)
    if not image_paths:
        raise RuntimeError("No images found.")
    if args.limit:
        image_paths = image_paths[:args.limit]
    print(f"Processing {len(image_paths)} images on {device}")

    dataset = MaskedImageDataset(image_paths, target_size=sz)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # Move tensors to device
    source_tensor_dev = source_tensor.to(device)
    bg_mask_dev = bg_mask_torch.to(device)

    # ── Process batches ──
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "bg_clip_similarity", "bg_mse", "bg_ssim", "fg_clip_score"])
        rows_buffer = []

        for paths_batch, img_tensors in tqdm(dataloader, desc="BG Metrics", unit="batch"):
            if not paths_batch or img_tensors is None:
                continue

            B = img_tensors.shape[0]
            imgs_dev = img_tensors.to(device, non_blocking=True)  # (B, 3, H, W) [0,1]

            # ── bg_mse: MSE on background pixels only ──
            diff_sq = (imgs_dev - source_tensor_dev) ** 2  # (B, 3, H, W)
            # Mask: multiply by bg_mask, sum, divide by bg pixel count
            masked_diff = diff_sq * bg_mask_dev  # zero out foreground
            bg_mse = masked_diff.sum(dim=(1, 2, 3)) / (bg_pixels * 3)  # per-image scalar
            bg_mse_np = bg_mse.cpu().numpy()

            # ── bg_ssim: SSIM on background region ──
            with torch.no_grad():
                smap = ssim_map(imgs_dev, source_tensor_dev.expand(B, -1, -1, -1))  # (B,1,H,W)
                # Average SSIM only over background
                masked_ssim = (smap * bg_mask_dev).sum(dim=(1, 2, 3)) / bg_pixels
            bg_ssim_np = masked_ssim.cpu().numpy()

            # ── Apply masks on CPU (for CLIP processor) ──
            imgs_np = (img_tensors.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            bg_mask_uint8 = bg_mask.astype(np.uint8)
            bg_masked_pils = []  # foreground grayed out → background only
            fg_masked_pils = []  # background grayed out → foreground only
            for k in range(B):
                arr_bg = imgs_np[k].copy()
                arr_bg[bg_mask_uint8 == 0] = 128  # gray out foreground
                bg_masked_pils.append(Image.fromarray(arr_bg))

                arr_fg = imgs_np[k].copy()
                arr_fg[bg_mask_uint8 == 1] = 128  # gray out background
                fg_masked_pils.append(Image.fromarray(arr_fg))

            # ── bg_clip_similarity: masked-bg generated vs masked-bg source ──
            clip_inputs = clip_processor(images=bg_masked_pils, return_tensors="pt")
            clip_inputs = {k: v.to(device, non_blocking=True) for k, v in clip_inputs.items()}
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                  enabled=(device.type == "cuda")):
                bg_img_emb = _to_tensor(clip_model.get_image_features(**clip_inputs))
            bg_img_emb = F.normalize(bg_img_emb.float(), p=2, dim=-1)
            bg_clip_sim = (bg_img_emb * masked_source_emb).sum(dim=-1).cpu().numpy()

            # ── fg_clip_score: foreground-only generated vs target prompt ──
            clip_inputs_fg = clip_processor(images=fg_masked_pils, return_tensors="pt")
            clip_inputs_fg = {k: v.to(device, non_blocking=True) for k, v in clip_inputs_fg.items()}
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                  enabled=(device.type == "cuda")):
                fg_img_emb = _to_tensor(clip_model.get_image_features(**clip_inputs_fg))
            fg_img_emb = F.normalize(fg_img_emb.float(), p=2, dim=-1)
            fg_clip_score = (fg_img_emb * text_emb).sum(dim=-1).cpu().numpy()

            # ── Write results ──
            for k, path in enumerate(paths_batch):
                rows_buffer.append([
                    os.path.basename(path),
                    round(float(bg_clip_sim[k]), 6),
                    round(float(bg_mse_np[k]), 6),
                    round(float(bg_ssim_np[k]), 6),
                    round(float(fg_clip_score[k]), 6),
                ])
            if len(rows_buffer) >= 1000:
                w.writerows(rows_buffer)
                rows_buffer = []

        if rows_buffer:
            w.writerows(rows_buffer)

    print(f"\nSaved {out_path}")
    # Summary stats
    import pandas as pd
    df = pd.read_csv(out_path)
    for col in ["bg_clip_similarity", "bg_mse", "bg_ssim", "fg_clip_score"]:
        print(f"  {col}: mean={df[col].mean():.4f}  std={df[col].std():.4f}")


if __name__ == "__main__":
    main()
