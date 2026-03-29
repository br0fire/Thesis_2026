"""
Segment the foreground object in the source image.

Two-stage approach:
  1. CLIPSeg (text-prompted) gives a coarse location of the object → bounding box
  2. SAM (Segment Anything) refines it to pixel-perfect mask at full resolution

This handles thin structures (tails, legs, ears) that CLIPSeg misses due to
its low (~64x64) internal resolution.

Produces a binary mask: 1 = background, 0 = foreground (object to edit).

Usage:
  python segment_source.py [--prompt "cat"] [--image istanbul-cats-history.jpg] [--dilate 15]
  python segment_source.py --method clipseg   # fallback: CLIPSeg only (no SAM needed)
"""
import argparse
import os
import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────
# Method 1: CLIPSeg only (coarse, fast, no SAM)
# ──────────────────────────────────────────────

def compute_mask_clipseg(image_path, prompt, dilate_px=15, threshold=0.5, device=None):
    """Segment using CLIPSeg alone. Coarse — may miss thin parts like tails."""
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (W, H)

    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    upsampled = torch.nn.functional.interpolate(
        logits.unsqueeze(1).float(),
        size=(orig_size[1], orig_size[0]),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    prob = torch.sigmoid(upsampled).cpu().numpy()
    foreground = (prob > threshold).astype(np.uint8)

    if dilate_px > 0:
        from scipy.ndimage import binary_dilation
        struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1))
        foreground = binary_dilation(foreground, structure=struct).astype(np.uint8)

    background_mask = 1 - foreground
    return background_mask, prob


# ──────────────────────────────────────────────
# Method 2: CLIPSeg → SAM (precise, full-res)
# ──────────────────────────────────────────────

def _clipseg_to_bbox(image_path, prompt, threshold=0.3, pad_frac=0.05, device=None):
    """Use CLIPSeg to get a bounding box around the object (for SAM input).

    Uses a lower threshold (0.3) to capture a generous region,
    then adds padding to ensure thin parts near the boundary are included.
    """
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    upsampled = torch.nn.functional.interpolate(
        logits.unsqueeze(1).float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    prob = torch.sigmoid(upsampled).cpu().numpy()
    fg = prob > threshold

    if not fg.any():
        # Fallback: use center of image
        cx, cy = W // 2, H // 2
        return np.array([cx - W // 4, cy - H // 4, cx + W // 4, cy + H // 4]), prob

    ys, xs = np.where(fg)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

    # Add padding
    pad_x = int((x2 - x1) * pad_frac) + 10
    pad_y = int((y2 - y1) * pad_frac) + 10
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W - 1, x2 + pad_x)
    y2 = min(H - 1, y2 + pad_y)

    bbox = np.array([x1, y1, x2, y2])

    del model
    torch.cuda.empty_cache()

    return bbox, prob


def _find_sam_checkpoint():
    """Find SAM checkpoint in common locations."""
    for candidate in [
        os.path.expanduser("~/.cache/sam_vit_b.pth"),
        os.path.expanduser("~/.cache/sam_vit_h.pth"),
        "sam_vit_b.pth",
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "SAM checkpoint not found. Download it:\n"
        "  wget -O ~/.cache/sam_vit_b.pth "
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    )


def compute_mask_sam(image_path, prompt, dilate_px=15, sam_checkpoint=None, device=None):
    """
    Three-stage segmentation: CLIPSeg ∪ SAM.

    Why both?
    - SAM gives pixel-perfect contours (ears, paws) but needs a prompt
      to know *which* object to segment.
    - CLIPSeg knows *what* to look for ("cat") but works at low resolution
      (~64x64) — it produces a blobby mask that covers thin parts (tail)
      that SAM's automatic segmentation may split into separate tiny masks.

    Strategy:
      1. CLIPSeg (low threshold=0.25) → coarse foreground that covers
         even thin/diffuse parts like the tail.
      2. SAM automatic mask generation → find the mask with highest
         overlap with CLIPSeg = the main object body (pixel-perfect).
      3. Union: SAM body ∪ CLIPSeg coarse → combines precise contours
         with full coverage of thin parts.
      4. Dilate for safety margin.
    """
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Stage 1: CLIPSeg → coarse foreground (low threshold to capture thin parts)
    print("  Stage 1: CLIPSeg → coarse foreground mask...")
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    image_np = np.array(image)

    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clipseg_model(**inputs)

    logits = outputs.logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    upsampled = torch.nn.functional.interpolate(
        logits.unsqueeze(1).float(), size=(H, W),
        mode="bilinear", align_corners=False,
    ).squeeze()
    clipseg_prob = torch.sigmoid(upsampled).cpu().numpy()
    clipseg_fg = clipseg_prob > 0.25  # low threshold for generous coverage
    print(f"    CLIPSeg foreground (thr=0.25): {100 * clipseg_fg.mean():.1f}% of image")

    del clipseg_model
    torch.cuda.empty_cache()

    # Stage 2: SAM automatic → pixel-perfect main object mask
    print("  Stage 2: SAM → pixel-perfect segmentation...")
    if sam_checkpoint is None:
        sam_checkpoint = _find_sam_checkpoint()

    model_type = "vit_h" if "vit_h" in sam_checkpoint else \
                 "vit_l" if "vit_l" in sam_checkpoint else "vit_b"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        sam, points_per_side=32,
        pred_iou_thresh=0.86, stability_score_thresh=0.92,
        min_mask_region_area=100,
    )
    auto_masks = mask_generator.generate(image_np)
    print(f"    SAM found {len(auto_masks)} automatic masks")

    # Pick the mask with highest overlap with CLIPSeg = the main object
    for m in auto_masks:
        inter = (m["segmentation"] & clipseg_fg).sum()
        m["overlap"] = inter / (clipseg_fg.sum() + 1e-8)

    auto_masks.sort(key=lambda m: m["overlap"], reverse=True)
    sam_main = auto_masks[0]["segmentation"]
    print(f"    Best SAM mask: overlap={auto_masks[0]['overlap']:.3f}, "
          f"area={100 * sam_main.mean():.1f}%, "
          f"iou_score={auto_masks[0]['predicted_iou']:.3f}")

    del sam, mask_generator
    torch.cuda.empty_cache()

    # Stage 3: Union — SAM (precise body) ∪ CLIPSeg (coarse, covers thin parts)
    foreground = (sam_main | clipseg_fg).astype(np.uint8)
    print(f"    Union: {100 * foreground.mean():.1f}% of image")

    # Dilate for safety margin
    if dilate_px > 0:
        from scipy.ndimage import binary_dilation
        struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1))
        foreground = binary_dilation(foreground, structure=struct).astype(np.uint8)

    background_mask = 1 - foreground
    print(f"    Final background: {100 * background_mask.mean():.1f}%")

    return background_mask, clipseg_prob


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="istanbul-cats-history.jpg")
    p.add_argument("--prompt", default="cat", help="Text prompt for the object to segment")
    p.add_argument("--method", choices=["sam", "clipseg"], default="sam",
                   help="'sam' = CLIPSeg+SAM (precise), 'clipseg' = CLIPSeg only (coarse)")
    p.add_argument("--dilate", type=int, default=15, help="Dilation radius (px) around object")
    p.add_argument("--threshold", type=float, default=0.5, help="CLIPSeg threshold (only for --method clipseg)")
    p.add_argument("--sam_checkpoint", default=None, help="Path to SAM checkpoint (.pth)")
    p.add_argument("--output", default="background_mask.npy")
    p.add_argument("--save_vis", action="store_true", help="Save visualization of the mask")
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = args.image if os.path.isabs(args.image) else os.path.join(script_dir, args.image)

    print(f"Segmenting '{args.prompt}' in {image_path}  (method={args.method})")

    if args.method == "sam":
        bg_mask, prob_map = compute_mask_sam(
            image_path, args.prompt, args.dilate, args.sam_checkpoint)
    else:
        bg_mask, prob_map = compute_mask_clipseg(
            image_path, args.prompt, args.dilate, args.threshold)

    out_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)
    np.save(out_path, bg_mask)
    print(f"Background mask saved to {out_path}")
    print(f"  Shape: {bg_mask.shape}, Background pixels: {bg_mask.sum()} / {bg_mask.size} "
          f"({100 * bg_mask.mean():.1f}%)")

    if args.save_vis:
        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        # CLIPSeg probability heatmap
        prob_img = Image.fromarray((prob_map * 255).astype(np.uint8), mode="L").resize((W, H))

        # Foreground overlay (red)
        red_overlay = np.array(image).copy()
        fg = bg_mask == 0
        red_overlay[fg, 0] = np.clip(red_overlay[fg, 0].astype(int) + 100, 0, 255)
        red_overlay[fg, 1] = (red_overlay[fg, 1] * 0.5).astype(np.uint8)
        red_overlay[fg, 2] = (red_overlay[fg, 2] * 0.5).astype(np.uint8)
        mask_overlay = Image.fromarray(red_overlay)

        # Masked source (foreground → gray)
        masked_src = np.array(image).copy()
        masked_src[fg] = 128
        masked_img = Image.fromarray(masked_src)

        canvas = Image.new("RGB", (W * 4, H))
        canvas.paste(image, (0, 0))
        canvas.paste(prob_img.convert("RGB"), (W, 0))
        canvas.paste(mask_overlay, (W * 2, 0))
        canvas.paste(masked_img, (W * 3, 0))

        vis_path = out_path.replace(".npy", "_vis.jpg")
        canvas.save(vis_path, quality=95)
        print(f"Visualization saved to {vis_path}")


if __name__ == "__main__":
    main()
