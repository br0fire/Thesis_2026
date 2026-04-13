"""
Scalable segmentation-based metrics for ~1M generated images.

Multi-GPU, batched CLIP (bg+fg in one forward pass), GPU-side masking,
resume support, streaming CSV output.

Metrics:
  Background preservation:
    - bg_clip_similarity: CLIP cosine sim (masked-bg generated vs masked-bg source)
    - bg_ssim: SSIM on background only
  Prompt following:
    - fg_clip_score: CLIP cosine sim (foreground-only generated vs target prompt)

Usage:
  python calc_seg_metrics.py --gpus 4,5,6,7 --batch_size 256
  python calc_seg_metrics.py --gpus 0,1,2,3,4,5,6,7  # after generation done
  python calc_seg_metrics.py --resume  # skip already-computed files
"""
import argparse
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import time


# ────────────────────────────────────────────
# SSIM (pure torch)
# ────────────────────────────────────────────

def _gaussian_kernel_2d(size=11, sigma=1.5, channels=3, device="cpu", dtype=torch.float32):
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k2d = g[:, None] * g[None, :]
    k2d = k2d / k2d.sum()
    return k2d.expand(channels, 1, size, size).contiguous()


def ssim_map(img1, img2, kernel, pad):
    """Compute per-pixel SSIM between (B,C,H,W) tensors in [0,1]. Kernel pre-computed."""
    C = img1.shape[1]
    mu1 = F.conv2d(img1, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=C) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean(dim=1, keepdim=True)


# ────────────────────────────────────────────
# CLIP: GPU-side preprocessing (skip PIL roundtrip)
# ────────────────────────────────────────────

def build_clip_transform(device, dtype=torch.float32):
    """Pre-compute CLIP ViT-B/32 normalization constants on GPU.
    CLIP expects 224x224, normalized with ImageNet stats."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def clip_preprocess_gpu(images_01, clip_mean, clip_std):
    """GPU-side CLIP preprocessing: resize + normalize. Input: (B,3,H,W) in [0,1]."""
    x = F.interpolate(images_01, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - clip_mean) / clip_std


# ────────────────────────────────────────────
# Dataset: raw tensor loading (no CLIP processor)
# ────────────────────────────────────────────

class ImageTensorDataset(Dataset):
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
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
            return os.path.basename(path), tensor
        except Exception:
            return os.path.basename(path), None


def _collate(samples):
    valid = [(n, t) for n, t in samples if t is not None]
    if not valid:
        return [], None
    names = [s[0] for s in valid]
    tensors = torch.stack([s[1] for s in valid])
    return names, tensors


# ────────────────────────────────────────────
# Worker: one GPU, one shard
# ────────────────────────────────────────────

def worker_fn(rank, gpu_id, image_paths, shared_args, output_path, progress_queue):
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        dtype = torch.float32

        args = shared_args
        sz = (args["img_size"], args["img_size"])

        # ── Load mask ──
        bg_mask_full = np.load(args["mask_path"])
        bg_mask_pil = Image.fromarray((bg_mask_full * 255).astype(np.uint8)).resize(sz, Image.NEAREST)
        bg_mask = (np.array(bg_mask_pil) > 127).astype(np.float32)
        bg_mask_dev = torch.from_numpy(bg_mask).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        fg_mask_dev = 1.0 - bg_mask_dev
        bg_pixels = bg_mask_dev.sum().item()

        # Foreground bounding box for cropping (gives CLIP a natural-looking object image)
        fg_ys, fg_xs = np.where(bg_mask < 0.5)
        fg_y1, fg_y2 = int(fg_ys.min()), int(fg_ys.max()) + 1
        fg_x1, fg_x2 = int(fg_xs.min()), int(fg_xs.max()) + 1
        # Pad to square for better CLIP input
        fg_h, fg_w = fg_y2 - fg_y1, fg_x2 - fg_x1
        fg_side = max(fg_h, fg_w)
        fg_cy, fg_cx = (fg_y1 + fg_y2) // 2, (fg_x1 + fg_x2) // 2
        fg_y1 = max(0, fg_cy - fg_side // 2)
        fg_x1 = max(0, fg_cx - fg_side // 2)
        fg_y2 = min(sz[1], fg_y1 + fg_side)
        fg_x2 = min(sz[0], fg_x1 + fg_side)

        # ── Source image ──
        source_pil = Image.open(args["source_path"]).convert("RGB")
        if source_pil.size != sz:
            source_pil = source_pil.resize(sz, Image.LANCZOS)
        source_arr = np.array(source_pil, dtype=np.float32) / 255.0
        source_dev = torch.from_numpy(source_arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)

        # ── SSIM kernel (pre-compute once) ──
        ssim_kernel = _gaussian_kernel_2d(11, 1.5, 3, device=device, dtype=dtype)
        ssim_pad = 5

        # ── CLIP model ──
        from transformers import CLIPModel
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_model.eval()
        clip_vision = clip_model.vision_model
        clip_proj = clip_model.visual_projection
        clip_mean, clip_std = build_clip_transform(device, dtype)

        # Gray fill value as tensor for GPU-side masking
        gray = 128.0 / 255.0

        # ── Precompute source BG CLIP embedding ──
        source_bg = source_dev * bg_mask_dev + gray * fg_mask_dev  # fg → gray
        source_bg_clip = clip_preprocess_gpu(source_bg, clip_mean, clip_std)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            src_bg_emb = clip_proj(clip_vision(pixel_values=source_bg_clip).pooler_output)
        src_bg_emb = F.normalize(src_bg_emb.float(), p=2, dim=-1)

        # ── Precompute delta text embedding (target - source direction) ──
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def _get_text_emb(prompt):
            t = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
            t = {k: v.to(device) for k, v in t.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                emb = clip_model.get_text_features(**t)
                if not isinstance(emb, torch.Tensor):
                    emb = emb.pooler_output
            return emb.float()

        src_text_emb = _get_text_emb(args["source_prompt"])
        tgt_text_emb = _get_text_emb(args["target_prompt"])
        # Delta direction: normalized (target - source) captures "what changed"
        text_emb = F.normalize(tgt_text_emb - src_text_emb, p=2, dim=-1)

        del processor
        torch.cuda.empty_cache()

        # ── DataLoader ──
        nw = args["num_workers"]
        dataset = ImageTensorDataset(image_paths, target_size=sz)
        dataloader = DataLoader(
            dataset,
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=nw,
            collate_fn=_collate,
            pin_memory=True,
            prefetch_factor=2 if nw > 0 else None,
            persistent_workers=nw > 0,
        )

        # ── Process ──
        with open(output_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "bg_clip_similarity", "bg_ssim", "fg_clip_score"])
            buf = []

            for names_batch, img_tensors in dataloader:
                if not names_batch or img_tensors is None:
                    continue

                B = img_tensors.shape[0]
                imgs = img_tensors.to(device, non_blocking=True)  # (B,3,H,W) [0,1]

                # ── SSIM on background ──
                with torch.no_grad():
                    smap = ssim_map(imgs, source_dev.expand(B, -1, -1, -1), ssim_kernel, ssim_pad)
                    bg_ssim = (smap * bg_mask_dev).sum(dim=(1, 2, 3)) / bg_pixels

                # ── GPU-side masking for bg CLIP ──
                imgs_bg = imgs * bg_mask_dev + gray * fg_mask_dev   # (B,3,H,W) bg only

                # ── Crop foreground bbox for fg CLIP (natural image, no gray fill) ──
                imgs_fg_crop = imgs[:, :, fg_y1:fg_y2, fg_x1:fg_x2]  # (B,3,crop_h,crop_w)

                # Preprocess both for CLIP (separate sizes → both resize to 224x224)
                bg_clip_input = clip_preprocess_gpu(imgs_bg, clip_mean, clip_std)
                fg_clip_input = clip_preprocess_gpu(imgs_fg_crop, clip_mean, clip_std)

                # Single forward pass with stacked batch
                combined_clip = torch.cat([bg_clip_input, fg_clip_input], dim=0)  # (2B,3,224,224)

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    pooled = clip_vision(pixel_values=combined_clip).pooler_output
                    embs = clip_proj(pooled)
                embs = F.normalize(embs.float(), p=2, dim=-1)
                bg_embs, fg_embs = embs[:B], embs[B:]

                bg_clip_sim = (bg_embs * src_bg_emb).sum(dim=-1)
                fg_clip_score = (fg_embs * text_emb).sum(dim=-1)

                # ── To CPU + write ──
                bg_ssim_np = bg_ssim.cpu().numpy()
                bg_clip_np = bg_clip_sim.cpu().numpy()
                fg_clip_np = fg_clip_score.cpu().numpy()

                for k, name in enumerate(names_batch):
                    buf.append([
                        name,
                        f"{bg_clip_np[k]:.6f}",
                        f"{bg_ssim_np[k]:.6f}",
                        f"{fg_clip_np[k]:.6f}",
                    ])
                if len(buf) >= 2000:
                    w.writerows(buf)
                    f.flush()
                    buf = []

                if progress_queue is not None:
                    progress_queue.put(B)

            if buf:
                w.writerows(buf)

        print(f"[GPU {gpu_id}] Done: {len(image_paths)} images → {output_path}")

    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()


# ────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────

def get_image_list(images_dir):
    return sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def load_done_set(csv_path):
    """Load already-computed filenames for resume."""
    done = set()
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as f:
            r = csv.reader(f)
            next(r, None)  # skip header
            for row in r:
                if row:
                    done.add(row[0])
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_image", default="istanbul-cats-history.jpg")
    p.add_argument("--mask", default="masks/background_mask.npy")
    p.add_argument("--source_prompt", default="tabby kitten walking confidently across a stone pavement.",
                   help="Source prompt (for delta direction)")
    p.add_argument("--target_prompt", default="tabby dog walking confidently across a stone pavement.",
                   help="Target prompt (for delta direction)")
    p.add_argument("--images_dir", default="/home/jovyan/shares/SR006.nfs3/svgrozny/generated_samples_40step")
    p.add_argument("--output_csv", default="results/seg_metrics.csv")
    p.add_argument("--batch_size", type=int, default=256, help="Per-GPU batch size")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers per GPU")
    p.add_argument("--gpus", default="4,5,6,7", help="Comma-separated GPU IDs")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--resume", action="store_true", help="Skip files already in output CSV")
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    def resolve(path):
        if os.path.isabs(path):
            return path
        # Try script dir first, then project dir
        if os.path.exists(os.path.join(script_dir, path)):
            return os.path.join(script_dir, path)
        return os.path.join(project_dir, path)

    source_path = resolve(args.source_image)
    mask_path = resolve(args.mask)
    images_dir = resolve(args.images_dir)
    out_path = resolve(args.output_csv)

    for fp, name in [(source_path, "Source image"), (mask_path, "Background mask")]:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"{name} not found: {fp}")

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    avail = list(range(torch.cuda.device_count()))
    gpu_ids = [g for g in gpu_ids if g in avail]
    if not gpu_ids:
        raise RuntimeError(f"No valid GPUs. Available: {avail}")

    # ── Collect image list ──
    print("Scanning images...")
    t0 = time.time()
    image_paths = get_image_list(images_dir)
    print(f"Found {len(image_paths)} images ({time.time()-t0:.1f}s)")

    if not image_paths:
        raise RuntimeError("No images found.")

    # ── Resume: filter already-done ──
    if args.resume:
        done = load_done_set(out_path)
        if done:
            before = len(image_paths)
            image_paths = [p for p in image_paths if os.path.basename(p) not in done]
            print(f"Resume: {before - len(image_paths)} already done, {len(image_paths)} remaining")
        if not image_paths:
            print("All images already processed.")
            return

    # ── Shared args dict (pickle-friendly) ──
    shared_args = {
        "source_path": source_path,
        "mask_path": mask_path,
        "source_prompt": args.source_prompt,
        "target_prompt": args.target_prompt,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    # ── Split across GPUs ──
    num_gpus = len(gpu_ids)
    chunks = [image_paths[i::num_gpus] for i in range(num_gpus)]
    tmp_outputs = [f"{out_path}.part{rank}" for rank in range(num_gpus)]

    print(f"Distributing across {num_gpus} GPUs: {gpu_ids}")
    for rank, (gid, chunk) in enumerate(zip(gpu_ids, chunks)):
        print(f"  GPU {gid}: {len(chunk)} images")

    # ── Launch workers ──
    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []
    for rank in range(num_gpus):
        proc = ctx.Process(
            target=worker_fn,
            args=(rank, gpu_ids[rank], chunks[rank], shared_args, tmp_outputs[rank], progress_queue),
        )
        proc.start()
        processes.append(proc)

    # ── Progress bar + cleanup on interrupt ──
    import signal

    def _kill_workers(sig=None, frame=None):
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _kill_workers)
    signal.signal(signal.SIGTERM, _kill_workers)

    total = len(image_paths)
    done_count = 0
    try:
        with tqdm(total=total, unit="img", desc="Seg Metrics") as pbar:
            while done_count < total:
                if not any(p.is_alive() for p in processes) and progress_queue.empty():
                    break
                try:
                    n = progress_queue.get(timeout=2)
                    done_count += n
                    pbar.update(n)
                except:
                    pass
    finally:
        for proc in processes:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()

    # ── Merge part files ──
    print("Merging results...")
    mode = "a" if args.resume else "w"
    with open(out_path, mode, newline="") as f_out:
        w = csv.writer(f_out)
        if mode == "w":
            w.writerow(["filename", "bg_clip_similarity", "bg_ssim", "fg_clip_score"])
        for tmp in tmp_outputs:
            if os.path.isfile(tmp):
                with open(tmp, newline="") as f_in:
                    r = csv.reader(f_in)
                    next(r, None)  # skip header
                    w.writerows(r)
                os.remove(tmp)

    # ── Summary ──
    import pandas as pd
    df = pd.read_csv(out_path)
    print(f"\nSaved {out_path}: {len(df)} rows")
    for col in ["bg_clip_similarity", "bg_ssim", "fg_clip_score"]:
        print(f"  {col}: mean={df[col].mean():.4f}  std={df[col].std():.4f}")


if __name__ == "__main__":
    main()
