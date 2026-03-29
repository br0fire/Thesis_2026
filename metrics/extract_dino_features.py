"""
Extract DINOv2 features from generated images. Multi-GPU, streaming.

Produces feature_dictionary.pkl compatible with test.py output.

Usage:
  python extract_dino_features.py --gpus 0,1,2,3,4,5,6,7 --images_dir /path/to/images
  python extract_dino_features.py --gpus 0,1,2,3 --batch_size 512 --output features_catdog.pkl
"""
import argparse
import os
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


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
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            return os.path.basename(path), tensor
        except Exception:
            return os.path.basename(path), None


def _collate(samples):
    valid = [(n, t) for n, t in samples if t is not None]
    if not valid:
        return [], None
    return [s[0] for s in valid], torch.stack([s[1] for s in valid])


def worker_fn(rank, gpu_id, image_paths, args, output_path, progress_queue):
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        # Load DINOv2
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', force_reload=False).to(device)
        dinov2.eval()
        dinov2 = torch.compile(dinov2, mode="max-autotune", fullgraph=False)

        dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        dino_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        dataset = ImageTensorDataset(image_paths, target_size=(args["img_size"], args["img_size"]))
        dataloader = DataLoader(
            dataset,
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=args["num_workers"],
            collate_fn=_collate,
            pin_memory=True,
            prefetch_factor=2 if args["num_workers"] > 0 else None,
            persistent_workers=args["num_workers"] > 0,
        )

        features = {}

        # Warmup
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            _ = dinov2(dummy)
        torch.cuda.synchronize(device)

        for names, imgs in dataloader:
            if not names or imgs is None:
                continue
            imgs_dev = imgs.to(device, non_blocking=True)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                dino_input = F.interpolate(imgs_dev, size=224, mode='bilinear', align_corners=False)
                dino_input = (dino_input - dino_mean) / dino_std
                feats = dinov2(dino_input)

            feats_np = feats.float().cpu().numpy().astype(np.float32)
            for k, name in enumerate(names):
                features[name] = feats_np[k]

            if progress_queue is not None:
                progress_queue.put(len(names))

        # Save part file
        with open(output_path, "wb") as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[GPU {gpu_id}] Done: {len(features)} features → {output_path}")

    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--output", default="feature_dictionary.pkl")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--img_size", type=int, default=512)
    args = p.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    avail = list(range(torch.cuda.device_count()))
    gpu_ids = [g for g in gpu_ids if g in avail]
    if not gpu_ids:
        raise RuntimeError(f"No valid GPUs. Available: {avail}")

    print("Scanning images...")
    t0 = time.time()
    image_paths = sorted(
        os.path.join(args.images_dir, f)
        for f in os.listdir(args.images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"Found {len(image_paths)} images ({time.time()-t0:.1f}s)")
    if not image_paths:
        raise RuntimeError("No images found.")

    # Pre-cache DINOv2
    print("Pre-caching DINOv2 model...")
    torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', force_reload=False)

    shared_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "img_size": args.img_size,
    }

    num_gpus = len(gpu_ids)
    chunks = [image_paths[i::num_gpus] for i in range(num_gpus)]
    tmp_outputs = [f"{args.output}.part{rank}" for rank in range(num_gpus)]

    print(f"Distributing across {num_gpus} GPUs: {gpu_ids}")
    for rank, (gid, chunk) in enumerate(zip(gpu_ids, chunks)):
        print(f"  GPU {gid}: {len(chunk)} images")

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

    import signal
    def _kill(sig=None, frame=None):
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _kill)
    signal.signal(signal.SIGTERM, _kill)

    total = len(image_paths)
    done_count = 0
    try:
        with tqdm(total=total, unit="img", desc="DINOv2 Features") as pbar:
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

    # Merge
    print("Merging feature dictionaries...")
    merged = {}
    for tmp in tmp_outputs:
        if os.path.isfile(tmp):
            with open(tmp, "rb") as f:
                part = pickle.load(f)
            merged.update(part)
            os.remove(tmp)
            print(f"  Loaded {len(part)} from {os.path.basename(tmp)}")

    with open(args.output, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved {args.output}: {len(merged)} features")
    feat_dim = next(iter(merged.values())).shape[0] if merged else 0
    print(f"  Feature dim: {feat_dim}")
    fsize_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  File size: {fsize_mb:.1f} MB")


if __name__ == "__main__":
    main()
