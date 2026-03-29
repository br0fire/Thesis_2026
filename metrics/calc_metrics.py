"""
Compute metrics for generated images:
  - CLIP score: similarity between each output image and the target prompt.
  - Image similarity: similarity between each output image and the source image (CLIP embeddings).
Usage: pip install transformers && python calc_metrics.py [--images_dir ...] [--output_csv metrics.csv]
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

def _to_tensor(out):
    """Handle both tensor and BaseModelOutputWithPooling from get_*_features."""
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

class ImagePathsDataset(Dataset):
    """Load and preprocess images in workers. Returns (path, preprocessed_dict) or (path, None) on failure."""

    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            img = Image.open(path).convert("RGB")
            out = self.processor(images=[img], return_tensors="pt")
            return (path, {k: v.squeeze(0) for k, v in out.items()})
        except Exception:
            return (path, None)


def _collate_preprocessed(samples):
    """Filter failed loads and stack preprocessed dicts."""
    valid = [s for s in samples if s[1] is not None]
    if not valid:
        return [], None
    paths = [s[0] for s in valid]
    batch_dict = {k: torch.stack([s[1][k] for s in valid]) for k in valid[0][1].keys()}
    return paths, batch_dict


@torch.no_grad()
def compute_batch_metrics(model, batch_dict, text_emb, source_emb, device):
    """One CLIP forward for preprocessed batch; returns (clip_scores, image_sims) as numpy arrays."""
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch_dict.items()}
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        image_emb = _to_tensor(model.get_image_features(**inputs))
    image_emb = F.normalize(image_emb.float(), p=2, dim=-1)
    B = image_emb.shape[0]
    clip_scores = (image_emb * text_emb.expand(B, -1)).sum(dim=-1).cpu().numpy()
    image_sims = (image_emb * source_emb).sum(dim=-1).cpu().numpy()
    return clip_scores, image_sims

def get_image_list(images_dir):
    """List images from directory directly (avoids loading ~1GB pkl just for keys)."""
    return sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_image", default="istanbul-cats-history.jpg")
    p.add_argument("--target_prompt", default="tabby dog walking confidently across a stone pavement.")
    p.add_argument("--images_dir", default="/home/jovyan/shares/SR006.nfs3/svgrozny/generated_samples")
    p.add_argument("--output_csv", default="metrics.csv")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16, help="DataLoader workers for parallel load+preprocess.")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = args.source_image if os.path.isabs(args.source_image) else os.path.join(script_dir, args.source_image)
    images_dir = args.images_dir if os.path.isabs(args.images_dir) else os.path.join(script_dir, args.images_dir)
    if not os.path.isfile(source_path):
        raise FileNotFoundError("Source image not found: " + source_path)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError("Images directory not found: " + images_dir)
    image_paths = get_image_list(images_dir)
    if not image_paths:
        raise RuntimeError("No images found.")
    if args.limit:
        image_paths = image_paths[: args.limit]
    print("Processing", len(image_paths), "images. Device:", device)
    model, processor = load_clip(device)
    # Precompute text and source embeddings once
    source_pil = Image.open(source_path).convert("RGB")
    src_in = processor(images=[source_pil], return_tensors="pt")
    src_in = {k: v.to(device) for k, v in src_in.items()}
    text_in = processor(text=[args.target_prompt], return_tensors="pt")
    text_in = {k: v.to(device) for k, v in text_in.items()}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        source_emb = _to_tensor(model.get_image_features(**src_in))
        text_emb = _to_tensor(model.get_text_features(**text_in))
    source_emb = F.normalize(source_emb.float(), p=2, dim=-1)
    text_emb = F.normalize(text_emb.float(), p=2, dim=-1)
    del source_pil, src_in, text_in

    out_path = args.output_csv if os.path.isabs(args.output_csv) else os.path.join(script_dir, args.output_csv)
    batch_size = args.batch_size
    num_workers = max(0, args.num_workers)
    dataset = ImagePathsDataset(image_paths, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_preprocessed,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    all_clip, all_sim = [], []
    n_skipped = len(image_paths)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "clip_score_target", "image_similarity_source"])
        rows_buffer = []
        for paths_batch, batch_dict in tqdm(dataloader, desc="Metrics", unit="batch"):
            if not paths_batch or batch_dict is None:
                continue
            n_skipped -= len(paths_batch)
            clip_batch, sim_batch = compute_batch_metrics(
                model, batch_dict, text_emb, source_emb, device
            )
            for path, cs, sim in zip(paths_batch, clip_batch.tolist(), sim_batch.tolist()):
                rows_buffer.append([os.path.basename(path), round(cs, 6), round(sim, 6)])
            all_clip.extend(clip_batch.tolist())
            all_sim.extend(sim_batch.tolist())
            if len(rows_buffer) >= 1000:
                w.writerows(rows_buffer)
                rows_buffer = []
        if rows_buffer:
            w.writerows(rows_buffer)
    n_skipped = max(0, n_skipped)
    if n_skipped:
        print("Skipped", n_skipped, "paths (missing or unreadable).")
    print("Saved", out_path)
    if all_clip:
        print("CLIP score (target): mean=%.4f std=%.4f" % (np.mean(all_clip), np.std(all_clip)))
        print("Image similarity (source): mean=%.4f std=%.4f" % (np.mean(all_sim), np.std(all_sim)))

if __name__ == "__main__":
    main()
