"""
Ablation: does the delta-vs-relative fg reward still disagree with SigLIP 2?

For each v2 experiment (SigLIP2 trained), we take:
  - source image
  - target image
  - top-K REINFORCE outputs

For every image we compute the foreground-cropped SigLIP 2 embedding, then compute
both fg_clip variants (delta and relative) and check whether they agree.

If SigLIP 2 makes the two formulas numerically similar, you can safely use either.
If they still disagree, the sigmoid-trained embeddings didn't fix the geometric issue.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

NFS3 = "/home/jovyan/shares/SR006.nfs3/svgrozny"
PROJECT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
sys.path.insert(0, os.path.join(PROJECT, "generation"))

# Reuse the preprocessing helpers from reinforce_search.py
from reinforce_search import (
    vision_preprocess_gpu,
    build_vision_transform,
    _detect_model_family,
    _model_input_size,
)

VISION_MODEL = "google/siglip2-so400m-patch14-384"
DEVICE = torch.device("cuda:0")

EXPERIMENTS = [
    "catdog_v2", "car_taxi_v2", "sunflower_lavender_v2", "chair_throne_v2",
    "penguin_flamingo_v2", "cake_books_v2", "lighthouse_castle_v2", "violin_guitar_v2",
    "horse_v2", "room_v2", "snow_volcano_v2", "butterfly_hummingbird_v2", "sail_pirate_v2",
    "bgrich_teapot_globe", "bgrich_candle_crystal", "bgrich_typewriter_laptop",
]


def load_image_tensor(path, size=512):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def load_prompts_and_mask(exp_dir):
    """Parse prompts.txt and find the background mask."""
    prompts_path = os.path.join(exp_dir, "prompts.txt")
    if not os.path.isfile(prompts_path):
        return None, None, None
    src, tgt = None, None
    with open(prompts_path) as f:
        for line in f:
            if line.startswith("source:"):
                src = line.split(":", 1)[1].strip()
            elif line.startswith("target:"):
                tgt = line.split(":", 1)[1].strip()
    # Mask is separate — not stored in exp_dir. Skip fg crop and use full image as approximation.
    return src, tgt, None


def main():
    from transformers import AutoModel, AutoProcessor
    print(f"Loading {VISION_MODEL}...")
    model = AutoModel.from_pretrained(VISION_MODEL, torch_dtype=torch.float32).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(VISION_MODEL)

    family = _detect_model_family(VISION_MODEL)
    vis_mean, vis_std = build_vision_transform(DEVICE, model_family=family)
    vis_size = _model_input_size(VISION_MODEL)

    def embed_text(prompt):
        t = processor(text=[prompt], return_tensors="pt", padding="max_length", truncation=True)
        t = {k: v.to(DEVICE) for k, v in t.items()}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            emb = model.get_text_features(**t)
        return emb.float()

    def embed_images(paths):
        imgs = torch.cat([load_image_tensor(p) for p in paths], dim=0).to(DEVICE)
        pp = vision_preprocess_gpu(imgs, vis_mean, vis_std, vis_size)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            emb = model.get_image_features(pixel_values=pp)
        return F.normalize(emb.float(), p=2, dim=-1)

    results = []
    for name in EXPERIMENTS:
        exp_dir = os.path.join(NFS3, f"reinforce_{name}")
        if not os.path.isdir(exp_dir):
            continue
        src_prompt, tgt_prompt, _ = load_prompts_and_mask(exp_dir)
        if src_prompt is None:
            continue

        # Collect image paths: source, target, top-K
        img_paths = []
        labels = []
        src_img = os.path.join(exp_dir, "source_b0.jpg")
        if os.path.isfile(src_img):
            img_paths.append(src_img); labels.append("source")
        # target image (name varies by n_bits)
        for fn in sorted(os.listdir(exp_dir)):
            if fn.startswith("target_b") and fn.endswith(".jpg"):
                img_paths.append(os.path.join(exp_dir, fn)); labels.append("target")
                break
        # top-K
        top_files = sorted([f for f in os.listdir(exp_dir)
                           if f.startswith("reinforce_top") and f.endswith(".jpg")])
        for f in top_files[:8]:
            img_paths.append(os.path.join(exp_dir, f))
            labels.append(f.split("_")[1])  # top0..top10

        if len(img_paths) < 3:
            continue

        # Text embeddings
        src_text = embed_text(src_prompt)
        tgt_text = embed_text(tgt_prompt)
        src_norm = F.normalize(src_text, p=2, dim=-1)
        tgt_norm = F.normalize(tgt_text, p=2, dim=-1)
        delta_dir = F.normalize(tgt_text - src_text, p=2, dim=-1)

        # Image embeddings (full image — no fg crop, since we don't have the mask cached)
        img_embs = embed_images(img_paths)  # (N, D)

        # Compute both metrics for each image
        delta_scores = (img_embs * delta_dir).sum(dim=-1).cpu().numpy()
        rel_scores = ((img_embs * tgt_norm).sum(dim=-1) - (img_embs * src_norm).sum(dim=-1)).cpu().numpy()

        # Pearson correlation
        corr = np.corrcoef(delta_scores, rel_scores)[0, 1] if len(delta_scores) > 1 else np.nan
        # Sign agreement
        sign_agree = ((delta_scores > 0) == (rel_scores > 0)).mean()
        # Ranking agreement (Spearman)
        from scipy.stats import spearmanr
        sp, _ = spearmanr(delta_scores, rel_scores)

        results.append({
            "name": name,
            "n_images": len(img_paths),
            "delta_min": delta_scores.min(),
            "delta_max": delta_scores.max(),
            "delta_range": delta_scores.max() - delta_scores.min(),
            "relative_min": rel_scores.min(),
            "relative_max": rel_scores.max(),
            "relative_range": rel_scores.max() - rel_scores.min(),
            "pearson": corr,
            "spearman": sp,
            "sign_agreement": sign_agree,
            "delta_sign_source": np.sign(delta_scores[0]),  # source image should have negative
            "delta_sign_target": np.sign(delta_scores[1]) if len(delta_scores) > 1 else np.nan,
            "relative_sign_source": np.sign(rel_scores[0]),
            "relative_sign_target": np.sign(rel_scores[1]) if len(rel_scores) > 1 else np.nan,
        })

        print(f"\n{name}:")
        print(f"  n_images={len(img_paths)}")
        print(f"  delta:    range=[{delta_scores.min():+.4f}, {delta_scores.max():+.4f}]  source={delta_scores[0]:+.4f}  target={delta_scores[1]:+.4f}")
        print(f"  relative: range=[{rel_scores.min():+.4f}, {rel_scores.max():+.4f}]  source={rel_scores[0]:+.4f}  target={rel_scores[1]:+.4f}")
        print(f"  pearson={corr:.3f}  spearman={sp:.3f}  sign_agree={sign_agree:.2%}")

    # Summary
    import pandas as pd
    df = pd.DataFrame(results)
    out_csv = os.path.join(PROJECT, "analysis/reinforce_analysis/delta_vs_relative_siglip2.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n\nSaved: {out_csv}")
    print()
    print("=" * 100)
    print("SUMMARY — does SigLIP 2 make delta and relative agree?")
    print("=" * 100)
    print(f"{'experiment':<30s} {'pearson':>9s} {'spearman':>10s} {'sign_agree':>12s} {'delta_src':>11s} {'rel_src':>10s}")
    for r in results:
        print(f"{r['name']:<30s} {r['pearson']:>9.3f} {r['spearman']:>10.3f} {r['sign_agreement']:>12.1%} "
              f"{r['delta_sign_source']:>+11.0f} {r['relative_sign_source']:>+10.0f}")
    print()
    print(f"Mean pearson: {df['pearson'].mean():.3f}")
    print(f"Mean spearman: {df['spearman'].mean():.3f}")
    print(f"Mean sign agreement: {df['sign_agreement'].mean():.1%}")
    print()
    print("Interpretation:")
    print("  pearson > 0.95 → the two rewards are nearly linearly equivalent (same gradient direction)")
    print("  pearson 0.7-0.95 → correlated but scale differs")
    print("  pearson < 0.7 → the two rewards disagree substantially")
    print("  delta_sign_source < 0 AND relative_sign_source < 0 → both correctly rate source as source-like")


if __name__ == "__main__":
    main()
