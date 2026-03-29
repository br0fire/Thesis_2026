#!/usr/bin/env python3
"""
Cluster visualizations from DINOv2 embeddings + segmentation metrics.

Outputs:
  - UMAP/PCA 2D map colored by clusters
  - UMAP map colored by seg metrics (bg_ssim, fg_clip_score, bg_clip_sim, combined)
  - Grid: top 20 clusters × 5 images each
  - Grid: one image per cluster
  - Grids: top images by each seg metric
"""

import os
import math
import pickle
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


# -------------------- config --------------------
EMB_PATH = "feature_dictionary.pkl"
IMAGES_ROOT = "/home/jovyan/shares/SR006.nfs3/svgrozny/generated_samples_40step"
OUT_DIR = "visualizations"
COORDS_PATH = "coords.npy"
LABELS_PATH = "labels.npy"

PCA_DIM = 64
HDBSCAN_MIN_CLUSTER_SIZE = 5000
HDBSCAN_MIN_SAMPLES = 100
N_BITS = 20
BIT0_IS_STEP0 = False
THUMB = 160
TOP_N_CLUSTERS = 20
IMAGES_PER_ROW_TOP = 5
TOP_N_BY_METRIC = 20
COLS_METRIC_GRID = 5
METADATA_STRIP_H = 48

# Metric display names (short labels for image annotations)
METRIC_LABELS = {
    "bg_clip_similarity": "bg_clip",
    "bg_ssim": "bg_ssim",
    "fg_clip_score": "fg_clip",
}


# -------------------- helpers --------------------

def extract_b_value(filename: str):
    try:
        base = os.path.basename(filename)
        return int(base.rsplit("_b", 1)[1].split(".")[0])
    except Exception:
        return None


def b_to_bits_20(b: int) -> np.ndarray:
    raw = ((b >> np.arange(N_BITS)) & 1).astype(np.uint8)
    return raw if BIT0_IS_STEP0 else raw[::-1]


def quadrant_ones(b: int):
    bits = b_to_bits_20(b)
    return (int(bits[0:5].sum()), int(bits[5:10].sum()),
            int(bits[10:15].sum()), int(bits[15:20].sum()))


# -------------------- data loading --------------------

def load_embeddings_and_keys(emb_path: str):
    with open(emb_path, "rb") as f:
        emb = pickle.load(f)
    keys = list(emb.keys())
    dim = next(iter(emb.values())).shape[0]
    X = np.empty((len(keys), dim), dtype=np.float32)
    for i, k in enumerate(keys):
        X[i] = emb[k]
    return X, keys


def compute_coords(X: np.ndarray, use_umap: bool = True, gpu_id: int = 0):
    X = normalize(X)
    X = PCA(min(PCA_DIM, X.shape[1], X.shape[0])).fit_transform(X)
    if use_umap:
        # Try GPU UMAP (cuML) first — 10-50x faster than CPU on 1M points
        try:
            import cupy as cp
            from cuml.manifold import UMAP as cuUMAP
            print(f"  Using GPU UMAP (cuML) on cuda:{gpu_id}...")
            with cp.cuda.Device(gpu_id):
                reducer = cuUMAP(n_neighbors=30, min_dist=0.05, metric="cosine",
                                 n_epochs=100, output_type="numpy")
                coords = reducer.fit_transform(X.astype(np.float32))
            return coords
        except ImportError:
            print("  cuML not available, falling back to CPU UMAP...")
        except Exception as e:
            print(f"  cuML UMAP failed ({e}), falling back to CPU UMAP...")
        # Fallback: CPU UMAP (fit on subsample, transform all)
        try:
            import umap
            print("  Using CPU UMAP (fit on 100K subsample)...")
            idx = np.random.choice(X.shape[0], size=min(100_000, X.shape[0]), replace=False)
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, metric="cosine", n_epochs=100)
            reducer.fit(X[idx])
            return reducer.transform(X)
        except Exception:
            pass
    return X[:, :2]


def _assign_noise_to_nearest(coords, labels):
    """Assign noise points (label==-1) to their nearest cluster centroid."""
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels
    labels = labels.copy()
    centroids = {}
    for lab in set(labels.tolist()):
        if lab == -1:
            continue
        centroids[lab] = coords[labels == lab].mean(axis=0)
    centroid_labs = list(centroids.keys())
    centroid_arr = np.array([centroids[l] for l in centroid_labs])
    noise_coords = coords[noise_mask]
    dists = np.linalg.norm(noise_coords[:, None, :] - centroid_arr[None, :, :], axis=2)
    nearest = dists.argmin(axis=1)
    labels[noise_mask] = np.array([centroid_labs[i] for i in nearest])
    return labels


def _compute_labels(coords, gpu_id=0):
    """Cluster with GPU HDBSCAN (cuML), fallback to sklearn. Noise assigned to nearest cluster."""
    labels = None
    try:
        import cupy as cp
        from cuml.cluster import HDBSCAN
        print(f"  Using GPU HDBSCAN (cuML) on cuda:{gpu_id}...")
        with cp.cuda.Device(gpu_id):
            clusterer = HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                output_type="numpy",
            )
            labels = clusterer.fit_predict(coords.astype(np.float32))
    except ImportError:
        print("  cuML not available, falling back to sklearn HDBSCAN...")
    except Exception as e:
        print(f"  cuML HDBSCAN failed ({e}), falling back to sklearn...")

    if labels is None:
        from sklearn.cluster import HDBSCAN as skHDBSCAN
        labels = skHDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
        ).fit_predict(coords)

    n_noise = (labels == -1).sum()
    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    print(f"  {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(labels):.1f}%)")
    if n_noise > 0:
        labels = _assign_noise_to_nearest(coords, labels)
        print(f"  Noise assigned to nearest clusters")
    return labels


def load_seg_metrics_csv(path: str):
    """Load seg_metrics CSV → dict[filename → {bg_clip_similarity, bg_ssim, fg_clip_score}]."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = row.get("filename", "").strip()
            try:
                d = {}
                for col in ("bg_clip_similarity", "bg_ssim", "fg_clip_score"):
                    if col in row:
                        d[col] = float(row[col])
                if d:
                    out[fn] = d
            except (KeyError, ValueError):
                pass
    return out


def get_seg_metrics_arrays(keys, metrics_dict):
    """Returns arrays (bg_clip_sim, bg_ssim, fg_clip_score, combined) of length len(keys)."""
    n = len(keys)
    bg_clip_sim = np.full(n, np.nan)
    bg_ssim = np.full(n, np.nan)
    fg_clip_score = np.full(n, np.nan)
    for i, fn in enumerate(keys):
        if fn in metrics_dict:
            m = metrics_dict[fn]
            bg_clip_sim[i] = m.get("bg_clip_similarity", np.nan)
            bg_ssim[i] = m.get("bg_ssim", np.nan)
            fg_clip_score[i] = m.get("fg_clip_score", np.nan)
    # Combined score: geometric mean of bg_ssim and fg_clip_score (delta embedding)
    # fg_clip_score is bipolar: negative=source-like, positive=target-like
    # Clamp to [0, 1] so negative (failed edits) score 0
    valid = ~np.isnan(bg_ssim) & ~np.isnan(fg_clip_score)
    combined = np.full(n, np.nan)
    combined[valid] = np.sqrt(np.clip(bg_ssim[valid], 0, 1) * np.clip(fg_clip_score[valid], 0, 1))

    return bg_clip_sim, bg_ssim, fg_clip_score, combined


# -------------------- clustering helpers --------------------

def get_cluster_indices_by_size(labels, top_n):
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = np.argsort(-counts)[:top_n]
    return [(int(unique[i]), int(counts[i])) for i in order]


def get_one_image_per_cluster(coords, cluster_indices):
    result = []
    for lab in sorted(cluster_indices.keys()):
        idx = cluster_indices[lab]
        center = coords[idx].mean(axis=0)
        order = np.argsort(((coords[idx] - center) ** 2).sum(axis=1))
        result.append((lab, idx[order[0]]))
    return result


def get_top_cluster_image_paths(keys, labels, coords, images_root, top_n, per_cluster, cluster_indices):
    cluster_sizes = get_cluster_indices_by_size(labels, top_n)
    paths_per_cluster = []
    for lab, _ in cluster_sizes:
        idx = cluster_indices[lab]
        center = coords[idx].mean(axis=0)
        order = np.argsort(((coords[idx] - center) ** 2).sum(axis=1))
        taken = idx[order][:per_cluster]
        paths = [os.path.join(images_root, keys[i]) for i in taken if os.path.exists(os.path.join(images_root, keys[i]))]
        paths_per_cluster.append(paths)
    return paths_per_cluster


def get_top_indices_by_metric(values, top_n):
    valid = ~np.isnan(values)
    if not valid.any():
        return []
    order = np.argsort(-np.where(valid, values, -np.inf))[:top_n]
    return [int(i) for i in order if valid[i]]


# -------------------- fonts --------------------

_font_cache = {}

def _get_cached_font(font_path, size):
    key = (font_path, size)
    if key not in _font_cache:
        try:
            _font_cache[key] = ImageFont.truetype(font_path, size)
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def _font_fitting_width(draw, text, max_width, font_path, initial_size=11):
    for size in range(initial_size, 5, -1):
        font = _get_cached_font(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
    return _get_cached_font(font_path, 6)


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{float(v):.3f}"


# -------------------- image annotation --------------------

def add_metadata_label(img, b, cluster_id, bg_clip=None, bg_ssim_v=None, fg_clip=None):
    """Add metadata strip below image: cluster, quadrants, seg metrics."""
    img = img.copy().convert("RGB")
    w, h = img.size
    q0, q1, q2, q3 = quadrant_ones(b)
    strip_h = METADATA_STRIP_H
    canvas = Image.new("RGB", (w, h + strip_h), (248, 248, 248))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    max_w = w - 8
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    line1 = f"C{cluster_id} | {q0},{q1},{q2},{q3}"
    line2 = f"bg:{_fmt(bg_clip)} ss:{_fmt(bg_ssim_v)} fg:{_fmt(fg_clip)}"
    font1 = _font_fitting_width(draw, line1, max_w, font_path, initial_size=10)
    font2 = _font_fitting_width(draw, line2, max_w, font_path, initial_size=10)
    y0 = h + 2
    draw.text((4, y0), line1, fill="black", font=font1)
    bbox1 = draw.textbbox((0, 0), line1, font=font1)
    draw.text((4, y0 + bbox1[3] - bbox1[1] + 2), line2, fill="black", font=font2)
    return canvas


def draw_cluster_id(img, cluster_id):
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    text = f"C{cluster_id}"
    font_size = max(12, min(img.size) // 12)
    font = _get_cached_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 4
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill="black", outline="white")
    draw.text((pad, pad), text, fill="white", font=font)
    return img


# -------------------- grid builders --------------------

ROW_LABEL_W = 72


def build_grid(paths_list, out_path, thumb, cluster_ids_for_rows, keys, labels,
               bg_clip_sim, bg_ssim_arr, fg_clip_score):
    if paths_list and isinstance(paths_list[0], (list, tuple)):
        flat = []
        for row in paths_list:
            flat.extend(row)
        n_cols = max(len(row) for row in paths_list) if paths_list else 0
        n_rows = len(paths_list)
    else:
        flat = list(paths_list)
        n_cols = max(1, len(flat))
        n_rows = math.ceil(len(flat) / n_cols)
    if not flat:
        return
    n_cols = min(n_cols, len(flat))
    cell_h = thumb + METADATA_STRIP_H
    has_row_labels = cluster_ids_for_rows is not None and len(cluster_ids_for_rows) >= n_rows
    canvas_w = (ROW_LABEL_W if has_row_labels else 0) + n_cols * thumb
    canvas = Image.new("RGB", (canvas_w, n_rows * cell_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    row_font = _get_cached_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    x_offset = ROW_LABEL_W if has_row_labels else 0
    key_to_idx = {k: i for i, k in enumerate(keys)} if keys is not None else {}
    for i, path in enumerate(flat):
        r, c = divmod(i, n_cols)
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb, thumb))
            basename = os.path.basename(path)
            b = extract_b_value(basename) or 0
            idx = key_to_idx.get(basename)
            if idx is not None and labels is not None:
                cid = int(labels[idx])
                img = add_metadata_label(img, b, cid,
                    bg_clip_sim[idx] if bg_clip_sim is not None else None,
                    bg_ssim_arr[idx] if bg_ssim_arr is not None else None,
                    fg_clip_score[idx] if fg_clip_score is not None else None,
)
            else:
                cid = int(cluster_ids_for_rows[r]) if has_row_labels else -1
                img = add_metadata_label(img, b, cid)
            canvas.paste(img, (x_offset + c * thumb, r * cell_h))
        except Exception:
            pass
    if has_row_labels:
        for r in range(n_rows):
            lab = cluster_ids_for_rows[r]
            y_center = r * cell_h + cell_h // 2
            text = f"Cluster {lab}"
            bbox = draw.textbbox((0, 0), text, font=row_font)
            draw.text((4, y_center - (bbox[3] - bbox[1]) // 2), text, fill="black", font=row_font)
    canvas.save(out_path)
    print("Saved:", out_path)


def build_grid_one_per_cluster(key_indices, keys, labels, coords, images_root, out_path, thumb,
                                bg_clip_sim, bg_ssim_arr, fg_clip_score):
    n = len(key_indices)
    if n == 0:
        return
    side = math.ceil(math.sqrt(n))
    items = []
    for lab, idx in key_indices:
        p = os.path.join(images_root, keys[idx])
        items.append((lab, idx, p if os.path.exists(p) else None))
    while len(items) < side * side:
        items.append((None, None, None))
    cell_h = thumb + METADATA_STRIP_H
    canvas = Image.new("RGB", (side * thumb, side * cell_h), (240, 240, 240))
    for i in range(side * side):
        r, c = divmod(i, side)
        lab, idx, path = items[i]
        if path is None:
            continue
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb, thumb))
            if lab is not None:
                img = draw_cluster_id(img, lab)
            b = extract_b_value(keys[idx]) if idx is not None else 0
            img = add_metadata_label(img, b or 0, int(lab) if lab is not None else -1,
                bg_clip_sim[idx] if bg_clip_sim is not None and idx is not None else None,
                bg_ssim_arr[idx] if bg_ssim_arr is not None and idx is not None else None,
                fg_clip_score[idx] if fg_clip_score is not None and idx is not None else None,
)
            canvas.paste(img, (c * thumb, r * cell_h))
        except Exception:
            pass
    canvas.save(out_path)
    print("Saved:", out_path)


def build_grid_metric_top(indices, keys, labels, images_root, out_path, thumb, cols,
                           bg_clip_sim, bg_ssim_arr, fg_clip_score):
    n = len(indices)
    if n == 0:
        return
    cell_h = thumb + METADATA_STRIP_H
    n_cols = min(cols, n)
    n_rows = math.ceil(n / n_cols)
    canvas = Image.new("RGB", (n_cols * thumb, n_rows * cell_h), (240, 240, 240))
    for pos, i in enumerate(indices):
        r, c = divmod(pos, n_cols)
        path = os.path.join(images_root, keys[i])
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb, thumb))
            img = draw_cluster_id(img, int(labels[i]))
            b = extract_b_value(keys[i]) or 0
            img = add_metadata_label(img, b, int(labels[i]),
                bg_clip_sim[i] if bg_clip_sim is not None else None,
                bg_ssim_arr[i] if bg_ssim_arr is not None else None,
                fg_clip_score[i] if fg_clip_score is not None else None,
)
            canvas.paste(img, (c * thumb, r * cell_h))
        except Exception:
            pass
    canvas.save(out_path)
    print("Saved:", out_path)


# -------------------- UMAP maps --------------------

def map_clusters(coords, labels, cluster_indices, out_dir, max_labels=40):
    """Cluster map. Only labels the top `max_labels` clusters by size to avoid overlap."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.scatter(coords[:, 0], coords[:, 1], s=0.5, c=labels, cmap="tab20", alpha=0.5, rasterized=True)

    texts = []
    for lab in sorted(cluster_indices.keys()):
        idx = cluster_indices[lab]
        center = coords[idx].mean(axis=0)
        t = ax.annotate(str(lab), xy=(center[0], center[1]),
                        fontsize=8, weight="bold", ha="center", va="center",
                        color="black",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85, lw=0.5))
        texts.append(t)

    # Try adjustText for automatic label repulsion
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    except ImportError:
        pass  # labels stay at centroids

    n_total = len(cluster_indices)
    ax.set_title(f"Clusters ({n_total} total)")
    ax.axis("off")
    path = os.path.join(out_dir, "map_clusters.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def map_seg_metrics(coords, out_dir, bg_clip_sim, bg_ssim, fg_clip_score, combined):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    panels = [
        (axes[0, 0], bg_ssim, "bg_ssim (background preservation)", "cividis"),
        (axes[0, 1], fg_clip_score, "fg_clip_score (prompt following)", "viridis"),
        (axes[1, 0], bg_clip_sim, "bg_clip_similarity", "plasma"),
        (axes[1, 1], combined, "combined: sqrt(bg_ssim * fg_clip)", "RdYlGn"),
    ]
    for ax, values, title, cmap in panels:
        valid = ~np.isnan(values)
        if valid.any():
            sc = ax.scatter(coords[valid, 0], coords[valid, 1], s=1,
                            c=values[valid], cmap=cmap, alpha=0.7, rasterized=True)
            ax.set_title(title)
            plt.colorbar(sc, ax=ax)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=1, c="lightgray", alpha=0.6, rasterized=True)
            ax.set_title(f"{title} (no data)")
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, "map_seg_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser(description="Cluster visualizations with seg metrics")
    parser.add_argument("--emb", default=EMB_PATH)
    parser.add_argument("--images", default=IMAGES_ROOT)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--metrics", default="seg_metrics.csv", help="Segmentation metrics CSV")
    parser.add_argument("--coords", default=COORDS_PATH)
    parser.add_argument("--labels", default=LABELS_PATH)
    parser.add_argument("--no-umap", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for cuML UMAP")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load embeddings
    print("Loading embeddings...")
    X, keys = load_embeddings_and_keys(args.emb)
    print(f"  {len(keys)} embeddings, dim={X.shape[1]}")

    # Coords
    print("Loading or computing coords...")
    if args.coords and os.path.isfile(args.coords):
        coords = np.load(args.coords)
        if len(coords) != len(keys):
            print(f"  Coords size mismatch ({len(coords)} vs {len(keys)}), recomputing...")
            coords = compute_coords(X, use_umap=not args.no_umap, gpu_id=args.gpu)
            np.save(os.path.join(args.out, "coords.npy"), coords)
    else:
        coords = compute_coords(X, use_umap=not args.no_umap, gpu_id=args.gpu)
        np.save(os.path.join(args.out, "coords.npy"), coords)

    # Labels
    print("Loading or computing labels...")
    if args.labels and os.path.isfile(args.labels):
        labels = np.load(args.labels)
        if len(labels) != len(keys):
            print(f"  Labels size mismatch ({len(labels)} vs {len(keys)}), recomputing...")
            labels = _compute_labels(coords, args.gpu)
            np.save(os.path.join(args.out, "labels.npy"), labels)
    else:
        labels = _compute_labels(coords, args.gpu)
        np.save(os.path.join(args.out, "labels.npy"), labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Clusters (excl. noise): {n_clusters}")

    cluster_indices = {
        int(lab): np.where(labels == lab)[0]
        for lab in set(labels.tolist()) if lab != -1
    }

    # Load seg metrics
    metrics_dict = {}
    bg_clip_sim = bg_ssim = fg_clip_score = combined = None
    if args.metrics and os.path.isfile(args.metrics):
        metrics_dict = load_seg_metrics_csv(args.metrics)
        print(f"Loaded seg metrics: {len(metrics_dict)} rows from {args.metrics}")
        bg_clip_sim, bg_ssim, fg_clip_score, combined = get_seg_metrics_arrays(keys, metrics_dict)
    else:
        print(f"Metrics file not found: {args.metrics}")

    # 1) Cluster map
    print("\n--- Building visualizations ---")
    map_clusters(coords, labels, cluster_indices, args.out)

    # 2) Seg metrics on UMAP map
    if bg_clip_sim is not None:
        map_seg_metrics(coords, args.out, bg_clip_sim, bg_ssim, fg_clip_score, combined)

    # 3) Grid: top 20 clusters × 5
    cluster_sizes_top = get_cluster_indices_by_size(labels, TOP_N_CLUSTERS)
    cluster_ids_for_rows = [lab for lab, _ in cluster_sizes_top]
    paths_top = get_top_cluster_image_paths(keys, labels, coords, args.images,
                                             TOP_N_CLUSTERS, IMAGES_PER_ROW_TOP, cluster_indices)
    build_grid(paths_top, os.path.join(args.out, "grid_top20_x5.png"), THUMB,
               cluster_ids_for_rows, keys, labels, bg_clip_sim, bg_ssim, fg_clip_score)

    # 4) Grid: one per cluster
    one_per = get_one_image_per_cluster(coords, cluster_indices)
    build_grid_one_per_cluster(one_per, keys, labels, coords, args.images,
                                os.path.join(args.out, "grid_one_per_cluster.png"), THUMB,
                                bg_clip_sim, bg_ssim, fg_clip_score)

    # 5) Top grids by seg metrics
    if bg_clip_sim is not None:
        metric_grids = [
            ("grid_top_bg_ssim.png", bg_ssim),
            ("grid_top_fg_clip_score.png", fg_clip_score),
            ("grid_top_bg_clip_sim.png", bg_clip_sim),
            ("grid_top_combined.png", combined),  # best balance of bg preservation + prompt following
        ]
        for fname, values in metric_grids:
            top_idx = get_top_indices_by_metric(values, TOP_N_BY_METRIC)
            if top_idx:
                build_grid_metric_top(top_idx, keys, labels, args.images,
                                       os.path.join(args.out, fname), THUMB, COLS_METRIC_GRID,
                                       bg_clip_sim, bg_ssim, fg_clip_score)

    print(f"\nDone. Output dir: {args.out}")


if __name__ == "__main__":
    main()
