"""Analyze bit patterns across clusters for all 3 experiments."""
import os, pickle, numpy as np, csv

PROJ = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project"
NFS3 = "/home/jovyan/shares/SR006.nfs3/svgrozny"

EXPERIMENTS = {
    "catdog": {
        "images_dir": f"{NFS3}/generated_samples_40step",
        "labels": f"{PROJ}/analysis/catdog_analysis/labels.npy",
        "coords": f"{PROJ}/analysis/catdog_analysis/coords.npy",
        "emb": f"{PROJ}/analysis/embeddings/feature_dictionary_catdog.pkl",
        "metrics_csv": f"{PROJ}/metrics/results/seg_metrics_catdog.csv",
    },
    "horse": {
        "images_dir": f"{NFS3}/generated_horse_300k",
        "labels": f"{PROJ}/analysis/horse_analysis/labels.npy",
        "coords": f"{PROJ}/analysis/horse_analysis/coords.npy",
        "emb": f"{PROJ}/analysis/embeddings/feature_dictionary_horse.pkl",
        "metrics_csv": f"{PROJ}/metrics/results/seg_metrics_horse.csv",
    },
    "room": {
        "images_dir": f"{NFS3}/generated_room",
        "labels": f"{PROJ}/analysis/room_analysis/labels.npy",
        "coords": f"{PROJ}/analysis/room_analysis/coords.npy",
        "emb": f"{PROJ}/analysis/embeddings/feature_dictionary_room.pkl",
        "metrics_csv": f"{PROJ}/metrics/results/seg_metrics_room.csv",
    },
}

N_BITS = 20


def fname_to_b(fname):
    """Extract integer b from filename like path_00123_b456789.jpg"""
    base = os.path.splitext(fname)[0]
    parts = base.split("_b")
    if len(parts) == 2:
        return int(parts[1])
    return None


def b_to_bits(b, n=N_BITS):
    return np.array([(b >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.int8)


def load_keys(emb_path):
    with open(emb_path, "rb") as f:
        emb = pickle.load(f)
    return list(emb.keys())


def analyze_experiment(name, cfg):
    print(f"\n{'='*70}")
    print(f"  {name.upper()}")
    print(f"{'='*70}")

    keys = load_keys(cfg["emb"])
    labels = np.load(cfg["labels"])
    n = len(keys)

    # Build bit matrix (N, 20)
    bits_matrix = np.zeros((n, N_BITS), dtype=np.int8)
    total_ones = np.zeros(n, dtype=np.int32)
    for i, fn in enumerate(keys):
        b = fname_to_b(fn)
        if b is not None:
            bits = b_to_bits(b)
            bits_matrix[i] = bits
            total_ones[i] = bits.sum()

    # Load metrics
    metrics = {}
    if os.path.isfile(cfg["metrics_csv"]):
        with open(cfg["metrics_csv"]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics[row["filename"]] = row

    # --- Global bit importance ---
    print("\n--- Bit-level statistics (across all images) ---")
    # Correlation of each bit with fg_clip_score
    fg_scores = np.full(n, np.nan)
    bg_ssim_arr = np.full(n, np.nan)
    bg_clip_arr = np.full(n, np.nan)
    for i, fn in enumerate(keys):
        if fn in metrics:
            fg_scores[i] = float(metrics[fn].get("fg_clip_score", "nan"))
            bg_ssim_arr[i] = float(metrics[fn].get("bg_ssim", "nan"))
            bg_clip_arr[i] = float(metrics[fn].get("bg_clip_similarity", "nan"))

    valid = ~np.isnan(fg_scores)
    print(f"\nCorrelation of each bit position with fg_clip_score:")
    fg_corr = []
    for bit_idx in range(N_BITS):
        corr = np.corrcoef(bits_matrix[valid, bit_idx], fg_scores[valid])[0, 1]
        fg_corr.append(corr)
    # Print as bar chart
    for i, c in enumerate(fg_corr):
        bar = "+" * int(abs(c) * 50) if c >= 0 else "-" * int(abs(c) * 50)
        step_range = f"steps {i*2+1:2d}-{i*2+2:2d}"
        print(f"  bit {i:2d} ({step_range}): {c:+.4f}  {'|' + bar}")

    print(f"\nCorrelation of each bit position with bg_ssim:")
    bg_corr = []
    for bit_idx in range(N_BITS):
        corr = np.corrcoef(bits_matrix[valid, bit_idx], bg_ssim_arr[valid])[0, 1]
        bg_corr.append(corr)
    for i, c in enumerate(bg_corr):
        bar = "+" * int(abs(c) * 50) if c >= 0 else "-" * int(abs(c) * 50)
        step_range = f"steps {i*2+1:2d}-{i*2+2:2d}"
        print(f"  bit {i:2d} ({step_range}): {c:+.4f}  {'|' + bar}")

    # --- Total ones vs metrics ---
    print(f"\nTotal ones (how many target steps) vs metrics:")
    for num_ones in [0, 5, 10, 15, 20]:
        mask = total_ones == num_ones
        cnt = mask.sum()
        if cnt > 0:
            fg_mean = np.nanmean(fg_scores[mask])
            bg_mean = np.nanmean(bg_ssim_arr[mask])
            bgc_mean = np.nanmean(bg_clip_arr[mask])
            print(f"  {num_ones:2d}/20 ones ({cnt:7d} imgs): fg_clip={fg_mean:+.4f}  bg_ssim={bg_mean:.4f}  bg_clip={bgc_mean:.4f}")

    # --- Quadrant analysis ---
    print(f"\n--- Quadrant analysis (which phase matters most) ---")
    quadrants = {
        "Q0 (steps 1-10, early)": bits_matrix[:, 0:5].sum(axis=1),
        "Q1 (steps 11-20)": bits_matrix[:, 5:10].sum(axis=1),
        "Q2 (steps 21-30)": bits_matrix[:, 10:15].sum(axis=1),
        "Q3 (steps 31-40, late)": bits_matrix[:, 15:20].sum(axis=1),
    }
    print(f"\n  Quadrant correlations:")
    print(f"  {'Quadrant':<30s} {'fg_clip':>10s} {'bg_ssim':>10s} {'bg_clip':>10s}")
    for qname, qvals in quadrants.items():
        fc = np.corrcoef(qvals[valid], fg_scores[valid])[0, 1]
        bc = np.corrcoef(qvals[valid], bg_ssim_arr[valid])[0, 1]
        bcc = np.corrcoef(qvals[valid], bg_clip_arr[valid])[0, 1]
        print(f"  {qname:<30s} {fc:+10.4f} {bc:+10.4f} {bcc:+10.4f}")

    # --- Top clusters bit profiles ---
    print(f"\n--- Top 10 clusters by size: bit profiles ---")
    unique_labels = sorted(set(labels.tolist()))
    unique_labels = [l for l in unique_labels if l >= 0]
    cluster_sizes = [(l, (labels == l).sum()) for l in unique_labels]
    cluster_sizes.sort(key=lambda x: -x[1])

    print(f"  {'Cluster':>8s} {'Size':>8s} {'Ones':>6s} {'Q0':>4s} {'Q1':>4s} {'Q2':>4s} {'Q3':>4s}  {'fg_clip':>8s} {'bg_ssim':>8s}  Bit pattern (avg)")
    for cid, csize in cluster_sizes[:10]:
        cmask = labels == cid
        c_bits = bits_matrix[cmask].mean(axis=0)
        c_ones = total_ones[cmask].mean()
        q0 = c_bits[0:5].sum()
        q1 = c_bits[5:10].sum()
        q2 = c_bits[10:15].sum()
        q3 = c_bits[15:20].sum()
        fg_m = np.nanmean(fg_scores[cmask])
        bg_m = np.nanmean(bg_ssim_arr[cmask])
        bit_str = "".join(["#" if v > 0.6 else "." if v < 0.4 else "~" for v in c_bits])
        print(f"  C{cid:>6d} {csize:8d} {c_ones:6.1f} {q0:4.1f} {q1:4.1f} {q2:4.1f} {q3:4.1f}  {fg_m:+8.4f} {bg_m:8.4f}  {bit_str}")

    # --- Best combined images: what bit pattern? ---
    print(f"\n--- Bit profiles of top-50 images by combined score ---")
    valid_idx = np.where(valid)[0]
    bg_v = np.clip(bg_ssim_arr[valid_idx], 0, None)
    fg_v = np.clip(fg_scores[valid_idx], 0, None)
    bg_min, bg_max = bg_v.min(), bg_v.max()
    fg_min, fg_max = fg_v.min(), fg_v.max()
    bg_norm = (bg_v - bg_min) / (bg_max - bg_min + 1e-8)
    fg_norm = (fg_v - fg_min) / (fg_max - fg_min + 1e-8)
    combined = np.sqrt(bg_norm * fg_norm)
    top50_local = np.argsort(-combined)[:50]
    top50_global = valid_idx[top50_local]

    top_bits = bits_matrix[top50_global]
    top_ones = total_ones[top50_global]
    print(f"  Mean ones: {top_ones.mean():.1f} / 20")
    print(f"  Mean bit profile: {' '.join(f'{v:.2f}' for v in top_bits.mean(axis=0))}")
    print(f"  Quadrant means: Q0={top_bits[:, 0:5].sum(axis=1).mean():.2f}  Q1={top_bits[:, 5:10].sum(axis=1).mean():.2f}  Q2={top_bits[:, 10:15].sum(axis=1).mean():.2f}  Q3={top_bits[:, 15:20].sum(axis=1).mean():.2f}")

    return fg_corr, bg_corr


for name, cfg in EXPERIMENTS.items():
    analyze_experiment(name, cfg)
