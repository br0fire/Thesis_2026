#!/bin/bash
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache

SOURCE_PROMPT="a tabby cat walking on stone pavement, photo"
TARGET_PROMPT="a golden retriever dog walking on stone pavement, photo"
OUTPUT_DIR="/home/jovyan/shares/SR006.nfs3/svgrozny/generated_flux_catdog"
SEG_PROMPT="cat"
N_BITS=${N_BITS:-14}
SEED=${SEED:-42}
START_STEP=${START_STEP:-1}
NUM_PATHS=${NUM_PATHS:-$((1 << N_BITS))}
TARGET_B=$(( (1 << N_BITS) - 1 ))
mkdir -p "$OUTPUT_DIR"

# ─────────────────────────────────────────────
# Step 1: Generate all 2^14 paths via FLUX
# ─────────────────────────────────────────────
if [ "$START_STEP" -le 1 ]; then
echo "=== Step 1: FLUX Generation (${NUM_PATHS} paths) ==="
echo "Started at $(date)"
$PYTHON generation/flux_generate.py \
    --source_prompt "$SOURCE_PROMPT" \
    --target_prompt "$TARGET_PROMPT" \
    --num_paths $NUM_PATHS \
    --n_bits $N_BITS \
    --gpus 0,1,2,3,4,5,6,7 \
    --output_dir "$OUTPUT_DIR" \
    --height 512 \
    --width 512 \
    --guidance_scale 4.0 \
    --seed $SEED \
    --batch_size 16
echo "Step 1 done at $(date)"
fi

# ─────────────────────────────────────────────
# Step 2: Segmentation (on source path_00000)
# ─────────────────────────────────────────────
if [ "$START_STEP" -le 2 ]; then
echo ""
echo "=== Step 2: Segmentation ==="
echo "Started at $(date)"
# Use the all-source image as reference for segmentation
$PYTHON metrics/segment_source.py \
    --image "$OUTPUT_DIR/path_00000_b0.jpg" \
    --prompt "$SEG_PROMPT" \
    --method clipseg \
    --dilate 10 \
    --threshold 0.5 \
    --output /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/metrics/background_mask_flux_catdog.npy \
    --save_vis
echo "Step 2 done at $(date)"
fi

# ─────────────────────────────────────────────
# Step 3: Compute seg metrics
# ─────────────────────────────────────────────
if [ "$START_STEP" -le 3 ]; then
echo ""
echo "=== Step 3: Seg metrics ==="
echo "Started at $(date)"
$PYTHON metrics/calc_seg_metrics.py \
    --gpus 0,1,2,3,4,5,6,7 \
    --batch_size 256 \
    --num_workers 8 \
    --source_image "$OUTPUT_DIR/path_00000_b0.jpg" \
    --mask metrics/background_mask_flux_catdog.npy \
    --source_prompt "$SOURCE_PROMPT" \
    --target_prompt "$TARGET_PROMPT" \
    --images_dir "$OUTPUT_DIR" \
    --output_csv metrics/seg_metrics_flux_catdog.csv
echo "Step 3 done at $(date)"
fi

# ─────────────────────────────────────────────
# Step 4: Extract DINOv2 features
# ─────────────────────────────────────────────
if [ "$START_STEP" -le 4 ]; then
echo ""
echo "=== Step 4: Extract DINOv2 features ==="
echo "Started at $(date)"
$PYTHON metrics/extract_dino_features.py \
    --gpus 0,1,2,3,4,5,6,7 \
    --batch_size 512 \
    --num_workers 8 \
    --images_dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/feature_dictionary.pkl"
echo "Step 4 done at $(date)"
fi

# ─────────────────────────────────────────────
# Step 5: Analysis (UMAP + HDBSCAN + grids)
# ─────────────────────────────────────────────
if [ "$START_STEP" -le 5 ]; then
echo ""
echo "=== Step 5: Analysis ==="
echo "Started at $(date)"
cp "$OUTPUT_DIR/feature_dictionary.pkl" analysis/feature_dictionary_flux_catdog.pkl
mkdir -p analysis/flux_catdog_analysis
# Copy source (all-0) and target (all-1) paths
cp "$OUTPUT_DIR"/*_b0.jpg analysis/flux_catdog_analysis/path_source.jpg 2>/dev/null || true
cp "$OUTPUT_DIR"/*_b${TARGET_B}.jpg analysis/flux_catdog_analysis/path_target.jpg 2>/dev/null || true
# Copy seg vis
MASK_VIS="/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/metrics/background_mask_flux_catdog_vis.jpg"
[ -f "$MASK_VIS" ] && cp "$MASK_VIS" analysis/flux_catdog_analysis/
N_BITS=$N_BITS CUDA_VISIBLE_DEVICES=4 $PYTHON -u analysis/visualize_clusters.py \
    --emb analysis/feature_dictionary_flux_catdog.pkl \
    --images "$OUTPUT_DIR" \
    --out analysis/flux_catdog_analysis \
    --metrics metrics/seg_metrics_flux_catdog.csv \
    --gpu 0 \
    --min_cluster_size 100 \
    --min_samples 10
echo "Step 5 done at $(date)"
fi

echo ""
echo "=== ALL DONE at $(date) ==="
echo "Images: $(ls "$OUTPUT_DIR" | wc -l)"
