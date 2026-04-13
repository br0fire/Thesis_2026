#!/bin/bash
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

SOURCE_PROMPT="A coffee table and a sofa in a modern living room."
TARGET_PROMPT="An aquarium and a sofa in a modern living room."
IMAGE=/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/generation/room.png
OUTPUT_DIR="/home/jovyan/shares/SR006.nfs3/svgrozny/generated_room"
SEG_PROMPT="coffee table"
START_STEP=${START_STEP:-1}
NUM_PATHS=${NUM_PATHS:-1048576}
mkdir -p "$OUTPUT_DIR"

# ─────────────────────────────────────────────
# Step 1: Generate images
# ─────────────────────────────────────────────
if [ "$START_STEP" -le 1 ]; then
echo "=== Step 1: Generation (${NUM_PATHS} paths) ==="
echo "Started at $(date)"
$PYTHON -c "
import torch
from generation.test import run_targeted_search
run_targeted_search(
    source_prompt='''$SOURCE_PROMPT''',
    target_prompt='''$TARGET_PROMPT''',
    input_image='$IMAGE',
    steps_total=40,
    mask_bits=20,
    num_paths_limit=$NUM_PATHS,
    gpu_ids=list(range(torch.cuda.device_count())),
    output_dir='$OUTPUT_DIR',
)
"
echo "Step 1 done at $(date)"
fi

# ─────────────────────────────────────────────
# Step 2: Segment source object (coffee table)
# ─────────────────────────────────────────────
if [ "$START_STEP" -le 2 ]; then
echo ""
echo "=== Step 2: Segmentation ==="
echo "Started at $(date)"
$PYTHON metrics/segment_source.py \
    --image "$IMAGE" \
    --prompt "$SEG_PROMPT" \
    --method clipseg \
    --dilate 10 \
    --threshold 0.5 \
    --output /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/metrics/background_mask_room.npy \
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
    --source_image "$IMAGE" \
    --mask metrics/background_mask_room.npy \
    --source_prompt "$SOURCE_PROMPT" \
    --target_prompt "$TARGET_PROMPT" \
    --images_dir "$OUTPUT_DIR" \
    --output_csv metrics/seg_metrics_room.csv
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
cp "$OUTPUT_DIR/feature_dictionary.pkl" analysis/feature_dictionary_room.pkl
# Copy segmentation visualization to analysis output
MASK_VIS="/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/metrics/background_mask_room_vis.jpg"
[ -f "$MASK_VIS" ] && cp "$MASK_VIS" analysis/room_analysis/
# Copy source (all-0) and target (all-1) path images
cp "$OUTPUT_DIR"/path_00000_b0.jpg analysis/room_analysis/path_source_b0.jpg
cp "$OUTPUT_DIR"/path_00001_b1048575.jpg analysis/room_analysis/path_target_b1048575.jpg
CUDA_VISIBLE_DEVICES=4 $PYTHON -u analysis/visualize_clusters.py \
    --emb analysis/feature_dictionary_room.pkl \
    --images "$OUTPUT_DIR" \
    --out analysis/room_analysis \
    --metrics metrics/seg_metrics_room.csv \
    --gpu 0 \
    --min_cluster_size 5000 \
    --min_samples 100
echo "Step 5 done at $(date)"
fi

echo ""
echo "=== ALL DONE at $(date) ==="
echo "Images: $(ls "$OUTPUT_DIR" | wc -l)"
