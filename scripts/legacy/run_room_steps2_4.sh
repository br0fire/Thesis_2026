#!/bin/bash
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

SOURCE_PROMPT="A high-resolution photo of a modern living room with a large gray fabric sofa, a wooden coffee table in front of it, a soft beige rug, and a floor-to-ceiling window letting in natural daylight. Minimalist interior design, clean lines, neutral color palette, cozy atmosphere, realistic lighting, 4k, interior photography style."
TARGET_PROMPT="A high-resolution photo of a modern living room with a large gray fabric sofa, a rectangular glass aquarium filled with clear water, colorful fish, and aquatic plants placed in front of it, a soft beige rug, and a floor-to-ceiling window letting in natural daylight. Minimalist interior design, clean lines, neutral color palette, cozy atmosphere, realistic lighting, 4k, interior photography style."
IMAGE=/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/generation/room.png
OUTPUT_DIR="/home/jovyan/shares/SR006.nfs3/svgrozny/generated_room"
SEG_PROMPT="wooden coffee table"

# Step 2: Segmentation
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

# Step 3: Seg metrics
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

# Step 4: Analysis
echo ""
echo "=== Step 4: Analysis ==="
echo "Started at $(date)"
cp "$OUTPUT_DIR/feature_dictionary.pkl" analysis/feature_dictionary_room.pkl
CUDA_VISIBLE_DEVICES=4 $PYTHON -u analysis/visualize_clusters.py \
    --emb analysis/feature_dictionary_room.pkl \
    --images "$OUTPUT_DIR" \
    --out analysis/room_analysis \
    --metrics metrics/seg_metrics_room.csv \
    --gpu 0
echo "Step 4 done at $(date)"

echo ""
echo "=== ALL DONE at $(date) ==="
echo "Images: $(ls "$OUTPUT_DIR" | wc -l)"
