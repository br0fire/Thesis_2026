#!/bin/bash
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

EXTRACT_PID=194642

echo "=== Waiting for DINOv2 extraction (PID $EXTRACT_PID) ==="
while kill -0 "$EXTRACT_PID" 2>/dev/null; do sleep 15; done
echo "=== Extraction done at $(date) ==="

# ─── Merge missing features into main pkl ───
echo ""
echo "=== Merge features ==="
$PYTHON -c "
import pickle
with open('analysis/feature_dictionary_horse.pkl', 'rb') as f:
    d = pickle.load(f)
print(f'  Existing: {len(d)}')
with open('/tmp/horse_missing_features.pkl', 'rb') as f:
    d2 = pickle.load(f)
print(f'  Missing extracted: {len(d2)}')
d.update(d2)
with open('analysis/feature_dictionary_horse.pkl', 'wb') as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f'  Merged total: {len(d)}')
"
echo "Merge done at $(date)"

# ─── Seg metrics (full 1M) ───
echo ""
echo "=== Seg metrics ==="
$PYTHON metrics/calc_seg_metrics.py \
    --gpus 0,1,2,3,4,5,6,7 \
    --batch_size 256 \
    --num_workers 8 \
    --source_image generation/horse.jpg \
    --mask metrics/background_mask_horse.npy \
    --source_prompt "a horse on the grass" \
    --target_prompt "a robot horse on the grass" \
    --images_dir /home/jovyan/shares/SR006.nfs3/svgrozny/generated_horse_300k \
    --output_csv metrics/seg_metrics_horse.csv
echo "Metrics done at $(date)"

# ─── Analysis ───
echo ""
echo "=== Analysis ==="
CUDA_VISIBLE_DEVICES=4 $PYTHON -u analysis/visualize_clusters.py \
    --emb analysis/feature_dictionary_horse.pkl \
    --images /home/jovyan/shares/SR006.nfs3/svgrozny/generated_horse_300k \
    --out analysis/horse_analysis \
    --metrics metrics/seg_metrics_horse.csv \
    --gpu 0
echo "Analysis done at $(date)"

# ─── Cleanup ───
rm -rf /tmp/horse_missing_images /tmp/horse_missing_features.pkl /tmp/horse_missing_features.txt

echo ""
echo "=== ALL DONE at $(date) ==="
