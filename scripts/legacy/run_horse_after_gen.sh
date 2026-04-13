#!/bin/bash
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

GEN_PID=119875

echo "=== Waiting for resume_horse.py (PID $GEN_PID) ==="
while kill -0 "$GEN_PID" 2>/dev/null; do sleep 30; done
echo "=== Generation done at $(date) ==="
echo "Images: $(ls /home/jovyan/shares/SR006.nfs3/svgrozny/generated_horse_300k/ | wc -l)"

# ─── Merge features ───
echo ""
echo "=== Merge DINOv2 features ==="
$PYTHON -c "
import pickle, os
merged = {}
for f in ['analysis/feature_dictionary_horse.pkl', 'analysis/feature_dictionary_horse_resume.pkl']:
    if os.path.isfile(f):
        with open(f, 'rb') as fh:
            d = pickle.load(fh)
        print(f'  Loaded {len(d)} from {f}')
        merged.update(d)
out_feat = '/home/jovyan/shares/SR006.nfs3/svgrozny/generated_horse_300k/feature_dictionary.pkl'
if os.path.isfile(out_feat):
    with open(out_feat, 'rb') as fh:
        d = pickle.load(fh)
    print(f'  Loaded {len(d)} from {out_feat}')
    merged.update(d)
with open('analysis/feature_dictionary_horse.pkl', 'wb') as f:
    pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f'Merged: {len(merged)} total features')
"
echo "Merge done at $(date)"

# ─── Seg metrics ───
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

echo ""
echo "=== ALL DONE at $(date) ==="
