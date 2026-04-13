#!/bin/bash
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

CATDOG_PID=37122
HORSE_PID=45048
PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

echo "=== Waiting for cat-dog metrics (PID $CATDOG_PID) ==="
while kill -0 "$CATDOG_PID" 2>/dev/null; do sleep 30; done
echo "=== Cat-dog metrics done at $(date) ==="

echo "=== Waiting for horse metrics (PID $HORSE_PID) ==="
while kill -0 "$HORSE_PID" 2>/dev/null; do sleep 30; done
echo "=== Horse metrics done at $(date) ==="

# Cat-dog analysis
echo ""
echo "=== Cat-dog analysis ==="
rm -f analysis/catdog_analysis/labels.npy  # force recluster with new metrics
CUDA_VISIBLE_DEVICES=4 $PYTHON -u analysis/visualize_clusters.py \
    --emb analysis/feature_dictionary_catdog.pkl \
    --images /home/jovyan/shares/SR006.nfs3/svgrozny/generated_samples_40step \
    --out analysis/catdog_analysis \
    --metrics metrics/seg_metrics_catdog.csv \
    --coords analysis/catdog_analysis/coords.npy \
    --gpu 0
echo "Cat-dog analysis done at $(date)"

# Horse analysis
echo ""
echo "=== Horse analysis ==="
CUDA_VISIBLE_DEVICES=4 $PYTHON -u analysis/visualize_clusters.py \
    --emb analysis/feature_dictionary_horse.pkl \
    --images /home/jovyan/shares/SR006.nfs3/svgrozny/generated_horse_300k \
    --out analysis/horse_analysis \
    --metrics metrics/seg_metrics_horse.csv \
    --gpu 0
echo "Horse analysis done at $(date)"

echo ""
echo "=== ALL DONE at $(date) ==="
