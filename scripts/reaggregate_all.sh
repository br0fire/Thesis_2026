#!/bin/bash
# Re-run aggregator for all completed sweeps with updated random_samples=640.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

for sweep_dir in analysis/reinforce_analysis/sweep_*; do
    [ -d "$sweep_dir/configs" ] || continue
    name=$(basename "$sweep_dir")
    echo ""
    echo "=== [$(date)] Re-aggregating: $name ==="
    $PYTHON analysis/parallel_sweep_aggregate.py \
        --sweep_dir "$sweep_dir" --gpu 0 --random_samples 640 \
        2>&1 | tail -10
    echo "[$(date)] $name done (rc=$?)"
done

echo ""
echo "=== All re-aggregation done at $(date) ==="
