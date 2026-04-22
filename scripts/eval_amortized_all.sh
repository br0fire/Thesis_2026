#!/bin/bash
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python

# Read all experiment names
readarray -t NAMES < <($PYTHON -c "
import json
d = json.load(open('analysis/reinforce_analysis/amortized/predictions.json'))
for k in sorted(d.keys()):
    print(k)
")
echo "Eval on ${#NAMES[@]} experiments"

GPU=0
pids=()
for name in "${NAMES[@]}"; do
    echo "GPU $GPU: $name"
    nohup $PYTHON -u analysis/eval_amortized.py --experiment "$name" --gpu $GPU \
        > logs/eval_amort_${name}.log 2>&1 &
    pids+=($!)
    GPU=$(( (GPU + 1) % 8 ))
    if [ $GPU -eq 0 ]; then
        for p in "${pids[@]}"; do wait $p 2>/dev/null; done
        pids=()
    fi
done
for p in "${pids[@]}"; do wait $p 2>/dev/null; done
echo "All done"
