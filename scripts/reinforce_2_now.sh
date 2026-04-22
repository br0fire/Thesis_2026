#!/bin/bash
# Launch 2 REINFORCE runs on idle GPUs 6 and 7 right now for new bgrich pairs.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project
source scripts/new_bgrich_prompts.sh

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
mkdir -p logs

launch_reinforce() {
    local GPU=$1
    local NAME=$2
    local OUTDIR="analysis/reinforce_analysis/new_bgrich/${NAME}"
    mkdir -p "$OUTDIR"
    echo "[$(date)] GPU $GPU: REINFORCE $NAME"
    nohup $PYTHON -u generation/reinforce_search.py \
        --source_prompt "${NEW_SRC[$NAME]}" \
        --target_prompt "${NEW_TGT[$NAME]}" \
        --seg_prompt "${NEW_SEG[$NAME]}" \
        --output_dir "$OUTDIR" \
        --gpu $GPU \
        --n_bits 14 --batch_size 8 --top_k 10 --log_interval 10 \
        --num_episodes 80 --min_episodes 80 --plateau_window 0 --entropy_stop 0 \
        --lr 0.10 --alpha 0.7 --entropy_coeff 0.05 \
        > logs/reinforce_new_${NAME}.log 2>&1 &
    echo $! > /tmp/reinforce_new_${NAME}.pid
}

launch_reinforce 6 "bgrich_globe_orrery"
launch_reinforce 7 "bgrich_telescope_microscope"
echo "Launched 2 REINFORCE on GPUs 6-7"
