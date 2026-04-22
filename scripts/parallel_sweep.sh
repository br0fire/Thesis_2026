#!/bin/bash
# Parallel hyperparameter sweep: 8 configs on 8 GPUs for ONE prompt pair.
# Much faster than sequential sweep_reinforce.py (~60 min vs ~6 hours).
#
# Each config runs the full reinforce_search.py pipeline (FLUX + SigLIP 2 +
# GDino+SAM segmentation) with fixed num_episodes=80 and all early-stop
# conditions disabled, so the 8 curves are directly comparable.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

# Required env vars (set by overnight_sweeps.sh):
#   EXPERIMENT, SRC_PROMPT, TGT_PROMPT, SEG_PROMPT
# Optional: NUM_EPISODES (default 80)
: "${EXPERIMENT:?EXPERIMENT env var required}"
: "${SRC_PROMPT:?SRC_PROMPT env var required}"
: "${TGT_PROMPT:?TGT_PROMPT env var required}"
: "${SEG_PROMPT:?SEG_PROMPT env var required}"
NUM_EPISODES="${NUM_EPISODES:-80}"

SWEEP_DIR="analysis/reinforce_analysis/sweep_${EXPERIMENT}"
CONFIGS_DIR="${SWEEP_DIR}/configs"
mkdir -p "$CONFIGS_DIR"

launch() {
    local GPU=$1
    local NAME=$2
    local LR=$3
    local ALPHA=$4
    local ENT=$5
    local OUTDIR="${CONFIGS_DIR}/${NAME}"
    local LOGFILE="logs/sweep_parallel_${EXPERIMENT}_${NAME}.log"
    mkdir -p "$OUTDIR"

    echo "[$(date)] GPU ${GPU}: ${NAME} (lr=${LR} alpha=${ALPHA} ent=${ENT})"
    nohup $PYTHON -u generation/reinforce_search.py \
        --source_prompt "$SRC_PROMPT" \
        --target_prompt "$TGT_PROMPT" \
        --seg_prompt "$SEG_PROMPT" \
        --output_dir "$OUTDIR" \
        --gpu "$GPU" \
        --n_bits 14 --batch_size 8 --top_k 10 --log_interval 10 \
        --num_episodes $NUM_EPISODES \
        --min_episodes 0 \
        --plateau_window 0 \
        --entropy_stop 0 \
        --lr "$LR" \
        --alpha "$ALPHA" \
        --entropy_coeff "$ENT" \
        > "$LOGFILE" 2>&1 &
    echo $! > "/tmp/sweep_parallel_${NAME}.pid"
}

echo "============================================"
echo "  Parallel sweep: ${EXPERIMENT}"
echo "  8 configs on 8 GPUs, ${NUM_EPISODES} episodes each"
echo "  $(date)"
echo "============================================"

# 8 configurations spanning the key hyperparameters
launch 0 "default"      0.10 0.5 0.05
launch 1 "alpha_low"    0.10 0.3 0.05
launch 2 "alpha_high"   0.10 0.7 0.05
launch 3 "lr_low"       0.05 0.5 0.05
launch 4 "lr_high"      0.20 0.5 0.05
launch 5 "no_entropy"   0.10 0.5 0.00
launch 6 "high_entropy" 0.10 0.5 0.10
launch 7 "combined"     0.20 0.3 0.10

echo ""
echo "Launched all 8. Waiting..."
for NAME in default alpha_low alpha_high lr_low lr_high no_entropy high_entropy combined; do
    PID=$(cat /tmp/sweep_parallel_${NAME}.pid 2>/dev/null)
    [ -n "$PID" ] && { wait $PID 2>/dev/null || true; echo "[$(date)] ${NAME} finished"; }
done

echo ""
echo "============================================"
echo "  Configs done, running aggregator..."
echo "============================================"
$PYTHON analysis/parallel_sweep_aggregate.py --sweep_dir "$SWEEP_DIR" --gpu 0 \
    > logs/sweep_parallel_${EXPERIMENT}_aggregate.log 2>&1

echo ""
echo "============================================"
echo "  All done at $(date)"
echo "  Results in $SWEEP_DIR"
echo "============================================"
