#!/bin/bash
# Re-run the 4 experiments that collapsed to source-only with the fixed reward.
# Uses new "relative" reward type: fg·tgt - fg·src (instead of fg · normalize(tgt-src))
# Removes the clamp on negative fg_clip, increases entropy bonus, rebalances alpha.
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

NFS3=/home/jovyan/shares/SR006.nfs3/svgrozny
MASKS=metrics/masks
NUM_EPISODES=500
BATCH_SIZE=8
TOP_K=10
SUFFIX="_v2"  # new runs go to reinforce_${NAME}_v2

run_reinforce() {
    local GPU=$1
    local NAME=$2
    local SRC="$3"
    local TGT="$4"
    local SEG="$5"
    local MASK="$6"
    local OUTDIR="${NFS3}/reinforce_${NAME}${SUFFIX}"
    local LOGFILE="logs/reinforce_${NAME}${SUFFIX}.log"

    local MASK_ARG=""
    if [ -n "$MASK" ] && [ -f "$MASK" ]; then
        MASK_ARG="--mask $MASK"
    fi

    echo "[$(date)] Starting ${NAME}${SUFFIX} on GPU ${GPU} → ${LOGFILE}"
    nohup $PYTHON -u generation/reinforce_search.py \
        --source_prompt "$SRC" \
        --target_prompt "$TGT" \
        --seg_prompt "$SEG" \
        $MASK_ARG \
        --output_dir "$OUTDIR" \
        --gpu $GPU \
        --n_bits 14 \
        --num_episodes $NUM_EPISODES \
        --batch_size $BATCH_SIZE \
        --top_k $TOP_K \
        --log_interval 10 \
        --alpha 0.3 \
        --entropy_coeff 0.05 \
        --normalize_advantages \
        > "$LOGFILE" 2>&1 &
    echo $! > "/tmp/reinforce_${NAME}${SUFFIX}.pid"
    echo "  PID=$(cat /tmp/reinforce_${NAME}${SUFFIX}.pid)"
}

echo "============================================"
echo "  REINFORCE re-runs with fixed reward (v2)"
echo "  $(date)"
echo "============================================"

# GPUs 1-5 are running wave 2 (horse, room, snow_volcano, butterfly_hummingbird, sail_pirate)
# Use idle GPUs 0, 6, 7 for 3 reruns; 4th waits

run_reinforce 0 "catdog" \
    "a tabby cat walking on stone pavement, photo" \
    "a golden retriever dog walking on stone pavement, photo" \
    "cat" \
    ""

run_reinforce 6 "car_taxi" \
    "A high-resolution photo of a red sports car parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
    "A high-resolution photo of a yellow taxi cab parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
    "car" \
    "${MASKS}/background_mask_flux_car_taxi.npy"

run_reinforce 7 "sunflower_lavender" \
    "A high-resolution landscape photo of a vast sunflower field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "A high-resolution landscape photo of a vast lavender field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "sunflower" \
    "${MASKS}/background_mask_flux_sunflower_lavender.npy"

echo ""
echo "3 experiments launched on GPUs 0, 6, 7. Waiting for one to finish..."

# Wait for catdog (GPU 0) — the shortest if we use the same 500 episodes it will take same time
# Just wait for any of the 3 to free up
for NAME in catdog car_taxi sunflower_lavender; do
    PID=$(cat /tmp/reinforce_${NAME}${SUFFIX}.pid 2>/dev/null)
    if [ -n "$PID" ]; then
        wait $PID 2>/dev/null || true
        echo "[$(date)] ${NAME}${SUFFIX} finished (exit=$?)"
        break
    fi
done

# Run the 4th (chair_throne) on the first freed-up GPU
FREE_GPU=0  # catdog will likely finish first on GPU 0
run_reinforce $FREE_GPU "chair_throne" \
    "A high-resolution photo of a simple wooden chair placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
    "A high-resolution photo of an ornate golden throne with red velvet cushions placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
    "chair" \
    "${MASKS}/background_mask_flux_chair_throne.npy"

# Wait for remaining 3
for NAME in car_taxi sunflower_lavender chair_throne; do
    PID=$(cat /tmp/reinforce_${NAME}${SUFFIX}.pid 2>/dev/null)
    if [ -n "$PID" ]; then
        wait $PID 2>/dev/null || true
        echo "[$(date)] ${NAME}${SUFFIX} finished (exit=$?)"
    fi
done

echo ""
echo "============================================"
echo "  All v2 experiments DONE at $(date)"
echo "============================================"
