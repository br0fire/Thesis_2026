#!/bin/bash
# Re-run the 4 formerly-collapsed experiments with SigLIP2 + relative reward
# (they previously only had CLIP+relative results under *_v2clip).
# Also run one control experiment (violin_guitar with CLIP+relative) so we can
# cleanly A/B compare CLIP vs SigLIP2 with the same relative reward.
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

NFS3=/home/jovyan/shares/SR006.nfs3/svgrozny
MASKS=metrics/masks

# All hyperparameters (num_episodes=300, min_episodes=200, plateau_patience=150,
# entropy_stop=0.5, alpha=0.3, reward_type=relative, normalize_advantages=on) are defaults.

run_siglip2() {
    local GPU=$1
    local NAME=$2
    local SUFFIX="$3"  # "_v2" for SigLIP2, "_v2clip" for CLIP+relative
    local SRC="$4"
    local TGT="$5"
    local SEG="$6"
    local MASK="$7"
    local OUTDIR="${NFS3}/reinforce_${NAME}${SUFFIX}"
    local LOGFILE="logs/reinforce_${NAME}${SUFFIX}.log"

    local MASK_ARG=""
    if [ -n "$MASK" ] && [ -f "$MASK" ]; then
        MASK_ARG="--mask $MASK"
    fi

    # Default: SigLIP2. Override only when suffix is _v2clip.
    local VISION_ARG=""
    if [ "$SUFFIX" = "_v2clip" ]; then
        VISION_ARG="--vision_model openai/clip-vit-base-patch32"
    fi

    echo "[$(date)] Starting ${NAME}${SUFFIX} on GPU ${GPU} → ${LOGFILE}"
    nohup $PYTHON -u generation/reinforce_search.py \
        --source_prompt "$SRC" \
        --target_prompt "$TGT" \
        --seg_prompt "$SEG" \
        $MASK_ARG \
        $VISION_ARG \
        --output_dir "$OUTDIR" \
        --gpu $GPU \
        --n_bits 14 \
        --batch_size 8 \
        --top_k 10 \
        --log_interval 10 \
        > "$LOGFILE" 2>&1 &
    echo $! > "/tmp/reinforce_${NAME}${SUFFIX}.pid"
    echo "  PID=$(cat /tmp/reinforce_${NAME}${SUFFIX}.pid)"
}

echo "============================================"
echo "  SigLIP2 missing experiments + 1 control"
echo "  $(date)"
echo "============================================"

# GPUs 0-3: 4 SigLIP2 reruns of the previously-collapsed experiments
run_siglip2 0 "catdog" "_v2" \
    "a tabby cat walking on stone pavement, photo" \
    "a golden retriever dog walking on stone pavement, photo" \
    "cat" \
    ""

run_siglip2 1 "car_taxi" "_v2" \
    "A high-resolution photo of a red sports car parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
    "A high-resolution photo of a yellow taxi cab parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
    "car" \
    "${MASKS}/background_mask_flux_car_taxi.npy"

run_siglip2 2 "sunflower_lavender" "_v2" \
    "A high-resolution landscape photo of a vast sunflower field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "A high-resolution landscape photo of a vast lavender field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "sunflower" \
    "${MASKS}/background_mask_flux_sunflower_lavender.npy"

run_siglip2 3 "chair_throne" "_v2" \
    "A high-resolution photo of a simple wooden chair placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
    "A high-resolution photo of an ornate golden throne with red velvet cushions placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
    "chair" \
    "${MASKS}/background_mask_flux_chair_throne.npy"

# GPU 4: Control — violin_guitar with CLIP + relative reward.
# We already have violin_guitar (CLIP+delta, v1) and violin_guitar_v2 (SigLIP2+relative).
# Adding CLIP+relative completes the 3-way comparison so we can attribute improvement.
run_siglip2 4 "violin_guitar" "_v2clip" \
    "A high-resolution photo of a polished wooden violin resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
    "A high-resolution photo of a sleek black electric guitar resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
    "violin" \
    "${MASKS}/background_mask_flux_violin_guitar.npy"

echo ""
echo "Launched 5 experiments (4 SigLIP2 + 1 CLIP control). Waiting..."

for PAIR in "catdog:_v2" "car_taxi:_v2" "sunflower_lavender:_v2" "chair_throne:_v2" "violin_guitar:_v2clip"; do
    NAME="${PAIR%:*}"
    SUFFIX="${PAIR#*:}"
    PID=$(cat /tmp/reinforce_${NAME}${SUFFIX}.pid 2>/dev/null)
    if [ -n "$PID" ]; then
        wait $PID 2>/dev/null || true
        echo "[$(date)] ${NAME}${SUFFIX} finished"
    fi
done

echo ""
echo "============================================"
echo "  All 5 experiments DONE at $(date)"
echo "============================================"
