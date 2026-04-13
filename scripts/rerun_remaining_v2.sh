#!/bin/bash
# Run v2 (fixed reward) for the remaining 9 experiments that weren't rerun.
# 8 launch in parallel on GPUs 0-7. The 9th (sail_pirate_v2) queues.
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

NFS3=/home/jovyan/shares/SR006.nfs3/svgrozny
MASKS=metrics/masks
# num_episodes=300, entropy_stop=0.5, plateau_patience=100 (all in code defaults now)
BATCH_SIZE=8
TOP_K=10
SUFFIX="_v2"

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
    # All hyperparameters (num_episodes=300, plateau_patience=100, entropy_stop=0.5,
    # alpha=0.3, entropy_coeff=0.05, reward_type=relative, normalize_advantages=on)
    # are now defaults in reinforce_search.py
    nohup $PYTHON -u generation/reinforce_search.py \
        --source_prompt "$SRC" \
        --target_prompt "$TGT" \
        --seg_prompt "$SEG" \
        $MASK_ARG \
        --output_dir "$OUTDIR" \
        --gpu $GPU \
        --n_bits 14 \
        --batch_size $BATCH_SIZE \
        --top_k $TOP_K \
        --log_interval 10 \
        > "$LOGFILE" 2>&1 &
    echo $! > "/tmp/reinforce_${NAME}${SUFFIX}.pid"
    echo "  PID=$(cat /tmp/reinforce_${NAME}${SUFFIX}.pid)"
}

echo "============================================"
echo "  REINFORCE v2 remaining (9 experiments)"
echo "  $(date)"
echo "============================================"

# ── Wave A: 8 concurrent experiments on GPUs 0-7 ──

run_reinforce 0 "penguin_flamingo" \
    "A high-resolution wildlife photo of an emperor penguin standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
    "A high-resolution wildlife photo of a pink flamingo standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
    "penguin" \
    "${MASKS}/background_mask_flux_penguin_flamingo.npy"

run_reinforce 1 "cake_books" \
    "A high-resolution photo of a white birthday cake with colorful candles on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, food photography style." \
    "A high-resolution photo of a tall stack of old leather-bound books on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, still life photography style." \
    "cake" \
    "${MASKS}/background_mask_flux_cake_books.npy"

run_reinforce 2 "lighthouse_castle" \
    "A high-resolution photo of a tall white-and-red lighthouse standing on a dramatic rocky cliff edge, crashing ocean waves below, a winding gravel path leading up to it, wild grass and wildflowers on the hillside, a vivid orange and purple sunset sky with wispy clouds, 4k, landscape photography style." \
    "A high-resolution photo of a medieval stone castle with towers and battlements standing on a dramatic rocky cliff edge, crashing ocean waves below, a winding gravel path leading up to it, wild grass and wildflowers on the hillside, a vivid orange and purple sunset sky with wispy clouds, 4k, landscape photography style." \
    "lighthouse" \
    "${MASKS}/background_mask_flux_lighthouse_castle.npy"

run_reinforce 3 "violin_guitar" \
    "A high-resolution photo of a polished wooden violin resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
    "A high-resolution photo of a sleek black electric guitar resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
    "violin" \
    "${MASKS}/background_mask_flux_violin_guitar.npy"

run_reinforce 4 "horse" \
    "a horse on the grass" \
    "a robot horse on the grass" \
    "horse" \
    "${MASKS}/background_mask_flux_horse.npy"

run_reinforce 5 "room" \
    "A high-resolution photo of a modern living room with a large gray fabric sofa, a wooden coffee table in front of it, a soft beige rug, and a floor-to-ceiling window letting in natural daylight. Minimalist interior design, clean lines, neutral color palette, cozy atmosphere, realistic lighting, 4k, interior photography style." \
    "A high-resolution photo of a modern living room with a large gray fabric sofa, a rectangular glass aquarium filled with clear water, colorful fish, and aquatic plants placed in front of it, a soft beige rug, and a floor-to-ceiling window letting in natural daylight. Minimalist interior design, clean lines, neutral color palette, cozy atmosphere, realistic lighting, 4k, interior photography style." \
    "wooden coffee table" \
    "${MASKS}/background_mask_flux_room.npy"

run_reinforce 6 "snow_volcano" \
    "A high-resolution landscape photo of a majestic snow-covered mountain peak towering above a dense pine forest, a frozen lake in the foreground reflecting the mountains, fresh powder snow on the ground, clear winter sky with a few high cirrus clouds, crisp cold atmosphere, 4k, landscape photography style." \
    "A high-resolution landscape photo of an active volcanic mountain with glowing lava streams towering above a dense pine forest, a frozen lake in the foreground reflecting the mountains, fresh powder snow on the ground, clear winter sky with a few high cirrus clouds, crisp cold atmosphere, 4k, landscape photography style." \
    "mountain peak" \
    ""

run_reinforce 7 "butterfly_hummingbird" \
    "A high-resolution macro photo of a monarch butterfly with orange and black wings perched on a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
    "A high-resolution macro photo of a tiny iridescent green hummingbird hovering near a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
    "butterfly" \
    ""

echo ""
echo "Wave A launched (8 experiments on GPUs 0-7). Waiting for one to finish..."

# Wait for any of the 8 to finish, then launch the 9th
WAVE_A_NAMES="penguin_flamingo cake_books lighthouse_castle violin_guitar horse room snow_volcano butterfly_hummingbird"
while true; do
    for NAME in $WAVE_A_NAMES; do
        if [ -f "$NFS3/reinforce_${NAME}${SUFFIX}/reinforce_result.pt" ]; then
            echo "[$(date)] ${NAME}${SUFFIX} finished — launching sail_pirate_v2"
            # Figure out which GPU was freed by matching name to index above
            case $NAME in
                penguin_flamingo) FREE_GPU=0 ;;
                cake_books) FREE_GPU=1 ;;
                lighthouse_castle) FREE_GPU=2 ;;
                violin_guitar) FREE_GPU=3 ;;
                horse) FREE_GPU=4 ;;
                room) FREE_GPU=5 ;;
                snow_volcano) FREE_GPU=6 ;;
                butterfly_hummingbird) FREE_GPU=7 ;;
            esac
            run_reinforce $FREE_GPU "sail_pirate" \
                "A high-resolution photo of a small white sailboat with a single mast gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style." \
                "A high-resolution photo of a large old wooden pirate ship with tattered black sails and a skull flag gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style." \
                "sailboat" \
                ""
            break 2
        fi
    done
    sleep 60
done

# Wait for all 9 to finish
ALL_NAMES="$WAVE_A_NAMES sail_pirate"
for NAME in $ALL_NAMES; do
    PID=$(cat /tmp/reinforce_${NAME}${SUFFIX}.pid 2>/dev/null)
    if [ -n "$PID" ]; then
        wait $PID 2>/dev/null || true
        echo "[$(date)] ${NAME}${SUFFIX} finished"
    fi
done

echo ""
echo "============================================"
echo "  All v2 remaining experiments DONE at $(date)"
echo "============================================"
