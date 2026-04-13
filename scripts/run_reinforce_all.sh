#!/bin/bash
# Launch REINFORCE search for all experiments in parallel (one per GPU).
# GPU 0 is already running catdog, so we use GPUs 1-7 for the first 7,
# then wait and reuse GPUs for the remaining 4.
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

run_reinforce() {
    local GPU=$1
    local NAME=$2
    local SRC="$3"
    local TGT="$4"
    local SEG="$5"
    local MASK="$6"
    local OUTDIR="${NFS3}/reinforce_${NAME}"
    local LOGFILE="logs/reinforce_${NAME}.log"

    local MASK_ARG=""
    if [ -n "$MASK" ] && [ -f "$MASK" ]; then
        MASK_ARG="--mask $MASK"
    fi

    echo "[$(date)] Starting ${NAME} on GPU ${GPU} → ${LOGFILE}"
    nohup $PYTHON generation/reinforce_search.py \
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
        --normalize_advantages \
        > "$LOGFILE" 2>&1 &
    echo $! > "/tmp/reinforce_${NAME}.pid"
    echo "  PID=$(cat /tmp/reinforce_${NAME}.pid)"
}

echo "============================================"
echo "  REINFORCE: launching all experiments"
echo "  $(date)"
echo "============================================"

# ── Wave 1: GPUs 1-7 (7 experiments) ──

run_reinforce 1 "car_taxi" \
    "A high-resolution photo of a red sports car parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
    "A high-resolution photo of a yellow taxi cab parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
    "car" \
    "${MASKS}/background_mask_flux_car_taxi.npy"

run_reinforce 2 "sunflower_lavender" \
    "A high-resolution landscape photo of a vast sunflower field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "A high-resolution landscape photo of a vast lavender field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "sunflower" \
    "${MASKS}/background_mask_flux_sunflower_lavender.npy"

run_reinforce 3 "chair_throne" \
    "A high-resolution photo of a simple wooden chair placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
    "A high-resolution photo of an ornate golden throne with red velvet cushions placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
    "chair" \
    "${MASKS}/background_mask_flux_chair_throne.npy"

run_reinforce 4 "penguin_flamingo" \
    "A high-resolution wildlife photo of an emperor penguin standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
    "A high-resolution wildlife photo of a pink flamingo standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
    "penguin" \
    "${MASKS}/background_mask_flux_penguin_flamingo.npy"

run_reinforce 5 "cake_books" \
    "A high-resolution photo of a white birthday cake with colorful candles on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, food photography style." \
    "A high-resolution photo of a tall stack of old leather-bound books on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, still life photography style." \
    "cake" \
    "${MASKS}/background_mask_flux_cake_books.npy"

run_reinforce 6 "lighthouse_castle" \
    "A high-resolution photo of a tall white-and-red lighthouse standing on a dramatic rocky cliff edge, crashing ocean waves below, a winding gravel path leading up to it, wild grass and wildflowers on the hillside, a vivid orange and purple sunset sky with wispy clouds, 4k, landscape photography style." \
    "A high-resolution photo of a medieval stone castle with towers and battlements standing on a dramatic rocky cliff edge, crashing ocean waves below, a winding gravel path leading up to it, wild grass and wildflowers on the hillside, a vivid orange and purple sunset sky with wispy clouds, 4k, landscape photography style." \
    "lighthouse" \
    "${MASKS}/background_mask_flux_lighthouse_castle.npy"

run_reinforce 7 "violin_guitar" \
    "A high-resolution photo of a polished wooden violin resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
    "A high-resolution photo of a sleek black electric guitar resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
    "violin" \
    "${MASKS}/background_mask_flux_violin_guitar.npy"

echo ""
echo "Wave 1 launched (7 experiments on GPUs 1-7). Waiting for completion..."

# Wait for all wave 1 jobs
for NAME in car_taxi sunflower_lavender chair_throne penguin_flamingo cake_books lighthouse_castle violin_guitar; do
    PID=$(cat /tmp/reinforce_${NAME}.pid 2>/dev/null)
    if [ -n "$PID" ]; then
        wait $PID 2>/dev/null || true
        echo "[$(date)] ${NAME} finished (exit=$?)"
    fi
done

echo ""
echo "============================================"
echo "  Wave 2: remaining 4 experiments"
echo "  $(date)"
echo "============================================"

# ── Wave 2: reuse GPUs 1-4 for remaining experiments ──

run_reinforce 1 "horse" \
    "a horse on the grass" \
    "a robot horse on the grass" \
    "horse" \
    "${MASKS}/background_mask_flux_horse.npy"

run_reinforce 2 "room" \
    "A high-resolution photo of a modern living room with a large gray fabric sofa, a wooden coffee table in front of it, a soft beige rug, and a floor-to-ceiling window letting in natural daylight. Minimalist interior design, clean lines, neutral color palette, cozy atmosphere, realistic lighting, 4k, interior photography style." \
    "A high-resolution photo of a modern living room with a large gray fabric sofa, a rectangular glass aquarium filled with clear water, colorful fish, and aquatic plants placed in front of it, a soft beige rug, and a floor-to-ceiling window letting in natural daylight. Minimalist interior design, clean lines, neutral color palette, cozy atmosphere, realistic lighting, 4k, interior photography style." \
    "wooden coffee table" \
    "${MASKS}/background_mask_flux_room.npy"

run_reinforce 3 "snow_volcano" \
    "A high-resolution landscape photo of a majestic snow-covered mountain peak towering above a dense pine forest, a frozen lake in the foreground reflecting the mountains, fresh powder snow on the ground, clear winter sky with a few high cirrus clouds, crisp cold atmosphere, 4k, landscape photography style." \
    "A high-resolution landscape photo of an active volcanic mountain with glowing lava streams towering above a dense pine forest, a frozen lake in the foreground reflecting the mountains, fresh powder snow on the ground, clear winter sky with a few high cirrus clouds, crisp cold atmosphere, 4k, landscape photography style." \
    "mountain peak" \
    ""

run_reinforce 4 "butterfly_hummingbird" \
    "A high-resolution macro photo of a monarch butterfly with orange and black wings perched on a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
    "A high-resolution macro photo of a tiny iridescent green hummingbird hovering near a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
    "butterfly" \
    ""

# sail_pirate also has no pre-computed mask
run_reinforce 5 "sail_pirate" \
    "A high-resolution photo of a small white sailboat with a single mast gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style." \
    "A high-resolution photo of a large old wooden pirate ship with tattered black sails and a skull flag gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style." \
    "sailboat" \
    ""

echo ""
echo "Wave 2 launched (5 experiments on GPUs 1-5). Waiting..."

for NAME in horse room snow_volcano butterfly_hummingbird sail_pirate; do
    PID=$(cat /tmp/reinforce_${NAME}.pid 2>/dev/null)
    if [ -n "$PID" ]; then
        wait $PID 2>/dev/null || true
        echo "[$(date)] ${NAME} finished (exit=$?)"
    fi
done

echo ""
echo "============================================"
echo "  ALL REINFORCE experiments DONE at $(date)"
echo "============================================"
