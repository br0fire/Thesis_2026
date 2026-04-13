#!/bin/bash
# Re-run all 16 experiments with the simplified reward code (v3):
#   - SigLIP 2 SO400M + unclamped delta fg formula
#   - min_episodes=200, plateau_patience=150, alpha=0.3, normalize_advantages=on
# Two waves of 8 on 8 GPUs.
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

NFS3=/home/jovyan/shares/SR006.nfs3/svgrozny
MASKS=metrics/masks
SUFFIX="_v3"

launch() {
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
        --batch_size 8 \
        --top_k 10 \
        --log_interval 10 \
        > "$LOGFILE" 2>&1 &
    echo $! > "/tmp/reinforce_${NAME}${SUFFIX}.pid"
}

wait_for_wave() {
    local NAMES="$1"
    for NAME in $NAMES; do
        PID=$(cat /tmp/reinforce_${NAME}${SUFFIX}.pid 2>/dev/null)
        if [ -n "$PID" ]; then
            wait $PID 2>/dev/null || true
            echo "[$(date)] ${NAME}${SUFFIX} finished"
        fi
    done
}

echo "============================================"
echo "  REINFORCE v3: all 16 experiments"
echo "  $(date)"
echo "============================================"

# ── Wave A (GPUs 0-7): 8 experiments ──

launch 0 "catdog" \
    "a tabby cat walking on stone pavement, photo" \
    "a golden retriever dog walking on stone pavement, photo" \
    "cat" \
    ""

launch 1 "car_taxi" \
    "A high-resolution photo of a red sports car parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
    "A high-resolution photo of a yellow taxi cab parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
    "car" \
    "${MASKS}/background_mask_flux_car_taxi.npy"

launch 2 "sunflower_lavender" \
    "A high-resolution landscape photo of a vast sunflower field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "A high-resolution landscape photo of a vast lavender field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "sunflower" \
    "${MASKS}/background_mask_flux_sunflower_lavender.npy"

launch 3 "chair_throne" \
    "A high-resolution photo of a simple wooden chair placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
    "A high-resolution photo of an ornate golden throne with red velvet cushions placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
    "chair" \
    "${MASKS}/background_mask_flux_chair_throne.npy"

launch 4 "penguin_flamingo" \
    "A high-resolution wildlife photo of an emperor penguin standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
    "A high-resolution wildlife photo of a pink flamingo standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
    "penguin" \
    "${MASKS}/background_mask_flux_penguin_flamingo.npy"

launch 5 "cake_books" \
    "A high-resolution photo of a white birthday cake with colorful candles on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, food photography style." \
    "A high-resolution photo of a tall stack of old leather-bound books on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, still life photography style." \
    "cake" \
    "${MASKS}/background_mask_flux_cake_books.npy"

launch 6 "lighthouse_castle" \
    "A high-resolution photo of a tall white-and-red lighthouse standing on a dramatic rocky cliff edge, crashing ocean waves below, a winding gravel path leading up to it, wild grass and wildflowers on the hillside, a vivid orange and purple sunset sky with wispy clouds, 4k, landscape photography style." \
    "A high-resolution photo of a medieval stone castle with towers and battlements standing on a dramatic rocky cliff edge, crashing ocean waves below, a winding gravel path leading up to it, wild grass and wildflowers on the hillside, a vivid orange and purple sunset sky with wispy clouds, 4k, landscape photography style." \
    "lighthouse" \
    "${MASKS}/background_mask_flux_lighthouse_castle.npy"

launch 7 "violin_guitar" \
    "A high-resolution photo of a polished wooden violin resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
    "A high-resolution photo of a sleek black electric guitar resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
    "violin" \
    "${MASKS}/background_mask_flux_violin_guitar.npy"

echo ""
echo "Wave A launched (8 experiments). Waiting..."
wait_for_wave "catdog car_taxi sunflower_lavender chair_throne penguin_flamingo cake_books lighthouse_castle violin_guitar"

echo ""
echo "============================================"
echo "  Wave B: remaining 8 experiments"
echo "  $(date)"
echo "============================================"

# ── Wave B (GPUs 0-7): 8 experiments ──

launch 0 "horse" \
    "a horse on the grass" \
    "a robot horse on the grass" \
    "horse" \
    "${MASKS}/background_mask_flux_horse.npy"

launch 1 "room" \
    "A high-resolution photo of a modern living room with a large gray fabric sofa, a wooden coffee table in front of it, a soft beige rug, and a floor-to-ceiling window letting in natural daylight. Minimalist interior design, clean lines, neutral color palette, cozy atmosphere, realistic lighting, 4k, interior photography style." \
    "A high-resolution photo of a modern living room with a large gray fabric sofa, a rectangular glass aquarium filled with clear water, colorful fish, and aquatic plants placed in front of it, a soft beige rug, and a floor-to-ceiling window letting in natural daylight. Minimalist interior design, clean lines, neutral color palette, cozy atmosphere, realistic lighting, 4k, interior photography style." \
    "wooden coffee table" \
    "${MASKS}/background_mask_flux_room.npy"

launch 2 "snow_volcano" \
    "A high-resolution landscape photo of a majestic snow-covered mountain peak towering above a dense pine forest, a frozen lake in the foreground reflecting the mountains, fresh powder snow on the ground, clear winter sky with a few high cirrus clouds, crisp cold atmosphere, 4k, landscape photography style." \
    "A high-resolution landscape photo of an active volcanic mountain with glowing lava streams towering above a dense pine forest, a frozen lake in the foreground reflecting the mountains, fresh powder snow on the ground, clear winter sky with a few high cirrus clouds, crisp cold atmosphere, 4k, landscape photography style." \
    "mountain peak" \
    ""

launch 3 "butterfly_hummingbird" \
    "A high-resolution macro photo of a monarch butterfly with orange and black wings perched on a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
    "A high-resolution macro photo of a tiny iridescent green hummingbird hovering near a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
    "butterfly" \
    ""

launch 4 "sail_pirate" \
    "A high-resolution photo of a small white sailboat with a single mast gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style." \
    "A high-resolution photo of a large old wooden pirate ship with tattered black sails and a skull flag gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style." \
    "sailboat" \
    ""

launch 5 "bgrich_teapot_globe" \
    "A high-resolution photo of a porcelain teapot on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography." \
    "A high-resolution photo of an antique terrestrial globe on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography." \
    "teapot" \
    ""

launch 6 "bgrich_candle_crystal" \
    "A high-resolution photo of a tall beeswax candle with a flickering flame on a cluttered wooden workbench in an alchemist's workshop, surrounded by bubbling glass flasks of colored potions, dusty spellbooks, dried herbs hanging from the ceiling beams, a brass astrolabe, a human skull, raven feathers, a stone mortar and pestle, rolled yellowed scrolls, glowing amber jars of unknown substances, iron tongs, and a cauldron simmering on coals, moody chiaroscuro lighting with deep shadows, 4k, atmospheric fantasy photography." \
    "A high-resolution photo of a glowing purple crystal cluster on a cluttered wooden workbench in an alchemist's workshop, surrounded by bubbling glass flasks of colored potions, dusty spellbooks, dried herbs hanging from the ceiling beams, a brass astrolabe, a human skull, raven feathers, a stone mortar and pestle, rolled yellowed scrolls, glowing amber jars of unknown substances, iron tongs, and a cauldron simmering on coals, moody chiaroscuro lighting with deep shadows, 4k, atmospheric fantasy photography." \
    "candle" \
    ""

launch 7 "bgrich_typewriter_laptop" \
    "A high-resolution photo of a vintage black typewriter on a cluttered wooden writer's desk, surrounded by towering stacks of manuscript papers with handwritten edits, three half-empty coffee mugs, a jar of ballpoint pens and pencils, pink and yellow sticky notes with scribbled reminders, a silver framed family photograph, a small potted succulent plant, a bronze gooseneck desk lamp, crumpled paper balls on the floor, a tall bookshelf filled with novels in the background, a wall calendar with red circled dates, and a brass letter opener, warm afternoon sunlight through a window, 4k, lifestyle photography." \
    "A high-resolution photo of a sleek silver open laptop computer on a cluttered wooden writer's desk, surrounded by towering stacks of manuscript papers with handwritten edits, three half-empty coffee mugs, a jar of ballpoint pens and pencils, pink and yellow sticky notes with scribbled reminders, a silver framed family photograph, a small potted succulent plant, a bronze gooseneck desk lamp, crumpled paper balls on the floor, a tall bookshelf filled with novels in the background, a wall calendar with red circled dates, and a brass letter opener, warm afternoon sunlight through a window, 4k, lifestyle photography." \
    "typewriter" \
    ""

echo ""
echo "Wave B launched. Waiting..."
wait_for_wave "horse room snow_volcano butterfly_hummingbird sail_pirate bgrich_teapot_globe bgrich_candle_crystal bgrich_typewriter_laptop"

echo ""
echo "============================================"
echo "  All v3 experiments DONE at $(date)"
echo "============================================"
