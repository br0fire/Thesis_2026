#!/bin/bash
# v4: geometric-mean reward + mean_reward plateau early-stop.
# Same 8 experiments as the previous validation run, so we can compare against
# the 8 unchanged v3 runs.
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

NFS3=/home/jovyan/shares/SR006.nfs3/svgrozny
MASKS=metrics/masks

launch() {
    local GPU=$1
    local NAME=$2
    local SRC="$3"
    local TGT="$4"
    local SEG="$5"
    local MASK="$6"
    local OUTDIR="${NFS3}/reinforce_${NAME}_v4"
    local LOGFILE="logs/reinforce_${NAME}_v4.log"
    local MASK_ARG=""
    [ -n "$MASK" ] && [ -f "$MASK" ] && MASK_ARG="--mask $MASK"
    echo "[$(date)] Starting ${NAME}_v4 on GPU ${GPU} → ${LOGFILE}"
    # Defaults: alpha=0.5 (geometric mean, balanced), num_episodes=300,
    # min_episodes=100, plateau_patience=80, ma_window=30, ma_tolerance=0.002.
    nohup $PYTHON -u generation/reinforce_search.py \
        --source_prompt "$SRC" --target_prompt "$TGT" --seg_prompt "$SEG" \
        $MASK_ARG --output_dir "$OUTDIR" --gpu $GPU \
        --n_bits 14 --batch_size 8 --top_k 10 --log_interval 10 \
        > "$LOGFILE" 2>&1 &
    echo $! > "/tmp/reinforce_${NAME}_v4.pid"
}

echo "============================================"
echo "  REINFORCE v4: geometric mean + reward-plateau stop"
echo "  $(date)"
echo "============================================"

launch 0 "catdog" \
    "a tabby cat walking on stone pavement, photo" \
    "a golden retriever dog walking on stone pavement, photo" \
    "cat" ""

launch 1 "car_taxi" \
    "A high-resolution photo of a red sports car parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
    "A high-resolution photo of a yellow taxi cab parked on a wet cobblestone city street at dusk, with warm streetlamp reflections on the pavement, old European brick buildings with glowing windows on both sides, a few pedestrians with umbrellas in the background, moody cinematic lighting, 4k, street photography style." \
    "car" "${MASKS}/background_mask_flux_car_taxi.npy"

launch 2 "sunflower_lavender" \
    "A high-resolution landscape photo of a vast sunflower field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "A high-resolution landscape photo of a vast lavender field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "sunflower" "${MASKS}/background_mask_flux_sunflower_lavender.npy"

launch 3 "chair_throne" \
    "A high-resolution photo of a simple wooden chair placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
    "A high-resolution photo of an ornate golden throne with red velvet cushions placed in the center of a grand empty hall with polished marble floors, tall arched windows letting in soft diffused daylight, ornate ceiling moldings, dust particles floating in the light beams, minimalist composition, 4k, fine art photography style." \
    "chair" "${MASKS}/background_mask_flux_chair_throne.npy"

launch 4 "penguin_flamingo" \
    "A high-resolution wildlife photo of an emperor penguin standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
    "A high-resolution wildlife photo of a pink flamingo standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
    "penguin" "${MASKS}/background_mask_flux_penguin_flamingo.npy"

launch 5 "violin_guitar" \
    "A high-resolution photo of a polished wooden violin resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
    "A high-resolution photo of a sleek black electric guitar resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
    "violin" "${MASKS}/background_mask_flux_violin_guitar.npy"

launch 6 "bgrich_teapot_globe" \
    "A high-resolution photo of a porcelain teapot on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography." \
    "A high-resolution photo of an antique terrestrial globe on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography." \
    "teapot" ""

launch 7 "bgrich_typewriter_laptop" \
    "A high-resolution photo of a vintage black typewriter on a cluttered wooden writer's desk, surrounded by towering stacks of manuscript papers with handwritten edits, three half-empty coffee mugs, a jar of ballpoint pens and pencils, pink and yellow sticky notes with scribbled reminders, a silver framed family photograph, a small potted succulent plant, a bronze gooseneck desk lamp, crumpled paper balls on the floor, a tall bookshelf filled with novels in the background, a wall calendar with red circled dates, and a brass letter opener, warm afternoon sunlight through a window, 4k, lifestyle photography." \
    "A high-resolution photo of a sleek silver open laptop computer on a cluttered wooden writer's desk, surrounded by towering stacks of manuscript papers with handwritten edits, three half-empty coffee mugs, a jar of ballpoint pens and pencils, pink and yellow sticky notes with scribbled reminders, a silver framed family photograph, a small potted succulent plant, a bronze gooseneck desk lamp, crumpled paper balls on the floor, a tall bookshelf filled with novels in the background, a wall calendar with red circled dates, and a brass letter opener, warm afternoon sunlight through a window, 4k, lifestyle photography." \
    "typewriter" ""

echo ""
echo "Launched 8 v4 experiments on GPUs 0-7. Waiting..."
for NAME in catdog car_taxi sunflower_lavender chair_throne penguin_flamingo violin_guitar bgrich_teapot_globe bgrich_typewriter_laptop; do
    PID=$(cat /tmp/reinforce_${NAME}_v4.pid 2>/dev/null)
    [ -n "$PID" ] && { wait $PID 2>/dev/null || true; echo "[$(date)] ${NAME}_v4 finished"; }
done

echo ""
echo "============================================"
echo "  All 8 v4 DONE at $(date)"
echo "============================================"
