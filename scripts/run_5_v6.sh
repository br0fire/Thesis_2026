#!/bin/bash
# v6: p-value based plateau detection (W=40, p=0.05) on the 5 experiments
# already completed in v5. Direct v5 vs v6 comparison on identical setups.
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
    local OUTDIR="${NFS3}/reinforce_${NAME}_v6"
    local LOGFILE="logs/reinforce_${NAME}_v6.log"
    local MASK_ARG=""
    [ -n "$MASK" ] && [ -f "$MASK" ] && MASK_ARG="--mask $MASK"
    echo "[$(date)] Starting ${NAME}_v6 on GPU ${GPU}"
    nohup $PYTHON -u generation/reinforce_search.py \
        --source_prompt "$SRC" --target_prompt "$TGT" --seg_prompt "$SEG" \
        $MASK_ARG --output_dir "$OUTDIR" --gpu $GPU \
        --n_bits 14 --batch_size 8 --top_k 10 --log_interval 10 \
        > "$LOGFILE" 2>&1 &
    echo $! > "/tmp/reinforce_${NAME}_v6.pid"
}

launch 2 "sunflower_lavender" \
    "A high-resolution landscape photo of a vast sunflower field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "A high-resolution landscape photo of a vast lavender field stretching to the horizon, under a deep blue sky with scattered white cumulus clouds, a single dirt path winding through the flowers, rolling green hills in the far background, golden hour sunlight casting long shadows, 4k, nature photography style." \
    "sunflower" "${MASKS}/background_mask_flux_sunflower_lavender.npy"

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
for NAME in sunflower_lavender penguin_flamingo violin_guitar bgrich_teapot_globe bgrich_typewriter_laptop; do
    PID=$(cat /tmp/reinforce_${NAME}_v6.pid 2>/dev/null)
    [ -n "$PID" ] && { wait $PID 2>/dev/null || true; echo "[$(date)] ${NAME}_v6 done"; }
done
