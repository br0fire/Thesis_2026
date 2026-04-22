#!/bin/bash
# Run n_bits=20 experiments on 8 GPUs in parallel with early stopping.
# Tests whether REINFORCE beats random more convincingly in a larger search space.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

OUT_BASE="analysis/reinforce_analysis/nbits20"
mkdir -p logs

launch() {
    local GPU=$1
    local NAME=$2
    local SRC=$3
    local TGT=$4
    local SEG=$5
    local OUTDIR="${OUT_BASE}/${NAME}"
    local LOGFILE="logs/nbits20_${NAME}.log"
    mkdir -p "$OUTDIR"

    echo "[$(date)] GPU ${GPU}: ${NAME}"
    nohup $PYTHON -u generation/reinforce_search.py \
        --source_prompt "$SRC" \
        --target_prompt "$TGT" \
        --seg_prompt "$SEG" \
        --output_dir "$OUTDIR" \
        --gpu "$GPU" \
        --n_bits 20 --batch_size 8 --top_k 10 --log_interval 10 \
        --num_episodes 300 \
        --min_episodes 80 \
        --plateau_window 40 \
        --plateau_pvalue 0.05 \
        --entropy_stop 0.3 \
        --lr 0.10 \
        --alpha 0.7 \
        --entropy_coeff 0.05 \
        > "$LOGFILE" 2>&1 &
    echo $! > "/tmp/nbits20_${NAME}.pid"
}

echo "============================================"
echo "  n_bits=20 experiment (8 GPUs, early stop)"
echo "  $(date)"
echo "============================================"

launch 0 "bgrich_typewriter_laptop" \
"A high-resolution photo of a vintage black typewriter on a cluttered wooden writer's desk, surrounded by towering stacks of manuscript papers with handwritten edits, three half-empty coffee mugs, a jar of ballpoint pens and pencils, pink and yellow sticky notes with scribbled reminders, a silver framed family photograph, a small potted succulent plant, a bronze gooseneck desk lamp, crumpled paper balls on the floor, a tall bookshelf filled with novels in the background, a wall calendar with red circled dates, and a brass letter opener, warm afternoon sunlight through a window, 4k, lifestyle photography." \
"A high-resolution photo of a sleek silver open laptop computer on a cluttered wooden writer's desk, surrounded by towering stacks of manuscript papers with handwritten edits, three half-empty coffee mugs, a jar of ballpoint pens and pencils, pink and yellow sticky notes with scribbled reminders, a silver framed family photograph, a small potted succulent plant, a bronze gooseneck desk lamp, crumpled paper balls on the floor, a tall bookshelf filled with novels in the background, a wall calendar with red circled dates, and a brass letter opener, warm afternoon sunlight through a window, 4k, lifestyle photography." \
"typewriter"

launch 1 "violin_guitar" \
"A high-resolution photo of a polished wooden violin resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
"A high-resolution photo of a sleek black electric guitar resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
"violin"

launch 2 "penguin_flamingo" \
"A high-resolution wildlife photo of an emperor penguin standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
"A high-resolution wildlife photo of a pink flamingo standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
"penguin"

launch 3 "bgrich_teapot_globe" \
"A high-resolution photo of a porcelain teapot on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography." \
"A high-resolution photo of an antique terrestrial globe on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography." \
"teapot"

launch 4 "bgrich_candle_crystal" \
"A high-resolution photo of a tall beeswax candle with a flickering flame on a cluttered wooden workbench in an alchemist's workshop, surrounded by bubbling glass flasks of colored potions, dusty spellbooks, dried herbs hanging from the ceiling beams, a brass astrolabe, a human skull, raven feathers, a stone mortar and pestle, rolled yellowed scrolls, glowing amber jars of unknown substances, iron tongs, and a cauldron simmering on coals, moody chiaroscuro lighting with deep shadows, 4k, atmospheric fantasy photography." \
"A high-resolution photo of a glowing purple crystal cluster on a cluttered wooden workbench in an alchemist's workshop, surrounded by bubbling glass flasks of colored potions, dusty spellbooks, dried herbs hanging from the ceiling beams, a brass astrolabe, a human skull, raven feathers, a stone mortar and pestle, rolled yellowed scrolls, glowing amber jars of unknown substances, iron tongs, and a cauldron simmering on coals, moody chiaroscuro lighting with deep shadows, 4k, atmospheric fantasy photography." \
"candle"

launch 5 "butterfly_hummingbird" \
"A high-resolution macro photo of a monarch butterfly with orange and black wings perched on a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
"A high-resolution macro photo of a tiny iridescent green hummingbird hovering near a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
"butterfly"

launch 6 "sail_pirate" \
"A high-resolution photo of a small white sailboat with a single mast gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style." \
"A high-resolution photo of a large old wooden pirate ship with tattered black sails and a skull flag gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style." \
"sailboat"

launch 7 "cake_books" \
"A high-resolution photo of a white birthday cake with colorful candles on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, food photography style." \
"A high-resolution photo of a tall stack of old leather-bound books on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, still life photography style." \
"cake"

echo ""
echo "Launched all 8. Waiting..."
for NAME in bgrich_typewriter_laptop violin_guitar penguin_flamingo bgrich_teapot_globe bgrich_candle_crystal butterfly_hummingbird sail_pirate cake_books; do
    PID=$(cat /tmp/nbits20_${NAME}.pid 2>/dev/null)
    [ -n "$PID" ] && { wait $PID 2>/dev/null || true; echo "[$(date)] ${NAME} finished (rc=$?)"; }
done

echo ""
echo "============================================"
echo "  All done at $(date)"
echo "  Results in $OUT_BASE"
echo "============================================"
