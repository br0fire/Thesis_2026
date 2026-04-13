#!/bin/bash
# New experiments with prompt pairs that have extremely rich, object-dense backgrounds.
# Goal: stress-test background preservation. The foreground swap should be clean;
# all the background objects/textures/details should remain intact across the edit.
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

NFS3=/home/jovyan/shares/SR006.nfs3/svgrozny

# Uses new defaults: SigLIP2 + relative reward + num_episodes=300, min_episodes=200,
# plateau_patience=150, alpha=0.3

run_bgrich() {
    local GPU=$1
    local NAME=$2
    local SRC="$3"
    local TGT="$4"
    local SEG="$5"
    local OUTDIR="${NFS3}/reinforce_${NAME}"
    local LOGFILE="logs/reinforce_${NAME}.log"

    echo "[$(date)] Starting ${NAME} on GPU ${GPU} → ${LOGFILE}"
    nohup $PYTHON -u generation/reinforce_search.py \
        --source_prompt "$SRC" \
        --target_prompt "$TGT" \
        --seg_prompt "$SEG" \
        --output_dir "$OUTDIR" \
        --gpu $GPU \
        --n_bits 14 \
        --batch_size 8 \
        --top_k 10 \
        --log_interval 10 \
        > "$LOGFILE" 2>&1 &
    echo $! > "/tmp/reinforce_${NAME}.pid"
    echo "  PID=$(cat /tmp/reinforce_${NAME}.pid)"
}

echo "============================================"
echo "  Background-rich experiments"
echo "  $(date)"
echo "============================================"

# 1. Victorian study: teapot → globe
# Background has: old books, brass compass, ink well, quill, reading glasses,
# pipe, desk lamp, rolled maps, leather chair, stacks of papers, antique clock
run_bgrich 5 "bgrich_teapot_globe" \
    "A high-resolution photo of a porcelain teapot on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography." \
    "A high-resolution photo of an antique terrestrial globe on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography." \
    "teapot"

# 2. Alchemist's workshop: candle → crystal
# Background has: bubbling potions, glass flasks, dried herbs hanging, dusty
# spellbooks, astrolabe, skull, feathers, mortar and pestle, old scrolls
run_bgrich 6 "bgrich_candle_crystal" \
    "A high-resolution photo of a tall beeswax candle with a flickering flame on a cluttered wooden workbench in an alchemist's workshop, surrounded by bubbling glass flasks of colored potions, dusty spellbooks, dried herbs hanging from the ceiling beams, a brass astrolabe, a human skull, raven feathers, a stone mortar and pestle, rolled yellowed scrolls, glowing amber jars of unknown substances, iron tongs, and a cauldron simmering on coals, moody chiaroscuro lighting with deep shadows, 4k, atmospheric fantasy photography." \
    "A high-resolution photo of a glowing purple crystal cluster on a cluttered wooden workbench in an alchemist's workshop, surrounded by bubbling glass flasks of colored potions, dusty spellbooks, dried herbs hanging from the ceiling beams, a brass astrolabe, a human skull, raven feathers, a stone mortar and pestle, rolled yellowed scrolls, glowing amber jars of unknown substances, iron tongs, and a cauldron simmering on coals, moody chiaroscuro lighting with deep shadows, 4k, atmospheric fantasy photography." \
    "candle"

# 3. Writer's cluttered desk: typewriter → laptop
# Background has: stacks of papers, coffee mugs, pens, sticky notes, photo
# frames, potted plant, desk lamp, bookshelf, crumpled paper, wall calendar
run_bgrich 7 "bgrich_typewriter_laptop" \
    "A high-resolution photo of a vintage black typewriter on a cluttered wooden writer's desk, surrounded by towering stacks of manuscript papers with handwritten edits, three half-empty coffee mugs, a jar of ballpoint pens and pencils, pink and yellow sticky notes with scribbled reminders, a silver framed family photograph, a small potted succulent plant, a bronze gooseneck desk lamp, crumpled paper balls on the floor, a tall bookshelf filled with novels in the background, a wall calendar with red circled dates, and a brass letter opener, warm afternoon sunlight through a window, 4k, lifestyle photography." \
    "A high-resolution photo of a sleek silver open laptop computer on a cluttered wooden writer's desk, surrounded by towering stacks of manuscript papers with handwritten edits, three half-empty coffee mugs, a jar of ballpoint pens and pencils, pink and yellow sticky notes with scribbled reminders, a silver framed family photograph, a small potted succulent plant, a bronze gooseneck desk lamp, crumpled paper balls on the floor, a tall bookshelf filled with novels in the background, a wall calendar with red circled dates, and a brass letter opener, warm afternoon sunlight through a window, 4k, lifestyle photography." \
    "typewriter"

echo ""
echo "Launched 3 bg-rich experiments on GPUs 5, 6, 7. Waiting..."

for NAME in bgrich_teapot_globe bgrich_candle_crystal bgrich_typewriter_laptop; do
    PID=$(cat /tmp/reinforce_${NAME}.pid 2>/dev/null)
    if [ -n "$PID" ]; then
        wait $PID 2>/dev/null || true
        echo "[$(date)] ${NAME} finished"
    fi
done

echo ""
echo "============================================"
echo "  Background-rich experiments DONE at $(date)"
echo "============================================"
