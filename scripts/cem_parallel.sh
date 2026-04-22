#!/bin/bash
# CEM search on 6 experiments × 2 budgets (80 and 160 images) in parallel.
# 12 jobs × 1 GPU each, but only 8 GPUs. Run in two waves of 6.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
mkdir -p logs

declare -A SRC TGT SEG
SRC[bgrich_typewriter_laptop]="A high-resolution photo of a vintage black typewriter on a cluttered wooden writer's desk, surrounded by towering stacks of manuscript papers with handwritten edits, three half-empty coffee mugs, a jar of ballpoint pens and pencils, pink and yellow sticky notes with scribbled reminders, a silver framed family photograph, a small potted succulent plant, a bronze gooseneck desk lamp, crumpled paper balls on the floor, a tall bookshelf filled with novels in the background, a wall calendar with red circled dates, and a brass letter opener, warm afternoon sunlight through a window, 4k, lifestyle photography."
TGT[bgrich_typewriter_laptop]="A high-resolution photo of a sleek silver open laptop computer on a cluttered wooden writer's desk, surrounded by towering stacks of manuscript papers with handwritten edits, three half-empty coffee mugs, a jar of ballpoint pens and pencils, pink and yellow sticky notes with scribbled reminders, a silver framed family photograph, a small potted succulent plant, a bronze gooseneck desk lamp, crumpled paper balls on the floor, a tall bookshelf filled with novels in the background, a wall calendar with red circled dates, and a brass letter opener, warm afternoon sunlight through a window, 4k, lifestyle photography."
SEG[bgrich_typewriter_laptop]="typewriter"

SRC[violin_guitar]="A high-resolution photo of a polished wooden violin resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style."
TGT[violin_guitar]="A high-resolution photo of a sleek black electric guitar resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style."
SEG[violin_guitar]="violin"

SRC[penguin_flamingo]="A high-resolution wildlife photo of an emperor penguin standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style."
TGT[penguin_flamingo]="A high-resolution wildlife photo of a pink flamingo standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style."
SEG[penguin_flamingo]="penguin"

SRC[bgrich_teapot_globe]="A high-resolution photo of a porcelain teapot on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography."
TGT[bgrich_teapot_globe]="A high-resolution photo of an antique terrestrial globe on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography."
SEG[bgrich_teapot_globe]="teapot"

SRC[bgrich_candle_crystal]="A high-resolution photo of a tall beeswax candle with a flickering flame on a cluttered wooden workbench in an alchemist's workshop, surrounded by bubbling glass flasks of colored potions, dusty spellbooks, dried herbs hanging from the ceiling beams, a brass astrolabe, a human skull, raven feathers, a stone mortar and pestle, rolled yellowed scrolls, glowing amber jars of unknown substances, iron tongs, and a cauldron simmering on coals, moody chiaroscuro lighting with deep shadows, 4k, atmospheric fantasy photography."
TGT[bgrich_candle_crystal]="A high-resolution photo of a glowing purple crystal cluster on a cluttered wooden workbench in an alchemist's workshop, surrounded by bubbling glass flasks of colored potions, dusty spellbooks, dried herbs hanging from the ceiling beams, a brass astrolabe, a human skull, raven feathers, a stone mortar and pestle, rolled yellowed scrolls, glowing amber jars of unknown substances, iron tongs, and a cauldron simmering on coals, moody chiaroscuro lighting with deep shadows, 4k, atmospheric fantasy photography."
SEG[bgrich_candle_crystal]="candle"

SRC[butterfly_hummingbird]="A high-resolution macro photo of a monarch butterfly with orange and black wings perched on a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style."
TGT[butterfly_hummingbird]="A high-resolution macro photo of a tiny iridescent green hummingbird hovering near a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style."
SEG[butterfly_hummingbird]="butterfly"

EXPERIMENTS=(bgrich_typewriter_laptop violin_guitar penguin_flamingo bgrich_teapot_globe bgrich_candle_crystal butterfly_hummingbird)

echo "============================================"
echo "  CEM parallel: 6 experiments, 2 budgets"
echo "  Wave 1: budget=80 (10 iter × 8)"
echo "  $(date)"
echo "============================================"

GPU=0
for name in "${EXPERIMENTS[@]}"; do
    OUTDIR="analysis/reinforce_analysis/cem/${name}_budget80"
    mkdir -p "$OUTDIR"
    echo "[$(date)] GPU ${GPU}: ${name} budget=80"
    nohup $PYTHON -u generation/cem_search.py \
        --source_prompt "${SRC[$name]}" \
        --target_prompt "${TGT[$name]}" \
        --seg_prompt "${SEG[$name]}" \
        --output_dir "$OUTDIR" \
        --gpu "$GPU" \
        --n_bits 14 --batch_size 8 --num_iterations 10 \
        --elite_frac 0.25 --smoothing 0.3 \
        --alpha 0.5 --top_k 10 \
        > logs/cem_${name}_budget80.log 2>&1 &
    echo $! > /tmp/cem_${name}_budget80.pid
    GPU=$((GPU + 1))
done

echo ""
echo "Wave 1 launched. Waiting..."
for name in "${EXPERIMENTS[@]}"; do
    PID=$(cat /tmp/cem_${name}_budget80.pid 2>/dev/null)
    [ -n "$PID" ] && { wait $PID 2>/dev/null || true; echo "[$(date)] ${name}_budget80 finished"; }
done

echo ""
echo "============================================"
echo "  Wave 2: budget=160 (20 iter × 8)"
echo "  $(date)"
echo "============================================"

GPU=0
for name in "${EXPERIMENTS[@]}"; do
    OUTDIR="analysis/reinforce_analysis/cem/${name}_budget160"
    mkdir -p "$OUTDIR"
    echo "[$(date)] GPU ${GPU}: ${name} budget=160"
    nohup $PYTHON -u generation/cem_search.py \
        --source_prompt "${SRC[$name]}" \
        --target_prompt "${TGT[$name]}" \
        --seg_prompt "${SEG[$name]}" \
        --output_dir "$OUTDIR" \
        --gpu "$GPU" \
        --n_bits 14 --batch_size 8 --num_iterations 20 \
        --elite_frac 0.25 --smoothing 0.3 \
        --alpha 0.5 --top_k 10 \
        > logs/cem_${name}_budget160.log 2>&1 &
    echo $! > /tmp/cem_${name}_budget160.pid
    GPU=$((GPU + 1))
done

echo ""
echo "Wave 2 launched. Waiting..."
for name in "${EXPERIMENTS[@]}"; do
    PID=$(cat /tmp/cem_${name}_budget160.pid 2>/dev/null)
    [ -n "$PID" ] && { wait $PID 2>/dev/null || true; echo "[$(date)] ${name}_budget160 finished"; }
done

echo ""
echo "============================================"
echo "  All CEM done at $(date)"
echo "============================================"
