#!/bin/bash
# Overnight driver: CEM on all 15 experiments × 3 budgets + amortized eval.
# Waits for currently-running CEM (scripts/cem_parallel.sh, 6 experiments) to finish,
# then launches everything else in parallel waves of 8 on 8 GPUs.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
LOG=logs/overnight_big.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

log "========== Overnight driver started =========="

# Wait for any REINFORCE and cem_parallel.sh to finish
any_running() {
    pgrep -f "scripts/cem_parallel.sh" > /dev/null && return 0
    pgrep -f "reinforce_search.py" > /dev/null && return 0
    return 1
}

while any_running; do
    log "Waiting for active REINFORCE + cem_parallel.sh to finish..."
    sleep 60
done
log "All prior work done — starting main overnight workload"

# ────────────────────────────────────────────
# Experiment definitions (14-bit, with prompts)
# ────────────────────────────────────────────
declare -A SRC TGT SEG

# Object-centric (from sweeps + v-series)
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

SRC[sail_pirate]="A high-resolution photo of a small white sailboat with a single mast gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style."
TGT[sail_pirate]="A high-resolution photo of a large old wooden pirate ship with tattered black sails and a skull flag gliding on a calm turquoise tropical ocean, a distant sandy beach with palm trees on the horizon, a few white seagulls in the bright blue sky, sparkling sunlight reflections on the water surface, 4k, nautical photography style."
SEG[sail_pirate]="sailboat"

SRC[cake_books]="A high-resolution photo of a white birthday cake with colorful candles on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, food photography style."
TGT[cake_books]="A high-resolution photo of a tall stack of old leather-bound books on a rustic wooden kitchen table, a checkered tablecloth underneath, warm morning sunlight streaming through a nearby window, blurred kitchen shelves with jars and plants in the background, cozy domestic atmosphere, 4k, still life photography style."
SEG[cake_books]="cake"

SRC[horse]="A high-resolution photo of a brown horse standing in a meadow, trees in the background, daylight, 4k, photography."
TGT[horse]="A high-resolution photo of a zebra standing in a meadow, trees in the background, daylight, 4k, photography."
SEG[horse]="horse"

SRC[lighthouse_castle]="A high-resolution photo of a white lighthouse on a rocky cliff overlooking the sea, stormy grey sky, crashing waves, dramatic lighting, 4k, photography."
TGT[lighthouse_castle]="A high-resolution photo of a medieval stone castle on a rocky cliff overlooking the sea, stormy grey sky, crashing waves, dramatic lighting, 4k, photography."
SEG[lighthouse_castle]="lighthouse"

SRC[car_taxi]="A high-resolution photo of a blue sedan car parked on a city street, tall buildings on both sides, warm afternoon sunlight, 4k, photography."
TGT[car_taxi]="A high-resolution photo of a yellow taxi cab parked on a city street, tall buildings on both sides, warm afternoon sunlight, 4k, photography."
SEG[car_taxi]="car"

SRC[catdog]="A high-resolution photo of a tabby cat walking on stone pavement in a park, trees and benches in the background, daylight, 4k, photography."
TGT[catdog]="A high-resolution photo of a golden retriever dog walking on stone pavement in a park, trees and benches in the background, daylight, 4k, photography."
SEG[catdog]="cat"

SRC[chair_throne]="A high-resolution photo of a simple wooden chair in a living room, beige walls, warm lighting, 4k, photography."
TGT[chair_throne]="A high-resolution photo of a golden medieval throne in a living room, beige walls, warm lighting, 4k, photography."
SEG[chair_throne]="chair"

SRC[room]="A high-resolution photo of a cozy living room with a sofa, wooden coffee table, rug, and bookshelf by a window, warm afternoon sunlight, 4k, photography."
TGT[room]="A high-resolution photo of a cozy living room with a sofa, glass coffee table, rug, and bookshelf by a window, warm afternoon sunlight, 4k, photography."
SEG[room]="wooden coffee table"

SRC[snow_volcano]="A high-resolution photo of a snow-covered mountain peak against a blue sky, crisp winter light, 4k, landscape photography."
TGT[snow_volcano]="A high-resolution photo of an erupting volcano with lava and smoke against a dark sky, 4k, landscape photography."
SEG[snow_volcano]="mountain peak"

# List of experiments to run
EXPERIMENTS=(bgrich_typewriter_laptop violin_guitar penguin_flamingo bgrich_teapot_globe bgrich_candle_crystal butterfly_hummingbird sail_pirate cake_books horse lighthouse_castle car_taxi catdog chair_throne room snow_volcano)

# ────────────────────────────────────────────
# CEM runner helper
# ────────────────────────────────────────────
run_cem_wave() {
    local BUDGET=$1
    local ITERS=$2
    shift 2
    local NAMES=("$@")
    local GPU=0
    for name in "${NAMES[@]}"; do
        local outdir="analysis/reinforce_analysis/cem/${name}_budget${BUDGET}"
        mkdir -p "$outdir"
        log "  GPU ${GPU}: CEM ${name} budget=${BUDGET}"
        nohup $PYTHON -u generation/cem_search.py \
            --source_prompt "${SRC[$name]}" \
            --target_prompt "${TGT[$name]}" \
            --seg_prompt "${SEG[$name]}" \
            --output_dir "$outdir" \
            --gpu "$GPU" \
            --n_bits 14 --batch_size 8 --num_iterations $ITERS \
            --elite_frac 0.25 --smoothing 0.3 \
            --alpha 0.5 --top_k 10 \
            > "logs/cem_${name}_budget${BUDGET}.log" 2>&1 &
        echo $! > /tmp/cem_${name}_budget${BUDGET}.pid
        GPU=$((GPU + 1))
    done
    for name in "${NAMES[@]}"; do
        local pid=$(cat /tmp/cem_${name}_budget${BUDGET}.pid 2>/dev/null)
        if [ -n "$pid" ]; then
            wait $pid 2>/dev/null || true
            log "    ${name} budget=${BUDGET} finished"
        fi
    done
}

# ────────────────────────────────────────────
# CEM at budget 40 for all 15 (we don't have any yet)
# ────────────────────────────────────────────
log "=== CEM budget=40 for all 15 experiments ==="
run_cem_wave 40 5 "${EXPERIMENTS[@]:0:8}"
run_cem_wave 40 5 "${EXPERIMENTS[@]:8:7}"

# ────────────────────────────────────────────
# CEM at budget 80 for all 15
# ────────────────────────────────────────────
log "=== CEM budget=80 for all 15 ==="
run_cem_wave 80 10 "${EXPERIMENTS[@]:0:8}"
run_cem_wave 80 10 "${EXPERIMENTS[@]:8:7}"

# ────────────────────────────────────────────
# CEM at budget 160 for all 15
# ────────────────────────────────────────────
log "=== CEM budget=160 for all 15 ==="
run_cem_wave 160 20 "${EXPERIMENTS[@]:0:8}"
run_cem_wave 160 20 "${EXPERIMENTS[@]:8:7}"

# ────────────────────────────────────────────
# Amortized policy evaluation (1 image per strategy per experiment)
# ────────────────────────────────────────────
log "=== Amortized policy evaluation on all 15 ==="
GPU=0
for name in "${EXPERIMENTS[@]}"; do
    log "  GPU ${GPU}: eval_amortized ${name}"
    nohup $PYTHON -u analysis/eval_amortized.py \
        --experiment "$name" --gpu $GPU \
        > "logs/eval_amort_${name}.log" 2>&1 &
    echo $! > /tmp/eval_amort_${name}.pid
    GPU=$(( (GPU + 1) % 8 ))
    # Wait for a slot if we've launched 8
    if [ $GPU -eq 0 ]; then
        for n in "${EXPERIMENTS[@]}"; do
            pid=$(cat /tmp/eval_amort_${n}.pid 2>/dev/null)
            [ -n "$pid" ] && wait $pid 2>/dev/null
        done
    fi
done
# Final wait
for name in "${EXPERIMENTS[@]}"; do
    pid=$(cat /tmp/eval_amort_${name}.pid 2>/dev/null)
    [ -n "$pid" ] && wait $pid 2>/dev/null || true
done

log "========== Overnight driver done =========="
