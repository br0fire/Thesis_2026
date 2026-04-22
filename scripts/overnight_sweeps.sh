#!/bin/bash
# Run the 8-config parallel sweep on multiple experiments sequentially.
# Each experiment occupies all 8 GPUs for ~60 min; 5 experiments ≈ 5-6 hours.
#
# Failures in one experiment are isolated — the driver continues to the next.
# Progress is logged to logs/overnight_sweeps.log, per-experiment logs to
# logs/sweep_parallel_<experiment>_*.log.

set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

LOG=logs/overnight_sweeps.log
mkdir -p logs
echo "========================================" | tee -a "$LOG"
echo "  Overnight sweep driver" | tee -a "$LOG"
echo "  started: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

run_one() {
    local NAME=$1
    local SRC=$2
    local TGT=$3
    local SEG=$4

    echo "" | tee -a "$LOG"
    echo "[$(date)] >>> Starting experiment: $NAME" | tee -a "$LOG"
    local T0=$(date +%s)

    EXPERIMENT="$NAME" \
    SRC_PROMPT="$SRC" \
    TGT_PROMPT="$TGT" \
    SEG_PROMPT="$SEG" \
    NUM_EPISODES=80 \
    bash scripts/parallel_sweep.sh >> "$LOG" 2>&1

    local RC=$?
    local ELAPSED=$(( $(date +%s) - T0 ))
    if [ $RC -eq 0 ]; then
        echo "[$(date)] <<< $NAME OK in ${ELAPSED}s" | tee -a "$LOG"
    else
        echo "[$(date)] <<< $NAME FAILED (rc=$RC) after ${ELAPSED}s — continuing" | tee -a "$LOG"
    fi
}

# ─── Experiment 1: violin_guitar (object-centric, fine-grained swap) ───
run_one "violin_guitar" \
"A high-resolution photo of a polished wooden violin resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
"A high-resolution photo of a sleek black electric guitar resting on a dark velvet cloth draped over an antique wooden table, soft warm spotlight from above, a blurred concert hall with red curtains and empty seats in the background, dramatic chiaroscuro lighting, 4k, product photography style." \
"violin"

# ─── Experiment 2: penguin_flamingo (object-centric, color change) ───
run_one "penguin_flamingo" \
"A high-resolution wildlife photo of an emperor penguin standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
"A high-resolution wildlife photo of a pink flamingo standing on a flat icy shore, with a calm dark blue ocean behind it, distant icebergs and snow-covered mountains on the horizon, overcast sky with soft gray clouds, crisp polar light, 4k, National Geographic photography style." \
"penguin"

# ─── Experiment 3: bgrich_teapot_globe (bg-rich, replicate α=0.7) ───
run_one "bgrich_teapot_globe" \
"A high-resolution photo of a porcelain teapot on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography." \
"A high-resolution photo of an antique terrestrial globe on a polished dark mahogany desk in a Victorian study, surrounded by stacks of leather-bound books, a brass compass, an ink well with a white quill pen, round reading glasses, a smoking pipe in an ashtray, rolled parchment maps, a green-shaded banker's desk lamp, an antique pocket watch, and a worn leather armchair in the background, warm amber light streaming through a tall arched window, dust motes floating in the air, oil painting on the wall, 4k, fine art photography." \
"teapot"

# ─── Experiment 4: bgrich_candle_crystal (bg-rich, second replication) ───
run_one "bgrich_candle_crystal" \
"A high-resolution photo of a tall beeswax candle with a flickering flame on a cluttered wooden workbench in an alchemist's workshop, surrounded by bubbling glass flasks of colored potions, dusty spellbooks, dried herbs hanging from the ceiling beams, a brass astrolabe, a human skull, raven feathers, a stone mortar and pestle, rolled yellowed scrolls, glowing amber jars of unknown substances, iron tongs, and a cauldron simmering on coals, moody chiaroscuro lighting with deep shadows, 4k, atmospheric fantasy photography." \
"A high-resolution photo of a glowing purple crystal cluster on a cluttered wooden workbench in an alchemist's workshop, surrounded by bubbling glass flasks of colored potions, dusty spellbooks, dried herbs hanging from the ceiling beams, a brass astrolabe, a human skull, raven feathers, a stone mortar and pestle, rolled yellowed scrolls, glowing amber jars of unknown substances, iron tongs, and a cauldron simmering on coals, moody chiaroscuro lighting with deep shadows, 4k, atmospheric fantasy photography." \
"candle"

# ─── Experiment 5: butterfly_hummingbird (object-centric, small fg) ───
run_one "butterfly_hummingbird" \
"A high-resolution macro photo of a monarch butterfly with orange and black wings perched on a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
"A high-resolution macro photo of a tiny iridescent green hummingbird hovering near a vibrant pink hibiscus flower, lush green garden foliage blurred in the background, tiny water droplets on the petals, soft natural daylight, shallow depth of field, 4k, macro photography style." \
"butterfly"

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "  Overnight driver done: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
