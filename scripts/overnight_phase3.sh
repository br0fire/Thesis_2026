#!/bin/bash
# Phase 3: after overnight_big.sh + reinforce_2_now.sh finish.
# 1. Wait for all prior work
# 2. REINFORCE on 13 remaining new bgrich experiments (2 waves on 8 GPUs)
# 3. CEM on all 15 new experiments at budgets 40, 80, 160
# 4. Retrain amortized policy with 30-experiment dataset
# 5. Eval amortized on all 30
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project
source scripts/new_bgrich_prompts.sh

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
LOG=logs/overnight_phase3.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

log "========== Phase 3 started =========="

# Wait for overnight_big and any REINFORCE/CEM jobs to finish
while pgrep -f "scripts/overnight_big.sh" > /dev/null \
   || pgrep -f "scripts/cem_parallel.sh" > /dev/null \
   || pgrep -f "reinforce_search.py" > /dev/null \
   || pgrep -f "cem_search.py" > /dev/null; do
    log "Waiting for prior work (overnight_big / REINFORCE / CEM)..."
    sleep 120
done
log "Prior work done"

# All 15 new experiments (skip the 2 already done by reinforce_2_now.sh)
NEW_EXPERIMENTS=(bgrich_globe_orrery bgrich_telescope_microscope bgrich_camera_binoculars \
                 bgrich_pocketwatch_compass bgrich_teapot_samovar bgrich_chess_checkers \
                 bgrich_violin_cello bgrich_guitar_banjo bgrich_lantern_candelabra \
                 bgrich_wine_whiskey bgrich_revolver_flintlock bgrich_sewing_typewriter \
                 bgrich_piano_harp bgrich_crown_tiara bgrich_skull_hourglass)

# Skip any experiment that already has reinforce_result.pt (done by reinforce_2_now or reinforce_6_now)
REMAINING_REINFORCE=()
for name in bgrich_globe_orrery bgrich_telescope_microscope bgrich_camera_binoculars \
            bgrich_pocketwatch_compass bgrich_teapot_samovar bgrich_chess_checkers \
            bgrich_violin_cello bgrich_guitar_banjo bgrich_lantern_candelabra \
            bgrich_wine_whiskey bgrich_revolver_flintlock bgrich_sewing_typewriter \
            bgrich_piano_harp bgrich_crown_tiara bgrich_skull_hourglass; do
    if [ ! -f "analysis/reinforce_analysis/new_bgrich/${name}/reinforce_result.pt" ]; then
        REMAINING_REINFORCE+=("$name")
    fi
done
log "REINFORCE remaining: ${#REMAINING_REINFORCE[@]} — ${REMAINING_REINFORCE[*]}"

# ────────────────────────────────────────────
# REINFORCE runner
# ────────────────────────────────────────────
run_reinforce_wave() {
    local NAMES=("$@")
    local GPU=0
    for name in "${NAMES[@]}"; do
        local outdir="analysis/reinforce_analysis/new_bgrich/${name}"
        mkdir -p "$outdir"
        log "  GPU ${GPU}: REINFORCE ${name}"
        nohup $PYTHON -u generation/reinforce_search.py \
            --source_prompt "${NEW_SRC[$name]}" \
            --target_prompt "${NEW_TGT[$name]}" \
            --seg_prompt "${NEW_SEG[$name]}" \
            --output_dir "$outdir" \
            --gpu $GPU \
            --n_bits 14 --batch_size 8 --top_k 10 --log_interval 10 \
            --num_episodes 80 --min_episodes 80 --plateau_window 0 --entropy_stop 0 \
            --lr 0.10 --alpha 0.7 --entropy_coeff 0.05 \
            > "logs/reinforce_new_${name}.log" 2>&1 &
        echo $! > /tmp/reinforce_new_${name}.pid
        GPU=$((GPU + 1))
    done
    for name in "${NAMES[@]}"; do
        local pid=$(cat /tmp/reinforce_new_${name}.pid 2>/dev/null)
        [ -n "$pid" ] && wait $pid 2>/dev/null || true
        log "    REINFORCE ${name} finished"
    done
}

if [ ${#REMAINING_REINFORCE[@]} -gt 0 ]; then
    log "=== REINFORCE on ${#REMAINING_REINFORCE[@]} remaining new experiments ==="
    run_reinforce_wave "${REMAINING_REINFORCE[@]:0:8}"
    if [ ${#REMAINING_REINFORCE[@]} -gt 8 ]; then
        run_reinforce_wave "${REMAINING_REINFORCE[@]:8}"
    fi
else
    log "=== All REINFORCE already done ==="
fi

# ────────────────────────────────────────────
# CEM on all 15 new at 3 budgets
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
            --source_prompt "${NEW_SRC[$name]}" \
            --target_prompt "${NEW_TGT[$name]}" \
            --seg_prompt "${NEW_SEG[$name]}" \
            --output_dir "$outdir" \
            --gpu $GPU \
            --n_bits 14 --batch_size 8 --num_iterations $ITERS \
            --elite_frac 0.25 --smoothing 0.3 \
            --alpha 0.5 --top_k 10 \
            > "logs/cem_${name}_budget${BUDGET}.log" 2>&1 &
        echo $! > /tmp/cem_${name}_budget${BUDGET}.pid
        GPU=$((GPU + 1))
    done
    for name in "${NAMES[@]}"; do
        local pid=$(cat /tmp/cem_${name}_budget${BUDGET}.pid 2>/dev/null)
        [ -n "$pid" ] && wait $pid 2>/dev/null || true
    done
}

for BUDGET_ITERS in "40 5" "80 10" "160 20"; do
    BUDGET=${BUDGET_ITERS% *}; ITERS=${BUDGET_ITERS#* }
    log "=== CEM budget=${BUDGET} for all 15 new ==="
    run_cem_wave "$BUDGET" "$ITERS" "${NEW_EXPERIMENTS[@]:0:8}"
    run_cem_wave "$BUDGET" "$ITERS" "${NEW_EXPERIMENTS[@]:8}"
done

# ────────────────────────────────────────────
# Retrain amortized policy with extended dataset (30 experiments)
# ────────────────────────────────────────────
log "=== Retraining amortized policy with extended dataset ==="
$PYTHON analysis/train_amortized_policy.py > logs/retrain_amortized.log 2>&1
log "  retraining done"

# ────────────────────────────────────────────
# Amortized eval on all experiments with predictions
# ────────────────────────────────────────────
log "=== Amortized eval on all experiments ==="
ALL_PRED=$($PYTHON -c "
import json
d = json.load(open('analysis/reinforce_analysis/amortized/predictions.json'))
print(' '.join(d.keys()))
")
read -ra ALL_NAMES <<< "$ALL_PRED"
log "  ${#ALL_NAMES[@]} experiments with predictions"

GPU=0
running=()
for name in "${ALL_NAMES[@]}"; do
    log "  GPU ${GPU}: eval_amortized ${name}"
    nohup $PYTHON -u analysis/eval_amortized.py \
        --experiment "$name" --gpu $GPU \
        > "logs/eval_amort_${name}.log" 2>&1 &
    running+=($!)
    GPU=$(( (GPU + 1) % 8 ))
    # Wait if we've filled 8 slots
    if [ ${#running[@]} -ge 8 ]; then
        for p in "${running[@]}"; do wait $p 2>/dev/null || true; done
        running=()
    fi
done
# Final wait
for p in "${running[@]}"; do wait $p 2>/dev/null || true; done

log "========== Phase 3 done =========="
