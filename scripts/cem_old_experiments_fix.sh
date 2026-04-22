#!/bin/bash
# Re-run CEM for the 9 old experiments using ORIGINAL v-series prompts.
# The previous CEM runs used different (made-up) prompts and are invalid.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
LOG=logs/cem_old_fix.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
log "========== CEM fix for 9 old experiments =========="

# Discover (name, prompts.txt) from v-series
declare -A PROMPTS_PATH
for name in catdog horse lighthouse_castle cake_books room sail_pirate snow_volcano car_taxi chair_throne; do
    for v in v6 v5 v4 v3; do
        p="analysis/reinforce_analysis/${v}/experiments/reinforce_${name}_${v}/prompts.txt"
        if [ -f "$p" ]; then PROMPTS_PATH[$name]="$p"; break; fi
    done
    [ -z "${PROMPTS_PATH[$name]}" ] && { log "  ERROR: no prompts for $name"; exit 1; }
done
log "Resolved prompts for ${#PROMPTS_PATH[@]} experiments"

# Read fields from prompts.txt
get_field() {
    local p="$1"; local k="$2"
    grep "^${k}:" "$p" | head -1 | sed "s/^${k}:[[:space:]]*//"
}

# Remove old (wrong-prompt) CEM dirs first
log "Removing old CEM dirs for 9 experiments..."
for name in "${!PROMPTS_PATH[@]}"; do
    for b in 40 80 160; do
        rm -rf "analysis/reinforce_analysis/cem/${name}_budget${b}"
    done
done

# Run CEM at 3 budgets for 9 experiments — each wave uses up to 8 GPUs
run_cem_wave() {
    local BUDGET=$1
    local ITERS=$2
    shift 2
    local NAMES=("$@")
    local GPU=0
    for name in "${NAMES[@]}"; do
        local p="${PROMPTS_PATH[$name]}"
        local src="$(get_field "$p" source)"
        local tgt="$(get_field "$p" target)"
        local seg="$(get_field "$p" seg)"
        local outdir="analysis/reinforce_analysis/cem/${name}_budget${BUDGET}"
        mkdir -p "$outdir"
        log "  GPU ${GPU}: CEM ${name} budget=${BUDGET} (seg='$seg')"
        nohup $PYTHON -u generation/cem_search.py \
            --source_prompt "$src" \
            --target_prompt "$tgt" \
            --seg_prompt "$seg" \
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
        [ -n "$pid" ] && wait $pid 2>/dev/null || true
    done
}

EXPS=(catdog horse lighthouse_castle cake_books room sail_pirate snow_volcano car_taxi chair_throne)

for CFG in "40 5" "80 10" "160 20"; do
    BUDGET=${CFG% *}; ITERS=${CFG#* }
    log "=== CEM budget=${BUDGET} wave 1 (8 exps) ==="
    run_cem_wave "$BUDGET" "$ITERS" "${EXPS[@]:0:8}"
    log "=== CEM budget=${BUDGET} wave 2 (1 exp) ==="
    run_cem_wave "$BUDGET" "$ITERS" "${EXPS[@]:8}"
done

log "========== Done =========="
