#!/bin/bash
# REINFORCE with alpha=0.5 for the 15 new_bgrich experiments.
# Runs BOTH n_bits=14 (repeat=2, matches new_bgrich) and n_bits=28 (repeat=1).
# Waits for prior scheduled jobs (exhaustive + reinforce_28bit) to finish.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project
source scripts/new_bgrich_prompts.sh

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
LOG=logs/reinforce_alpha05.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
log "========== alpha=0.5 REINFORCE waiting =========="

while pgrep -f "scripts/exhaustive" > /dev/null \
   || pgrep -f "scripts/reinforce_28bit" > /dev/null \
   || pgrep -f "exhaustive_search.py" > /dev/null \
   || pgrep -f "reinforce_search.py" > /dev/null \
   || pgrep -f "compute_random_baseline.py" > /dev/null \
   || pgrep -f "compute_all_ones_reward.py" > /dev/null; do
    log "Waiting for prior jobs..."
    sleep 300
done
log "Prior jobs done — starting alpha=0.5 REINFORCE"

EXPS=(bgrich_globe_orrery bgrich_telescope_microscope bgrich_camera_binoculars \
      bgrich_pocketwatch_compass bgrich_teapot_samovar bgrich_chess_checkers \
      bgrich_violin_cello bgrich_guitar_banjo bgrich_lantern_candelabra \
      bgrich_wine_whiskey bgrich_revolver_flintlock bgrich_sewing_typewriter \
      bgrich_piano_harp bgrich_crown_tiara bgrich_skull_hourglass)

run_wave() {
    # $1 = out_base, $2 = n_bits, $3 = steps (or "default")
    local OUT_BASE=$1 NBITS=$2 STEPS_ARG=$3
    shift 3
    local NAMES=("$@")
    local GPU=0
    local steps_opts=""
    [ "$STEPS_ARG" != "default" ] && steps_opts="--steps $STEPS_ARG"
    for name in "${NAMES[@]}"; do
        local outdir="${OUT_BASE}/${name}"
        mkdir -p "$outdir"
        log "  GPU ${GPU}: REINFORCE n_bits=${NBITS} α=0.5 ${name}"
        nohup $PYTHON -u generation/reinforce_search.py \
            --source_prompt "${NEW_SRC[$name]}" \
            --target_prompt "${NEW_TGT[$name]}" \
            --seg_prompt "${NEW_SEG[$name]}" \
            --output_dir "$outdir" \
            --gpu "$GPU" \
            --n_bits $NBITS $steps_opts \
            --batch_size 8 --top_k 10 --log_interval 10 \
            --num_episodes 80 --min_episodes 80 --plateau_window 0 --entropy_stop 0 \
            --lr 0.10 --alpha 0.5 --entropy_coeff 0.05 \
            > "logs/reinforce_alpha05_${OUT_BASE##*/}_${name}.log" 2>&1 &
        echo $! > /tmp/reinforce_alpha05_${name}.pid
        GPU=$((GPU + 1))
    done
    for name in "${NAMES[@]}"; do
        local pid=$(cat /tmp/reinforce_alpha05_${name}.pid 2>/dev/null)
        [ -n "$pid" ] && wait $pid 2>/dev/null || true
    done
}

# ── Phase A: n_bits=14 α=0.5 ──
OUT_BASE_14="analysis/reinforce_analysis/new_bgrich_alpha05"
log "=== n_bits=14 α=0.5 wave 1 (8 exps) ==="
run_wave "$OUT_BASE_14" 14 default "${EXPS[@]:0:8}"
log "=== n_bits=14 α=0.5 wave 2 (7 exps) ==="
run_wave "$OUT_BASE_14" 14 default "${EXPS[@]:8}"

# Baselines for 14-bit α=0.5
log "=== random baseline for 14-bit α=0.5 ==="
GPU=0; pids=()
for name in "${EXPS[@]}"; do
    nohup $PYTHON -u analysis/compute_random_baseline.py \
        --exp_dir "${OUT_BASE_14}/${name}" --gpu $GPU --n_samples 640 --n_bits 14 \
        > "logs/random_alpha05_14_${name}.log" 2>&1 &
    pids+=($!)
    GPU=$(( (GPU + 1) % 8 ))
    [ $GPU -eq 0 ] && { for p in "${pids[@]}"; do wait $p 2>/dev/null; done; pids=(); }
done
for p in "${pids[@]}"; do wait $p 2>/dev/null; done
$PYTHON analysis/compute_all_ones_reward.py --dir "$OUT_BASE_14" --gpu 0 --alpha 0.5 2>&1 | tee -a "$LOG"

# ── Phase B: n_bits=28 α=0.5 ──
OUT_BASE_28="analysis/reinforce_analysis/new_bgrich_28bit_alpha05"
log "=== n_bits=28 α=0.5 wave 1 (8 exps) ==="
run_wave "$OUT_BASE_28" 28 28 "${EXPS[@]:0:8}"
log "=== n_bits=28 α=0.5 wave 2 (7 exps) ==="
run_wave "$OUT_BASE_28" 28 28 "${EXPS[@]:8}"

# Baselines for 28-bit α=0.5
log "=== random baseline for 28-bit α=0.5 ==="
GPU=0; pids=()
for name in "${EXPS[@]}"; do
    nohup $PYTHON -u analysis/compute_random_baseline.py \
        --exp_dir "${OUT_BASE_28}/${name}" --gpu $GPU --n_samples 640 \
        --n_bits 28 --steps 28 \
        > "logs/random_alpha05_28_${name}.log" 2>&1 &
    pids+=($!)
    GPU=$(( (GPU + 1) % 8 ))
    [ $GPU -eq 0 ] && { for p in "${pids[@]}"; do wait $p 2>/dev/null; done; pids=(); }
done
for p in "${pids[@]}"; do wait $p 2>/dev/null; done
$PYTHON analysis/compute_all_ones_reward.py --dir "$OUT_BASE_28" --gpu 0 --alpha 0.5 2>&1 | tee -a "$LOG"

# Plots
log "=== Building plots ==="
sed 's|new_bgrich"|new_bgrich_alpha05"|; s|new_bgrich REINFORCE|new_bgrich α=0.5 REINFORCE|; s|new_bgrich_training_curves|new_bgrich_alpha05_training_curves|' \
    analysis/new_bgrich_curves.py > /tmp/plot_alpha05_14.py
$PYTHON /tmp/plot_alpha05_14.py 2>&1 | tee -a "$LOG"
sed 's|new_bgrich"|new_bgrich_28bit_alpha05"|; s|new_bgrich REINFORCE|new_bgrich 28-bit α=0.5 REINFORCE|; s|new_bgrich_training_curves|new_bgrich_28bit_alpha05_training_curves|' \
    analysis/new_bgrich_curves.py > /tmp/plot_alpha05_28.py
$PYTHON /tmp/plot_alpha05_28.py 2>&1 | tee -a "$LOG"

log "========== alpha=0.5 REINFORCE done =========="
