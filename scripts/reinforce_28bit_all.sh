#!/bin/bash
# REINFORCE on 15 new_bgrich experiments with n_bits=28 (steps=28, repeat=1).
# Waits for exhaustive search to finish, then runs 2 waves × 8 GPUs.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project
source scripts/new_bgrich_prompts.sh

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
LOG=logs/reinforce_28bit.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
log "========== n_bits=28 REINFORCE waiting =========="

# Wait for exhaustive drivers + their workers
while pgrep -f "scripts/exhaustive" > /dev/null \
   || pgrep -f "exhaustive_search.py" > /dev/null; do
    log "Waiting for exhaustive..."
    sleep 300
done
log "Exhaustive done — starting n_bits=28 REINFORCE"

OUT_BASE="analysis/reinforce_analysis/new_bgrich_28bit"
mkdir -p "$OUT_BASE"

EXPS=(bgrich_globe_orrery bgrich_telescope_microscope bgrich_camera_binoculars \
      bgrich_pocketwatch_compass bgrich_teapot_samovar bgrich_chess_checkers \
      bgrich_violin_cello bgrich_guitar_banjo bgrich_lantern_candelabra \
      bgrich_wine_whiskey bgrich_revolver_flintlock bgrich_sewing_typewriter \
      bgrich_piano_harp bgrich_crown_tiara bgrich_skull_hourglass)

run_wave() {
    local NAMES=("$@")
    local GPU=0
    for name in "${NAMES[@]}"; do
        local outdir="${OUT_BASE}/${name}"
        mkdir -p "$outdir"
        log "  GPU ${GPU}: REINFORCE 28-bit ${name}"
        nohup $PYTHON -u generation/reinforce_search.py \
            --source_prompt "${NEW_SRC[$name]}" \
            --target_prompt "${NEW_TGT[$name]}" \
            --seg_prompt "${NEW_SEG[$name]}" \
            --output_dir "$outdir" \
            --gpu "$GPU" \
            --n_bits 28 --steps 28 \
            --batch_size 8 --top_k 10 --log_interval 10 \
            --num_episodes 80 --min_episodes 80 --plateau_window 0 --entropy_stop 0 \
            --lr 0.10 --alpha 0.7 --entropy_coeff 0.05 \
            > "logs/reinforce_28bit_${name}.log" 2>&1 &
        echo $! > /tmp/reinforce_28bit_${name}.pid
        GPU=$((GPU + 1))
    done
    for name in "${NAMES[@]}"; do
        local pid=$(cat /tmp/reinforce_28bit_${name}.pid 2>/dev/null)
        [ -n "$pid" ] && wait $pid 2>/dev/null || true
        log "    ${name} finished"
    done
}

log "=== Wave 1: 8 experiments ==="
run_wave "${EXPS[@]:0:8}"

log "=== Wave 2: 7 experiments ==="
run_wave "${EXPS[@]:8}"

# Compute all-ones + random baselines for n_bits=28
log "=== Computing all-ones + random for n_bits=28 ==="
GPU=0
pids=()
for name in "${EXPS[@]}"; do
    exp_dir="${OUT_BASE}/${name}"
    nohup $PYTHON -u analysis/compute_random_baseline.py \
        --exp_dir "$exp_dir" --gpu $GPU --n_samples 640 --n_bits 28 --steps 28 \
        > "logs/random_28bit_${name}.log" 2>&1 &
    pids+=($!)
    GPU=$(( (GPU + 1) % 8 ))
    if [ $GPU -eq 0 ]; then
        for p in "${pids[@]}"; do wait $p 2>/dev/null; done
        pids=()
    fi
done
for p in "${pids[@]}"; do wait $p 2>/dev/null; done

$PYTHON analysis/compute_all_ones_reward.py --dir "$OUT_BASE" --gpu 0 2>&1 | tee -a "$LOG"

log "=== Rebuilding 28-bit training curves plot ==="
$PYTHON analysis/new_bgrich_28bit_curves.py 2>&1 | tee -a "$LOG"

log "========== n_bits=28 REINFORCE done =========="
