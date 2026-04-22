#!/bin/bash
# Recompute bg_ssim + fg_clip for all 15 exhaustive experiments (no FLUX).
# 8 in parallel on 8 GPUs, then remaining 7. ~10-15 min total.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
LOG=logs/recompute_exhaustive.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
log "========== Recompute exhaustive components started =========="

EXPS=(bgrich_globe_orrery bgrich_telescope_microscope bgrich_camera_binoculars \
      bgrich_pocketwatch_compass bgrich_teapot_samovar bgrich_chess_checkers \
      bgrich_violin_cello bgrich_guitar_banjo bgrich_lantern_candelabra \
      bgrich_wine_whiskey bgrich_revolver_flintlock bgrich_sewing_typewriter \
      bgrich_piano_harp bgrich_crown_tiara bgrich_skull_hourglass)

run_wave() {
    local NAMES=("$@")
    local GPU=0
    for name in "${NAMES[@]}"; do
        log "  GPU ${GPU}: recompute ${name}"
        nohup $PYTHON -u analysis/recompute_exhaustive_components.py \
            --exp_name "$name" --gpu $GPU \
            > "logs/recompute_${name}.log" 2>&1 &
        echo $! > /tmp/recomp_${name}.pid
        GPU=$((GPU + 1))
    done
    for name in "${NAMES[@]}"; do
        local pid=$(cat /tmp/recomp_${name}.pid 2>/dev/null)
        [ -n "$pid" ] && wait $pid 2>/dev/null || true
        log "    ${name} done"
    done
}

log "=== Wave 1 (8 exps) ==="
run_wave "${EXPS[@]:0:8}"

log "=== Wave 2 (7 exps) ==="
run_wave "${EXPS[@]:8}"

log "========== All 15 recomputed =========="
