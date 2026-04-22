#!/bin/bash
# Re-run random baseline for all 28-bit experiments saving bg/fg components too.
# Covers both new_bgrich_28bit and new_bgrich_28bit_alpha05 (same source masks,
# so we only need one run per exp — save under new_bgrich_28bit, symlink to alpha05).
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
LOG=logs/rerun_random_28bit.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
log "========== Re-run random 28-bit (with bg/fg) =========="

EXPS=(bgrich_globe_orrery bgrich_telescope_microscope bgrich_camera_binoculars \
      bgrich_pocketwatch_compass bgrich_teapot_samovar bgrich_chess_checkers \
      bgrich_violin_cello bgrich_guitar_banjo bgrich_lantern_candelabra \
      bgrich_wine_whiskey bgrich_revolver_flintlock bgrich_sewing_typewriter \
      bgrich_piano_harp bgrich_crown_tiara bgrich_skull_hourglass)

# Delete old files so skip-check lets us re-run
for base in new_bgrich_28bit new_bgrich_28bit_alpha05; do
    for name in "${EXPS[@]}"; do
        rm -f analysis/reinforce_analysis/${base}/${name}/random_rewards.npy
        rm -f analysis/reinforce_analysis/${base}/${name}/random_bg_ssim.npy
        rm -f analysis/reinforce_analysis/${base}/${name}/random_fg_clip.npy
    done
done

run_wave() {
    local NAMES=("$@")
    local GPU=0
    for name in "${NAMES[@]}"; do
        local exp_dir="analysis/reinforce_analysis/new_bgrich_28bit/${name}"
        [ -d "$exp_dir" ] || continue
        log "  GPU ${GPU}: random 28-bit ${name}"
        nohup $PYTHON -u analysis/compute_random_baseline.py \
            --exp_dir "$exp_dir" --gpu $GPU --n_samples 640 \
            --n_bits 28 --steps 28 --alpha 0.5 \
            > "logs/random_28bit_rerun_${name}.log" 2>&1 &
        echo $! > /tmp/rand28_${name}.pid
        GPU=$((GPU + 1))
    done
    for name in "${NAMES[@]}"; do
        local pid=$(cat /tmp/rand28_${name}.pid 2>/dev/null)
        [ -n "$pid" ] && wait $pid 2>/dev/null || true
        log "    ${name} done"
    done
}

log "=== Wave 1 (8 exps) ==="
run_wave "${EXPS[@]:0:8}"
log "=== Wave 2 (7 exps) ==="
run_wave "${EXPS[@]:8}"

# Copy bg/fg arrays to alpha05 dir (same data, different consumer)
log "=== Copying bg/fg to new_bgrich_28bit_alpha05 ==="
for name in "${EXPS[@]}"; do
    src="analysis/reinforce_analysis/new_bgrich_28bit/${name}"
    dst="analysis/reinforce_analysis/new_bgrich_28bit_alpha05/${name}"
    [ -d "$dst" ] || continue
    cp "$src/random_bg_ssim.npy" "$dst/random_bg_ssim.npy" 2>/dev/null
    cp "$src/random_fg_clip.npy" "$dst/random_fg_clip.npy" 2>/dev/null
done

log "========== Done =========="
