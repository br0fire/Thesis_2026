#!/bin/bash
# Exhaustive search (all 2^14 = 16384 paths) for all 15 new_bgrich experiments.
# 2 waves × 8 GPUs; ~5.5h per wave with contention = ~11h total overnight.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
LOG=logs/exhaustive_all.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
log "========== Exhaustive search on 15 new_bgrich experiments =========="

EXPS=(bgrich_globe_orrery bgrich_telescope_microscope bgrich_camera_binoculars \
      bgrich_pocketwatch_compass bgrich_teapot_samovar bgrich_chess_checkers \
      bgrich_violin_cello bgrich_guitar_banjo bgrich_lantern_candelabra \
      bgrich_wine_whiskey bgrich_revolver_flintlock bgrich_sewing_typewriter \
      bgrich_piano_harp bgrich_crown_tiara bgrich_skull_hourglass)

OUT_BASE="analysis/reinforce_analysis/exhaustive"         # NFS2 — metadata + top-K
IMG_BASE="/home/jovyan/shares/SR006.nfs3/svgrozny/exhaustive_new_bgrich"  # NFS3 — 16384 JPGs per exp
mkdir -p "$OUT_BASE" "$IMG_BASE"

run_wave() {
    local NAMES=("$@")
    local GPU=0
    for name in "${NAMES[@]}"; do
        local exp_dir="analysis/reinforce_analysis/new_bgrich/${name}"
        local out_dir="${OUT_BASE}/${name}"
        local img_dir="${IMG_BASE}/${name}"
        mkdir -p "$out_dir" "$img_dir"
        log "  GPU ${GPU}: exhaustive ${name} (images → $img_dir)"
        nohup $PYTHON -u analysis/exhaustive_search.py \
            --exp_dir "$exp_dir" --out_dir "$out_dir" --gpu $GPU \
            --images_dir "$img_dir" --save_all_images --jpg_quality 85 \
            --n_bits 14 --batch_size 8 --top_k 16 --alpha 0.5 --resume \
            > "logs/exhaustive_${name}.log" 2>&1 &
        echo $! > /tmp/exh_${name}.pid
        GPU=$((GPU + 1))
    done
    for name in "${NAMES[@]}"; do
        local pid=$(cat /tmp/exh_${name}.pid 2>/dev/null)
        [ -n "$pid" ] && wait $pid 2>/dev/null || true
        log "    ${name} exhaustive finished (rc=$?)"
    done
}

log "=== Wave 1: 8 experiments ==="
run_wave "${EXPS[@]:0:8}"

log "=== Wave 2: 7 experiments ==="
run_wave "${EXPS[@]:8}"

log "========== Exhaustive all done =========="
