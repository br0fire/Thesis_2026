#!/bin/bash
# Retry wave 1 experiments that were killed during filename-format change.
# Runs after current exhaustive driver finishes.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
LOG=logs/exhaustive_wave1_retry.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
log "========== Wave 1 retry waiting =========="

# Wait for current exhaustive driver to finish
while pgrep -f "scripts/exhaustive_all_new_bgrich.sh" > /dev/null \
   || pgrep -f "exhaustive_search.py" > /dev/null; do
    log "Waiting for wave 2..."
    sleep 120
done
log "Wave 2 done — launching wave 1"

OUT_BASE="analysis/reinforce_analysis/exhaustive"
IMG_BASE="/home/jovyan/shares/SR006.nfs3/svgrozny/exhaustive_new_bgrich"

EXPS=(bgrich_globe_orrery bgrich_telescope_microscope bgrich_camera_binoculars \
      bgrich_pocketwatch_compass bgrich_teapot_samovar bgrich_chess_checkers \
      bgrich_violin_cello bgrich_guitar_banjo)

GPU=0
for name in "${EXPS[@]}"; do
    exp_dir="analysis/reinforce_analysis/new_bgrich/${name}"
    out_dir="${OUT_BASE}/${name}"
    img_dir="${IMG_BASE}/${name}"
    mkdir -p "$out_dir" "$img_dir"
    log "  GPU ${GPU}: exhaustive ${name}"
    nohup $PYTHON -u analysis/exhaustive_search.py \
        --exp_dir "$exp_dir" --out_dir "$out_dir" --gpu $GPU \
        --images_dir "$img_dir" --save_all_images --jpg_quality 85 \
        --n_bits 14 --batch_size 8 --top_k 16 --alpha 0.5 --resume \
        > "logs/exhaustive_${name}.log" 2>&1 &
    echo $! > /tmp/exh_${name}.pid
    GPU=$((GPU + 1))
done

for name in "${EXPS[@]}"; do
    pid=$(cat /tmp/exh_${name}.pid 2>/dev/null)
    [ -n "$pid" ] && wait $pid 2>/dev/null || true
    log "    ${name} finished"
done

log "========== Wave 1 retry done =========="
