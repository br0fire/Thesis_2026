#!/bin/bash
# After CEM fix driver finishes: compute all-ones rewards for new_bgrich
# and rebuild the training curves plot + visual grid.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
LOG=logs/post_cem_followup.log
mkdir -p logs

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
log "========== Post-CEM followup waiting =========="

while pgrep -f "scripts/cem_old_experiments_fix.sh" > /dev/null \
   || pgrep -f "generation/cem_search.py" > /dev/null; do
    log "Waiting for CEM fix..."
    sleep 120
done
log "CEM fix done — running followups"

# 1. Compute all-ones rewards for new_bgrich
log "Computing all-ones rewards for new_bgrich..."
$PYTHON analysis/compute_all_ones_reward.py --gpu 0 --dir analysis/reinforce_analysis/new_bgrich \
    2>&1 | tee -a "$LOG"

# 1b. Compute random baseline rewards for new_bgrich (640 samples each, parallel on 8 GPUs)
log "Computing random-path baseline for 15 new_bgrich experiments..."
GPU=0
pids=()
for name in bgrich_globe_orrery bgrich_telescope_microscope bgrich_camera_binoculars \
            bgrich_pocketwatch_compass bgrich_teapot_samovar bgrich_chess_checkers \
            bgrich_violin_cello bgrich_guitar_banjo bgrich_lantern_candelabra \
            bgrich_wine_whiskey bgrich_revolver_flintlock bgrich_sewing_typewriter \
            bgrich_piano_harp bgrich_crown_tiara bgrich_skull_hourglass; do
    exp_dir="analysis/reinforce_analysis/new_bgrich/${name}"
    [ -d "$exp_dir" ] || continue
    nohup $PYTHON -u analysis/compute_random_baseline.py \
        --exp_dir "$exp_dir" --gpu $GPU --n_samples 640 \
        > "logs/random_${name}.log" 2>&1 &
    pids+=($!)
    GPU=$(( (GPU + 1) % 8 ))
    if [ $GPU -eq 0 ]; then
        for p in "${pids[@]}"; do wait $p 2>/dev/null; done
        pids=()
    fi
done
for p in "${pids[@]}"; do wait $p 2>/dev/null; done
log "  Random baseline done"

# 2. Regenerate new_bgrich training curves with baseline
log "Regenerating new_bgrich training curves..."
$PYTHON analysis/new_bgrich_curves.py 2>&1 | tee -a "$LOG"

# 3. Regenerate full visual comparison grid (now all 30 have valid CEM)
log "Regenerating full visual grid..."
$PYTHON analysis/visual_comparison_grid.py 2>&1 | tee -a "$LOG"

# 4. Regenerate CEM training curves (now with 30 experiments at all budgets)
log "Regenerating CEM training curves..."
$PYTHON analysis/cem_training_curves.py 2>&1 | tee -a "$LOG"

# 5. Regenerate sample efficiency plot
log "Regenerating sample efficiency plot..."
$PYTHON analysis/sample_efficiency_plot.py 2>&1 | tee -a "$LOG"

log "========== Post-CEM followup done =========="
