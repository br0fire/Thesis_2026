#!/bin/bash
# Clean v1 pipeline for 16 bgrich experiments, α=0.5 fixed, canonical source.
# Phase 0: generate canonical source+mask+target per experiment (16 serial-ish, ~15 min)
# Phase 1: exhaustive 2^14 with images saved as uint8 .npy on NFS3 + bg/fg arrays (~10-14h)
# Phase 2: fast REINFORCE using lookup (~seconds)
# Phase 5: plots
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project
source scripts/clean_v1_prompts.sh

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1

NFS3_ROOT="/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1"
ROOT_CANON="${NFS3_ROOT}/canonical"
ROOT_EXH="${NFS3_ROOT}/exhaustive"
ROOT_IMG="${NFS3_ROOT}/exhaustive_images"
ROOT_REIN="${NFS3_ROOT}/reinforce_a05"
mkdir -p "$ROOT_CANON" "$ROOT_EXH" "$ROOT_IMG" "$ROOT_REIN" logs

LOG=logs/clean_v1_overnight.log
log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

log "========== clean_v1 pipeline =========="
log "  16 experiments, α=0.5, n_bits=14"
log "  Canonical: $ROOT_CANON"
log "  Exhaustive metadata: $ROOT_EXH"
log "  Exhaustive images: $ROOT_IMG (NFS3)"
log "  REINFORCE: $ROOT_REIN"

# ════════════════════════════════════════════════════════
# Phase 0: generate canonical source+mask+target per experiment
# 8 GPUs parallel, single wave (8 exps).
# ════════════════════════════════════════════════════════
run_canonical_wave() {
    local NAMES=("$@")
    local GPU=0
    for name in "${NAMES[@]}"; do
        local outdir="${ROOT_CANON}/${name}"
        if [ -f "$outdir/source.pt" ] && [ -f "$outdir/bg_mask.npy" ]; then
            log "  ${name}: canonical already exists, skip"
            continue
        fi
        mkdir -p "$outdir"
        log "  GPU ${GPU}: canonical ${name}"
        nohup $PYTHON -u generation/generate_canonical.py \
            --source_prompt "${SRC[$name]}" \
            --target_prompt "${TGT[$name]}" \
            --seg_prompt    "${SEG[$name]}" \
            --output_dir    "$outdir" \
            --gpu $GPU --seed 42 \
            > "logs/clean_canon_${name}.log" 2>&1 &
        echo $! > /tmp/canon_${name}.pid
        GPU=$((GPU + 1))
    done
    for name in "${NAMES[@]}"; do
        local pid=$(cat /tmp/canon_${name}.pid 2>/dev/null)
        [ -n "$pid" ] && wait $pid 2>/dev/null || true
    done
}

log "=== Phase 0: canonical (8 exps) ==="
run_canonical_wave "${EXPS[@]:0:8}"

# ════════════════════════════════════════════════════════
# Phase 1: exhaustive — save all 16384 images as uint8 .npy + bg/fg arrays
# 8 parallel, single wave on first 8 experiments. ~6-14h total.
# ════════════════════════════════════════════════════════
run_exhaustive_wave() {
    local NAMES=("$@")
    local GPU=0
    for name in "${NAMES[@]}"; do
        local canon="${ROOT_CANON}/${name}"
        local outdir="${ROOT_EXH}/${name}"
        local imgs_npy="${ROOT_IMG}/${name}/all_images.npy"
        if [ -f "$outdir/bg_ssim.npy" ] && [ -f "$outdir/fg_clip.npy" ] && [ -f "$imgs_npy" ]; then
            # Only skip if arrays full (no NaNs at end)
            if $PYTHON -c "import numpy as np, sys; a = np.load('$outdir/bg_ssim.npy'); sys.exit(0 if np.isfinite(a).all() else 1)"; then
                log "  ${name}: exhaustive already complete, skip"
                continue
            fi
        fi
        mkdir -p "$outdir" "$(dirname "$imgs_npy")"
        log "  GPU ${GPU}: exhaustive ${name}"
        nohup $PYTHON -u generation/exhaustive_clean.py \
            --canonical_dir "$canon" \
            --out_dir       "$outdir" \
            --images_npy    "$imgs_npy" \
            --gpu $GPU --batch_size 16 --top_k 32 --alpha 0.5 --resume \
            > "logs/clean_exh_${name}.log" 2>&1 &
        echo $! > /tmp/exh_clean_${name}.pid
        GPU=$((GPU + 1))
    done
    for name in "${NAMES[@]}"; do
        local pid=$(cat /tmp/exh_clean_${name}.pid 2>/dev/null)
        [ -n "$pid" ] && wait $pid 2>/dev/null || true
        log "    ${name} exhaustive finished"
    done
}

# Only run 8 experiments (1 wave) for faster overnight
log "=== Phase 1: exhaustive (8 exps, batch=16) ==="
run_exhaustive_wave "${EXPS[@]:0:8}"

# ════════════════════════════════════════════════════════
# Phase 2: fast REINFORCE α=0.5 — pure lookup, no FLUX
# All 16 run in parallel (CPU-bound, minimal GPU use)
# ════════════════════════════════════════════════════════
log "=== Phase 2: fast REINFORCE α=0.5 (first 8 parallel on CPU) ==="
for name in "${EXPS[@]:0:8}"; do
    local_exh="${ROOT_EXH}/${name}"
    local_out="${ROOT_REIN}/${name}"
    local_img="${ROOT_IMG}/${name}/all_images.npy"
    mkdir -p "$local_out"
    nohup $PYTHON -u generation/fast_reinforce.py \
        --exhaustive_dir "$local_exh" \
        --output_dir     "$local_out" \
        --images_npy     "$local_img" \
        --n_bits 14 --num_episodes 80 --batch_size 8 \
        --lr 0.10 --alpha 0.5 --entropy_coeff 0.05 \
        --top_k 10 --seed 42 \
        > "logs/clean_rein_${name}.log" 2>&1 &
    echo $! > /tmp/rein_fast_${name}.pid
done
for name in "${EXPS[@]:0:8}"; do
    pid=$(cat /tmp/rein_fast_${name}.pid 2>/dev/null)
    [ -n "$pid" ] && wait $pid 2>/dev/null || true
    log "    ${name} fast REINFORCE finished"
done


# ════════════════════════════════════════════════════════
# Phase 5: plots + summary
# ════════════════════════════════════════════════════════
log "=== Phase 5: plots + summary ==="
$PYTHON analysis/clean_v1_plots.py 2>&1 | tee -a "$LOG"

log "========== clean_v1 pipeline complete =========="
