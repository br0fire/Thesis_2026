#!/bin/bash
# Test bgrich prior generalization on 8 DIVERSE (non-bgrich) experiments,
# 2 per category (object swap / landscape / portrait / style-scene).
# 3 methods per exp: random | reinforce | reinforce_prior.  α=0.5, 14-bit, budget=120.
# 8 exps on 8 GPUs in ONE wave — each GPU runs 3 methods sequentially.  ETA ~45 min.
set +e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project
source scripts/diverse_prompts.sh

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export PYTHONUNBUFFERED=1
mkdir -p logs

NFS3="/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1"
CANON_ROOT="${NFS3}/diverse_canonical"
OUT_ROOT="${NFS3}/diverse"
PRIOR="${NFS3}/bgrich_prior.npy"
mkdir -p "$CANON_ROOT" "$OUT_ROOT"

LOG=logs/diverse_overnight.log
log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

# 8 diverse experiments: 2 per category
EXPS_8=(cat_dog mug_teacup \
        beach_mountain forest_desert \
        young_old_woman smile_frown \
        photo_painting day_night_city)

log "========== diverse overnight (8 exps, 1 wave) =========="
log "  8 experiments × 3 methods (random, reinforce, reinforce_prior)"
log "  α=0.5, n_bits=14, budget=120"

# ═══════════════════════════════════════════
# Phase 0: canonical (1 wave of 8)
# ═══════════════════════════════════════════
GPU=0
for name in "${EXPS_8[@]}"; do
    outdir="${CANON_ROOT}/${name}"
    if [ -f "$outdir/source.pt" ] && [ -f "$outdir/bg_mask.npy" ]; then
        log "  ${name}: canonical exists, skip"
        continue
    fi
    mkdir -p "$outdir"
    log "  GPU ${GPU}: canonical ${name}"
    nohup $PYTHON -u generation/generate_canonical.py \
        --source_prompt "${DIV_SRC[$name]}" \
        --target_prompt "${DIV_TGT[$name]}" \
        --seg_prompt    "${DIV_SEG[$name]}" \
        --output_dir    "$outdir" \
        --gpu $GPU --seed 42 \
        > "logs/div_canon_${name}.log" 2>&1 &
    echo $! > /tmp/div_canon_${name}.pid
    GPU=$((GPU + 1))
done
for name in "${EXPS_8[@]}"; do
    pid=$(cat /tmp/div_canon_${name}.pid 2>/dev/null)
    [ -n "$pid" ] && wait $pid 2>/dev/null || true
done
log "=== Canonical phase done ==="

# ═══════════════════════════════════════════
# Methods phase: 8 exps in parallel, each GPU runs 3 methods serially on ONE exp
# ═══════════════════════════════════════════
GPU=0
for name in "${EXPS_8[@]}"; do
    (
        MYGPU=$GPU
        canon="${CANON_ROOT}/${name}"
        src_pt="$canon/source.pt"
        mask_npy="$canon/bg_mask.npy"

        # ── 1) random baseline (120 samples) ──
        rand_dir="${OUT_ROOT}/${name}/random"
        mkdir -p "$rand_dir"
        cp "${canon}/prompts.txt"  "$rand_dir/" 2>/dev/null
        cp "$src_pt"   "$rand_dir/" 2>/dev/null
        cp "$mask_npy" "$rand_dir/" 2>/dev/null
        $PYTHON -u analysis/compute_random_baseline.py \
            --exp_dir "$rand_dir" --gpu $MYGPU \
            --n_samples 120 --n_bits 14 --alpha 0.5 \
            > "logs/div_${name}_random.log" 2>&1

        # ── 2) REINFORCE no prior ──
        $PYTHON -u generation/reinforce_search.py \
            --source_prompt "${DIV_SRC[$name]}" \
            --target_prompt "${DIV_TGT[$name]}" \
            --seg_prompt    "${DIV_SEG[$name]}" \
            --output_dir    "${OUT_ROOT}/${name}/reinforce" \
            --source_tensor_pt "$src_pt" --bg_mask_npy "$mask_npy" \
            --gpu $MYGPU --n_bits 14 --batch_size 8 --top_k 5 --log_interval 5 \
            --num_episodes 15 --min_episodes 15 --plateau_window 0 --entropy_stop 0 \
            --lr 0.10 --alpha 0.5 --entropy_coeff 0.05 \
            > "logs/div_${name}_reinforce.log" 2>&1

        # ── 3) REINFORCE with bgrich prior ──
        $PYTHON -u generation/reinforce_search.py \
            --source_prompt "${DIV_SRC[$name]}" \
            --target_prompt "${DIV_TGT[$name]}" \
            --seg_prompt    "${DIV_SEG[$name]}" \
            --output_dir    "${OUT_ROOT}/${name}/reinforce_prior" \
            --source_tensor_pt "$src_pt" --bg_mask_npy "$mask_npy" \
            --init_probs_npy "$PRIOR" \
            --gpu $MYGPU --n_bits 14 --batch_size 8 --top_k 5 --log_interval 5 \
            --num_episodes 15 --min_episodes 15 --plateau_window 0 --entropy_stop 0 \
            --lr 0.10 --alpha 0.5 --entropy_coeff 0.05 \
            > "logs/div_${name}_reinforce_prior.log" 2>&1
    ) &
    echo $! > /tmp/div_methods_${name}.pid
    log "  GPU ${GPU}: methods ${name} launched"
    GPU=$((GPU + 1))
done

for name in "${EXPS_8[@]}"; do
    pid=$(cat /tmp/div_methods_${name}.pid 2>/dev/null)
    [ -n "$pid" ] && wait $pid 2>/dev/null || true
    log "    ${name} all methods done"
done

log "========== diverse overnight done =========="
