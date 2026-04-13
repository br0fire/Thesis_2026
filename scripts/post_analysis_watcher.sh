#!/bin/bash
# Post-analysis watcher: waits for all REINFORCE experiments to finish,
# then runs summary + insights + generates comparison report.
# Designed to run via nohup and survive user disconnect.
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

NFS3=/home/jovyan/shares/SR006.nfs3/svgrozny
STATUS_LOG=logs/post_analysis_status.log
RESULT_LOG=logs/post_analysis_done.log

# Already done (wave 1 from before) + pending experiments
# Wave 2 (horse, room, snow_volcano, butterfly_hummingbird, sail_pirate) — running
# V2 reruns (catdog_v2, car_taxi_v2, sunflower_lavender_v2, chair_throne_v2) — running
PENDING_EXPERIMENTS=(
    "horse" "room" "snow_volcano" "butterfly_hummingbird" "sail_pirate"
    "catdog_v2" "car_taxi_v2" "sunflower_lavender_v2" "chair_throne_v2"
)

ORCHESTRATOR_PIDS=(1861002 1941663)  # wave2 orchestrator + v2 rerun orchestrator

# Max wait: 24 hours
MAX_WAIT_MIN=1440
POLL_INTERVAL=120  # seconds

echo "[$(date)] post_analysis_watcher started" | tee -a $STATUS_LOG
echo "  Waiting for ${#PENDING_EXPERIMENTS[@]} experiments: ${PENDING_EXPERIMENTS[*]}" | tee -a $STATUS_LOG
echo "  Orchestrator PIDs: ${ORCHESTRATOR_PIDS[*]}" | tee -a $STATUS_LOG

start_time=$(date +%s)

while true; do
    now=$(date +%s)
    elapsed_min=$(( (now - start_time) / 60 ))

    if [ "$elapsed_min" -ge "$MAX_WAIT_MIN" ]; then
        echo "[$(date)] TIMEOUT after ${MAX_WAIT_MIN} min, proceeding with analysis anyway" | tee -a $STATUS_LOG
        break
    fi

    # Check how many experiments are done
    done=0
    missing=()
    for name in "${PENDING_EXPERIMENTS[@]}"; do
        if [ -f "$NFS3/reinforce_${name}/reinforce_result.pt" ]; then
            done=$((done + 1))
        else
            missing+=("$name")
        fi
    done

    # Check if orchestrators still alive
    orch_alive=0
    for pid in "${ORCHESTRATOR_PIDS[@]}"; do
        if ps -p $pid >/dev/null 2>&1; then
            orch_alive=$((orch_alive + 1))
        fi
    done

    echo "[$(date)] progress: ${done}/${#PENDING_EXPERIMENTS[@]} done, ${orch_alive}/${#ORCHESTRATOR_PIDS[@]} orchestrators alive, elapsed ${elapsed_min}m" | tee -a $STATUS_LOG
    if [ ${#missing[@]} -gt 0 ] && [ ${#missing[@]} -le 5 ]; then
        echo "  still waiting: ${missing[*]}" | tee -a $STATUS_LOG
    fi

    # Done when all results exist
    if [ "$done" = "${#PENDING_EXPERIMENTS[@]}" ]; then
        echo "[$(date)] All experiments complete!" | tee -a $STATUS_LOG
        break
    fi

    # Also done if both orchestrators exited — no more experiments will launch
    if [ "$orch_alive" = "0" ]; then
        echo "[$(date)] Both orchestrators exited; proceeding with ${done} completed experiments" | tee -a $STATUS_LOG
        break
    fi

    sleep $POLL_INTERVAL
done

# Give a bit of time for final file writes (checkpoints, top images) to flush
echo "[$(date)] Waiting 60s for final file flushes..." | tee -a $STATUS_LOG
sleep 60

# ── Run analysis scripts ──
echo "" | tee -a $STATUS_LOG
echo "[$(date)] Running reinforce_summary.py..." | tee -a $STATUS_LOG
$PYTHON analysis/reinforce_summary.py > logs/post_analysis_summary.log 2>&1 \
    && echo "  summary: OK" | tee -a $STATUS_LOG \
    || echo "  summary: FAILED" | tee -a $STATUS_LOG

echo "[$(date)] Running reinforce_insights.py..." | tee -a $STATUS_LOG
$PYTHON analysis/reinforce_insights.py > logs/post_analysis_insights.log 2>&1 \
    && echo "  insights: OK" | tee -a $STATUS_LOG \
    || echo "  insights: FAILED" | tee -a $STATUS_LOG

# ── Build v1 vs v2 comparison ──
echo "[$(date)] Running reward comparison (v1 vs v2)..." | tee -a $STATUS_LOG
$PYTHON analysis/reinforce_v1_vs_v2.py > logs/post_analysis_v1_vs_v2.log 2>&1 \
    && echo "  v1_vs_v2: OK" | tee -a $STATUS_LOG \
    || echo "  v1_vs_v2: FAILED or not present" | tee -a $STATUS_LOG

# ── Final report ──
OUT_DIR=analysis/reinforce_analysis
echo "" | tee -a $STATUS_LOG
echo "[$(date)] DONE. Outputs:" | tee -a $STATUS_LOG
echo "  $OUT_DIR/summary.csv" | tee -a $STATUS_LOG
echo "  $OUT_DIR/training_curves.png" | tee -a $STATUS_LOG
echo "  $OUT_DIR/learned_probs.png" | tee -a $STATUS_LOG
echo "  $OUT_DIR/top_images_grids/*.jpg" | tee -a $STATUS_LOG
echo "  logs/post_analysis_summary.log (textual summary)" | tee -a $STATUS_LOG
echo "  logs/post_analysis_insights.log (training dynamics)" | tee -a $STATUS_LOG

# Touch a sentinel file so the user knows it's done
echo "finished $(date)" > $RESULT_LOG
echo "[$(date)] Post-analysis complete. Sentinel: $RESULT_LOG" | tee -a $STATUS_LOG
