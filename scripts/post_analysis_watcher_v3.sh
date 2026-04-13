#!/bin/bash
# Post-analysis watcher for the v3 reruns (SigLIP 2 + simplified delta reward).
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

NFS3=/home/jovyan/shares/SR006.nfs3/svgrozny
STATUS_LOG=logs/post_analysis_v3_status.log
RESULT_LOG=logs/post_analysis_v3_done.log

PENDING=(
    "catdog_v3" "car_taxi_v3" "sunflower_lavender_v3" "chair_throne_v3"
    "penguin_flamingo_v3" "cake_books_v3" "lighthouse_castle_v3" "violin_guitar_v3"
    "horse_v3" "room_v3" "snow_volcano_v3" "butterfly_hummingbird_v3" "sail_pirate_v3"
    "bgrich_teapot_globe_v3" "bgrich_candle_crystal_v3" "bgrich_typewriter_laptop_v3"
)

MAX_WAIT_MIN=720
POLL_INTERVAL=120

echo "[$(date)] post_analysis_v3_watcher started" | tee -a $STATUS_LOG
echo "  Waiting for ${#PENDING[@]} experiments" | tee -a $STATUS_LOG

start_time=$(date +%s)
while true; do
    now=$(date +%s)
    elapsed_min=$(( (now - start_time) / 60 ))
    if [ "$elapsed_min" -ge "$MAX_WAIT_MIN" ]; then
        echo "[$(date)] TIMEOUT after ${MAX_WAIT_MIN} min" | tee -a $STATUS_LOG
        break
    fi

    done=0
    missing=()
    for name in "${PENDING[@]}"; do
        if [ -f "$NFS3/reinforce_${name}/reinforce_result.pt" ]; then
            done=$((done + 1))
        else
            missing+=("$name")
        fi
    done

    echo "[$(date)] progress: ${done}/${#PENDING[@]} done, elapsed ${elapsed_min}m" | tee -a $STATUS_LOG
    if [ ${#missing[@]} -gt 0 ] && [ ${#missing[@]} -le 16 ]; then
        echo "  still waiting: ${missing[*]}" | tee -a $STATUS_LOG
    fi

    if [ "$done" = "${#PENDING[@]}" ]; then
        echo "[$(date)] All v3 experiments complete!" | tee -a $STATUS_LOG
        break
    fi
    sleep $POLL_INTERVAL
done

sleep 60

echo "" | tee -a $STATUS_LOG
echo "[$(date)] Running reinforce_summary.py..." | tee -a $STATUS_LOG
$PYTHON analysis/reinforce_summary.py > logs/post_analysis_v3_summary.log 2>&1 \
    && echo "  summary: OK" | tee -a $STATUS_LOG \
    || echo "  summary: FAILED" | tee -a $STATUS_LOG

echo "[$(date)] Running reinforce_insights.py..." | tee -a $STATUS_LOG
$PYTHON analysis/reinforce_insights.py > logs/post_analysis_v3_insights.log 2>&1 \
    && echo "  insights: OK" | tee -a $STATUS_LOG \
    || echo "  insights: FAILED" | tee -a $STATUS_LOG

echo "[$(date)] DONE." | tee -a $STATUS_LOG
echo "finished $(date)" > $RESULT_LOG
