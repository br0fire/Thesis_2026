#!/bin/bash
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

NFS3=/home/jovyan/shares/SR006.nfs3/svgrozny
STATUS_LOG=logs/watch_8_v5_status.log
RESULT_LOG=logs/watch_8_v5_done.log

PENDING=(
    "catdog_v5" "car_taxi_v5" "sunflower_lavender_v5" "chair_throne_v5"
    "penguin_flamingo_v5" "violin_guitar_v5"
    "bgrich_teapot_globe_v5" "bgrich_typewriter_laptop_v5"
)
MAX_WAIT_MIN=240  # expect ~1.5h with early stopping
POLL_INTERVAL=60  # faster polling, runs should be short

echo "[$(date)] v5 watcher started" | tee -a $STATUS_LOG
start_time=$(date +%s)
while true; do
    elapsed_min=$(( ($(date +%s) - start_time) / 60 ))
    [ "$elapsed_min" -ge "$MAX_WAIT_MIN" ] && { echo "[$(date)] TIMEOUT" | tee -a $STATUS_LOG; break; }

    done=0
    missing=()
    for name in "${PENDING[@]}"; do
        if [ -f "$NFS3/reinforce_${name}/reinforce_result.pt" ]; then
            done=$((done + 1))
        else
            missing+=("$name")
        fi
    done

    echo "[$(date)] ${done}/${#PENDING[@]} done, elapsed ${elapsed_min}m" | tee -a $STATUS_LOG
    [ ${#missing[@]} -gt 0 ] && echo "  waiting: ${missing[*]}" | tee -a $STATUS_LOG
    [ "$done" = "${#PENDING[@]}" ] && { echo "[$(date)] all complete" | tee -a $STATUS_LOG; break; }
    sleep $POLL_INTERVAL
done

sleep 30
$PYTHON analysis/reinforce_summary.py > logs/watch_8_v5_summary.log 2>&1 && echo "  summary OK" | tee -a $STATUS_LOG
$PYTHON analysis/reinforce_insights.py > logs/watch_8_v5_insights.log 2>&1 && echo "  insights OK" | tee -a $STATUS_LOG
echo "finished $(date)" > $RESULT_LOG
echo "[$(date)] DONE" | tee -a $STATUS_LOG
