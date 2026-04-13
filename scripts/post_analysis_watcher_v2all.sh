#!/bin/bash
# Post-analysis watcher for the 9 remaining v2 experiments.
# Waits until all 9 v2 reruns finish, then runs summary + insights + v1_vs_v2.
set -e
cd /home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project

PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
export PYTHONUNBUFFERED=1

NFS3=/home/jovyan/shares/SR006.nfs3/svgrozny
STATUS_LOG=logs/post_analysis_v2all_status.log
RESULT_LOG=logs/post_analysis_v2all_done.log

PENDING=(
    "penguin_flamingo_v2" "cake_books_v2" "lighthouse_castle_v2" "violin_guitar_v2"
    "horse_v2" "room_v2" "snow_volcano_v2" "butterfly_hummingbird_v2" "sail_pirate_v2"
)

MAX_WAIT_MIN=720  # 12 hours
POLL_INTERVAL=120

echo "[$(date)] post_analysis_v2all_watcher started" | tee -a $STATUS_LOG
echo "  Waiting for ${#PENDING[@]} experiments: ${PENDING[*]}" | tee -a $STATUS_LOG

start_time=$(date +%s)

while true; do
    now=$(date +%s)
    elapsed_min=$(( (now - start_time) / 60 ))

    if [ "$elapsed_min" -ge "$MAX_WAIT_MIN" ]; then
        echo "[$(date)] TIMEOUT after ${MAX_WAIT_MIN} min, proceeding with analysis anyway" | tee -a $STATUS_LOG
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
    if [ ${#missing[@]} -gt 0 ] && [ ${#missing[@]} -le 9 ]; then
        echo "  still waiting: ${missing[*]}" | tee -a $STATUS_LOG
    fi

    if [ "$done" = "${#PENDING[@]}" ]; then
        echo "[$(date)] All v2 experiments complete!" | tee -a $STATUS_LOG
        break
    fi

    sleep $POLL_INTERVAL
done

sleep 60  # allow final file writes to flush

echo "" | tee -a $STATUS_LOG
echo "[$(date)] Running reinforce_summary.py (all 26 experiments)..." | tee -a $STATUS_LOG
$PYTHON analysis/reinforce_summary.py > logs/post_analysis_v2all_summary.log 2>&1 \
    && echo "  summary: OK" | tee -a $STATUS_LOG \
    || echo "  summary: FAILED" | tee -a $STATUS_LOG

echo "[$(date)] Running reinforce_insights.py..." | tee -a $STATUS_LOG
$PYTHON analysis/reinforce_insights.py > logs/post_analysis_v2all_insights.log 2>&1 \
    && echo "  insights: OK" | tee -a $STATUS_LOG \
    || echo "  insights: FAILED" | tee -a $STATUS_LOG

echo "[$(date)] Running reinforce_v1_vs_v2.py (13 pairs)..." | tee -a $STATUS_LOG
$PYTHON analysis/reinforce_v1_vs_v2.py > logs/post_analysis_v2all_v1_vs_v2.log 2>&1 \
    && echo "  v1_vs_v2: OK" | tee -a $STATUS_LOG \
    || echo "  v1_vs_v2: FAILED" | tee -a $STATUS_LOG

echo "" | tee -a $STATUS_LOG
echo "[$(date)] DONE." | tee -a $STATUS_LOG
echo "finished $(date)" > $RESULT_LOG
