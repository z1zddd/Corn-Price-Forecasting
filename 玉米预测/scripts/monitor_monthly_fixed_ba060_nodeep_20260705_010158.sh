#!/usr/bin/env bash
set -euo pipefail

REMOTE_DIR="/root/corn_spike_sota/outputs/monthly_fixed_ba060_nodeep_fastfirst/20260705_010158"
LOCAL_PARENT="/Users/keyizhan/Downloads/corn_spike_official_pool_results/monthly_fixed_ba060_nodeep_fastfirst"
LOCAL_DIR="$LOCAL_PARENT/20260705_010158"
HOST="root@connect.westd.seetacloud.com"
PORT="10348"
MODEL_SCRIPT_DIR="${MODEL_SCRIPT_DIR:-/Users/keyizhan/Documents/时序/玉米预测/scripts}"

mkdir -p "$LOCAL_PARENT"

while true; do
  TMP_DIR="$LOCAL_PARENT/.20260705_010158.tmp.$$"
  rm -rf "$TMP_DIR"
  if sshpass -e scp -q -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no -P "$PORT" -r "$HOST:$REMOTE_DIR" "$TMP_DIR"; then
    rm -rf "$LOCAL_DIR"
    mv "$TMP_DIR" "$LOCAL_DIR"
    python3 "${MODEL_SCRIPT_DIR}/write_live_monthly_fixed_report.py" "$LOCAL_DIR" || true
  else
    rm -rf "$TMP_DIR"
  fi

  if [ -f "$LOCAL_DIR/PROGRESS.txt" ] && grep -q '^status=complete' "$LOCAL_DIR/PROGRESS.txt"; then
    break
  fi
  sleep 90
done
