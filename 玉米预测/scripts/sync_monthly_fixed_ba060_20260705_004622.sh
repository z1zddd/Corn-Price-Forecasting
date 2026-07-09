#!/usr/bin/env bash
set -euo pipefail

REMOTE_DIR="/root/corn_spike_sota/outputs/monthly_fixed_ba060/20260705_004622"
LOCAL_DIR="/Users/keyizhan/Downloads/corn_spike_official_pool_results/monthly_fixed_ba060/20260705_004622"
HOST="root@connect.westd.seetacloud.com"
PORT="10348"

mkdir -p "$(dirname "$LOCAL_DIR")"

while true; do
  rm -rf "$LOCAL_DIR.tmp"
  if sshpass -e scp -q -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no -P "$PORT" -r "$HOST:$REMOTE_DIR" "$LOCAL_DIR.tmp"; then
    rm -rf "$LOCAL_DIR"
    mv "$LOCAL_DIR.tmp" "$LOCAL_DIR"
  else
    rm -rf "$LOCAL_DIR.tmp"
  fi

  if [ -f "$LOCAL_DIR/PROGRESS.txt" ] && grep -q '^status=complete' "$LOCAL_DIR/PROGRESS.txt"; then
    break
  fi
  sleep 60
done
