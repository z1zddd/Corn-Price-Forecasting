#!/usr/bin/env zsh

REMOTE_HOST="root@connect.westd.seetacloud.com"
REMOTE_PORT="10348"
REMOTE_DIR="/root/corn_spike_sota/outputs/official_pool_full/20260704_175056"
LOCAL_DIR="/Users/keyizhan/Downloads/corn_spike_official_pool_results/20260704_175056"
LOG_FILE="/Users/keyizhan/Downloads/corn_spike_official_pool_results/sync_20260704_175056.log"

mkdir -p "$LOCAL_DIR"

write_status() {
  local run_status="$1"
  {
    print "remote=$REMOTE_DIR"
    print "local=$LOCAL_DIR"
    print "status=$run_status"
    print "last_checked=$(date)"
  } > "$LOCAL_DIR/STATUS.txt"
}

write_status "running"

while true; do
  print "checking $(date)"
  if sshpass -e ssh \
    -o StrictHostKeyChecking=no \
    -o PreferredAuthentications=password \
    -o PubkeyAuthentication=no \
    -p "$REMOTE_PORT" "$REMOTE_HOST" \
    "test -f '$REMOTE_DIR/summary_metrics.csv'"; then
    rm -rf "$LOCAL_DIR.tmp"
    mkdir -p "$LOCAL_DIR.tmp"
    sshpass -e scp \
      -o StrictHostKeyChecking=no \
      -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no \
      -P "$REMOTE_PORT" \
      -r "$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_DIR.tmp/"
    rm -rf "$LOCAL_DIR"
    mv "$LOCAL_DIR.tmp/$(basename "$REMOTE_DIR")" "$LOCAL_DIR"
    write_status "complete"
    exit 0
  fi

  write_status "running"
  sleep 60
done >> "$LOG_FILE" 2>&1
