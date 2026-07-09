#!/usr/bin/env bash
set -u

OUT="${OUT:-/root/corn_spike_sota/outputs/longlookback_2016fixed_50_plus_keras_20260705_162158}"
LOCAL_OUT="${LOCAL_OUT:-/Users/keyizhan/Downloads/corn_spike_official_pool_results/longlookback_2016fixed_50_plus_keras_20260705_162158}"
REMOTE="${REMOTE:-root@connect.westd.seetacloud.com}"
PORT="${PORT:-10348}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/corn_spike_sota}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-${REMOTE_ROOT}/玉米预测/scripts}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
MAX_STOP_MISSES="${MAX_STOP_MISSES:-3}"
STOP_MISSES=0
REMOTE_ACTIVE_PATTERN="run_longlookback_remote_parts.sh|rerun_longlookback_failed_remote.sh|run_longlookback_remote_parallel_remaining.sh|run_corn_monthly_spike_official_pool_two_heads.py"
LOCK_DIR="${LOCAL_OUT}/.sync.lock"
LOCK_HELD=0

mkdir -p "${LOCAL_OUT}"
cleanup_lock() {
  if (( LOCK_HELD == 1 )); then
    rmdir "${LOCK_DIR}" >/dev/null 2>&1 || true
    LOCK_HELD=0
  fi
}
trap cleanup_lock EXIT INT TERM

while true; do
  date "+%Y-%m-%d %H:%M:%S sync begin"

  if ! mkdir "${LOCK_DIR}" >/dev/null 2>&1; then
    date "+%Y-%m-%d %H:%M:%S another sync is active; skipping this cycle"
    sleep "${INTERVAL_SECONDS}"
    continue
  fi
  LOCK_HELD=1

  sshpass -e ssh -o StrictHostKeyChecking=no -p "${PORT}" "${REMOTE}" \
    "cd ${REMOTE_ROOT} && /root/miniconda3/bin/python ${REMOTE_SCRIPT_DIR}/aggregate_longlookback_results.py --root ${OUT} --out ${OUT}/live" || true

  RSYNC_OK=0
  for RSYNC_ATTEMPT in 1 2 3; do
    if sshpass -e rsync -az --delete --delay-updates --partial \
      --exclude="sync.pid" \
      --exclude="sync.log" \
      --exclude="screen_sync.log" \
      --exclude="live/all_rolling_predictions.csv" \
      --exclude="live/all_folds.csv" \
      -e "ssh -o StrictHostKeyChecking=no -p ${PORT}" \
      "${REMOTE}:${OUT}/" "${LOCAL_OUT}/"; then
      RSYNC_OK=1
      break
    fi
    date "+%Y-%m-%d %H:%M:%S rsync retry ${RSYNC_ATTEMPT}/3"
    sleep 5
  done
  if (( RSYNC_OK == 0 )); then
    date "+%Y-%m-%d %H:%M:%S rsync failed after retries; will retry next cycle"
  fi

  date "+%Y-%m-%d %H:%M:%S sync end"
  cleanup_lock

  if sshpass -e ssh -o StrictHostKeyChecking=no -p "${PORT}" "${REMOTE}" \
    "pgrep -f '${REMOTE_ACTIVE_PATTERN}' >/dev/null"; then
    STOP_MISSES=0
  else
    STOP_MISSES=$((STOP_MISSES + 1))
    date "+%Y-%m-%d %H:%M:%S remote runner check missed ${STOP_MISSES}/${MAX_STOP_MISSES}"
    if (( STOP_MISSES >= MAX_STOP_MISSES )); then
      date "+%Y-%m-%d %H:%M:%S remote runner stopped; starting final full sync"
      if mkdir "${LOCK_DIR}" >/dev/null 2>&1; then
        LOCK_HELD=1
      else
        date "+%Y-%m-%d %H:%M:%S another sync is active before final sync; waiting"
        sleep 10
        if mkdir "${LOCK_DIR}" >/dev/null 2>&1; then
          LOCK_HELD=1
        fi
      fi
      sshpass -e rsync -az --delete --delay-updates --partial \
        --exclude="sync.pid" \
        --exclude="sync.log" \
        --exclude="screen_sync.log" \
        -e "ssh -o StrictHostKeyChecking=no -p ${PORT}" \
        "${REMOTE}:${OUT}/" "${LOCAL_OUT}/" || true
      cleanup_lock
      date "+%Y-%m-%d %H:%M:%S remote runner stopped; final sync complete"
      break
    fi
  fi

  sleep "${INTERVAL_SECONDS}"
done
