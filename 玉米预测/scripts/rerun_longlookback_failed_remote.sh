#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/root/corn_spike_sota}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_SCRIPT_DIR="${MODEL_SCRIPT_DIR:-${ROOT_DIR}/玉米预测/scripts}"
CSV_PATH="${CSV_PATH:-${ROOT_DIR}/玉米预测/datasets/corn_monthly.csv}"
MODEL_LIST="${MODEL_LIST:-${MODEL_SCRIPT_DIR}/longlookback_2016fixed_50_plus_best_plus_keras_models.txt}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/longlookback_2016fixed_50_plus_keras_20260705_162158}"
RERUN_TIMEOUT_SECONDS="${RERUN_TIMEOUT_SECONDS:-21600}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
MAX_RERUN_PASSES="${MAX_RERUN_PASSES:-3}"

mkdir -p "${OUT_ROOT}/parts" "${OUT_ROOT}/logs" "${OUT_ROOT}/live"
cd "${ROOT_DIR}" || exit 1

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"

echo "rerun watcher started $(date --iso-8601=seconds)" | tee -a "${OUT_ROOT}/run.log"
while pgrep -f run_longlookback_remote_parts.sh >/dev/null; do
  echo "waiting for main runner $(date --iso-8601=seconds)" | tee -a "${OUT_ROOT}/run.log"
  sleep "${WAIT_SECONDS}"
done

latest_bad_models() {
  "${PYTHON_BIN}" - "${MODEL_LIST}" "${OUT_ROOT}/part_status.tsv" <<'PY'
from pathlib import Path
import csv
import sys

model_list = Path(sys.argv[1])
status_path = Path(sys.argv[2])
models = [model.strip() for model in model_list.read_text(encoding="utf-8").split(",") if model.strip()]
latest = {}
if status_path.exists():
    with status_path.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 3:
                latest[row[1]] = row[2]
bad_statuses = {"failed", "timeout", "running"}
for model in models:
    status = latest.get(model)
    if status is None or status in bad_statuses:
        print(model)
PY
}

for PASS in $(seq 1 "${MAX_RERUN_PASSES}"); do
  "${PYTHON_BIN}" "${MODEL_SCRIPT_DIR}/aggregate_longlookback_results.py" --root "${OUT_ROOT}" --out "${OUT_ROOT}/live" >> "${OUT_ROOT}/aggregate.log" 2>&1 || true
  mapfile -t BAD_MODELS < <(latest_bad_models)
  if [[ "${#BAD_MODELS[@]}" == "0" ]]; then
    echo "rerun watcher complete: no failed/timeout/missing models $(date --iso-8601=seconds)" | tee -a "${OUT_ROOT}/run.log"
    break
  fi

  echo "rerun pass ${PASS}/${MAX_RERUN_PASSES}: ${BAD_MODELS[*]}" | tee -a "${OUT_ROOT}/run.log"
  for MODEL in "${BAD_MODELS[@]}"; do
    PART_DIR="${OUT_ROOT}/parts/${MODEL}"
    LOG_PATH="${OUT_ROOT}/logs/${MODEL}_rerun_pass${PASS}.log"
    mkdir -p "${PART_DIR}"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${MODEL}" "running" "" "${PART_DIR}" >> "${OUT_ROOT}/part_status.tsv"

    timeout "${RERUN_TIMEOUT_SECONDS}" "${PYTHON_BIN}" "${MODEL_SCRIPT_DIR}/run_corn_monthly_spike_official_pool_two_heads.py" \
      --csv "${CSV_PATH}" \
      --out-dir "${PART_DIR}" \
      --origin-mode monthly \
      --threshold-mode validation \
      --test-start 2017-01-01 \
      --monthly-cutoff-lag 1 \
      --min-train 24 \
      --val-size 12 \
      --lookbacks 6,9,12 \
      --horizons 1,2 \
      --feature-sets no_news,with_news_precomputed_pca \
      --models "${MODEL}" \
      --heads cls,reg \
      --checkpoint \
      --resume-latest \
      --save-folds \
      --deep-epochs 6 \
      --deep-batch-size 16 \
      > "${LOG_PATH}" 2>&1
    RC=$?

    RUN_DIR="$(find "${PART_DIR}" -mindepth 1 -maxdepth 1 -type d -print | sort | tail -1)"
    if [[ "${RC}" == "0" ]]; then
      STATUS="done"
    elif [[ "${RC}" == "124" ]]; then
      STATUS="timeout"
    else
      STATUS="failed"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${MODEL}" "${STATUS}" "${RC}" "${RUN_DIR}" >> "${OUT_ROOT}/part_status.tsv"
    echo "rerun pass ${PASS}: ${MODEL} ${STATUS} rc=${RC}" | tee -a "${OUT_ROOT}/run.log"
    "${PYTHON_BIN}" "${MODEL_SCRIPT_DIR}/aggregate_longlookback_results.py" --root "${OUT_ROOT}" --out "${OUT_ROOT}/live" >> "${OUT_ROOT}/aggregate.log" 2>&1 || true
  done
done

"${PYTHON_BIN}" "${MODEL_SCRIPT_DIR}/aggregate_longlookback_results.py" --root "${OUT_ROOT}" --out "${OUT_ROOT}/live" >> "${OUT_ROOT}/aggregate.log" 2>&1 || true
echo "rerun watcher stopped $(date --iso-8601=seconds)" | tee -a "${OUT_ROOT}/run.log"
