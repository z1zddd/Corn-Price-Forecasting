#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/root/corn_spike_sota}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_SCRIPT_DIR="${MODEL_SCRIPT_DIR:-${ROOT_DIR}/玉米预测/scripts}"
CSV_PATH="${CSV_PATH:-${ROOT_DIR}/玉米预测/datasets/corn_monthly.csv}"
MODEL_LIST="${MODEL_LIST:-${MODEL_SCRIPT_DIR}/longlookback_2016fixed_50_plus_best_plus_keras_models.txt}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/longlookback_2016fixed_50_plus_best}"
MODEL_TIMEOUT_SECONDS="${MODEL_TIMEOUT_SECONDS:-7200}"

mkdir -p "${OUT_ROOT}/parts" "${OUT_ROOT}/logs" "${OUT_ROOT}/live"
touch "${OUT_ROOT}/part_status.tsv"

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"

cd "${ROOT_DIR}" || exit 1

IFS=',' read -r -a MODELS < "${MODEL_LIST}"
TOTAL="${#MODELS[@]}"
IDX=0

for RAW_MODEL in "${MODELS[@]}"; do
  MODEL="$(echo "${RAW_MODEL}" | xargs)"
  if [[ -z "${MODEL}" ]]; then
    continue
  fi
  IDX=$((IDX + 1))
  PART_DIR="${OUT_ROOT}/parts/${MODEL}"
  LOG_PATH="${OUT_ROOT}/logs/${MODEL}.log"
  mkdir -p "${PART_DIR}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${MODEL}" "running" "" "${PART_DIR}" >> "${OUT_ROOT}/part_status.tsv"
  echo "[${IDX}/${TOTAL}] running ${MODEL}" | tee -a "${OUT_ROOT}/run.log"

  timeout "${MODEL_TIMEOUT_SECONDS}" "${PYTHON_BIN}" "${MODEL_SCRIPT_DIR}/run_corn_monthly_spike_official_pool_two_heads.py" \
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
  echo "[${IDX}/${TOTAL}] ${MODEL} ${STATUS} rc=${RC}" | tee -a "${OUT_ROOT}/run.log"

  "${PYTHON_BIN}" "${MODEL_SCRIPT_DIR}/aggregate_longlookback_results.py" --root "${OUT_ROOT}" --out "${OUT_ROOT}/live" >> "${OUT_ROOT}/aggregate.log" 2>&1 || true
done

"${PYTHON_BIN}" "${MODEL_SCRIPT_DIR}/aggregate_longlookback_results.py" --root "${OUT_ROOT}" --out "${OUT_ROOT}/live" >> "${OUT_ROOT}/aggregate.log" 2>&1 || true
echo "complete $(date --iso-8601=seconds)" | tee -a "${OUT_ROOT}/run.log"
