#!/usr/bin/env bash
set -u

ROOT_DIR="${ROOT_DIR:-/root/corn_spike_sota}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_SCRIPT_DIR="${MODEL_SCRIPT_DIR:-${ROOT_DIR}/玉米预测/scripts}"
CSV_PATH="${CSV_PATH:-${ROOT_DIR}/玉米预测/datasets/corn_monthly.csv}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/longlookback_2016fixed_50_plus_keras_20260705_162158}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MAX_ACCEL_JOBS="${MAX_ACCEL_JOBS:-5}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"

mkdir -p "${OUT_ROOT}/parts" "${OUT_ROOT}/logs"
cd "${ROOT_DIR}" || exit 1

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

if [[ -n "${ACCEL_JOB_SPECS:-}" ]]; then
  mapfile -t JOBS < <(printf '%s\n' "${ACCEL_JOB_SPECS}" | sed '/^[[:space:]]*$/d')
else
  JOBS=(
    "keras_lstm_stack2_u32|zz_accel_with_news_lb12|with_news_precomputed_pca|12|1,2"
    "keras_lstm_stack2_u32|zz_accel_with_news_lb9_h2|with_news_precomputed_pca|9|2"
    "keras_gru_u16|zz_accel_with_news_all|with_news_precomputed_pca|6,9,12|1,2"
    "keras_bilstm_u16|zz_accel_with_news_all|with_news_precomputed_pca|6,9,12|1,2"
    "keras_tcn_filters8_k2_d1|zz_accel_with_news_all|with_news_precomputed_pca|6,9,12|1,2"
    "keras_tcn_filters16_k2_d1|zz_accel_with_news_all|with_news_precomputed_pca|6,9,12|1,2"
    "aeon_deep_inceptiontime|zz_accel_with_news_lb12|with_news_precomputed_pca|12|1,2"
  )
fi

ACCEL_LOG="${OUT_ROOT}/logs/accel_${RUN_TAG}.log"
echo "accel started $(date --iso-8601=seconds) max_jobs=${MAX_ACCEL_JOBS}" | tee -a "${ACCEL_LOG}" "${OUT_ROOT}/run.log"

run_job() {
  local SPEC="$1"
  local MODEL SUFFIX FEATURE_SETS LOOKBACKS HORIZONS ORIGIN_START ORIGIN_STOP
  IFS='|' read -r MODEL SUFFIX FEATURE_SETS LOOKBACKS HORIZONS ORIGIN_START ORIGIN_STOP <<< "${SPEC}"
  ORIGIN_START="${ORIGIN_START:-0}"
  ORIGIN_STOP="${ORIGIN_STOP:-0}"
  local PART_DIR="${OUT_ROOT}/parts/${MODEL}_${SUFFIX}"
  local LOG_PATH="${OUT_ROOT}/logs/${MODEL}_${SUFFIX}_${RUN_TAG}.log"
  mkdir -p "${PART_DIR}"

  printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${MODEL}__${SUFFIX}" "running_parallel" "" "${PART_DIR}" >> "${OUT_ROOT}/part_status.tsv"
  echo "accel running ${MODEL} suffix=${SUFFIX} features=${FEATURE_SETS} lookbacks=${LOOKBACKS} horizons=${HORIZONS} origins=${ORIGIN_START}:${ORIGIN_STOP} $(date --iso-8601=seconds)" | tee -a "${ACCEL_LOG}" "${OUT_ROOT}/run.log"

  timeout "${TIMEOUT_SECONDS}" "${PYTHON_BIN}" "${MODEL_SCRIPT_DIR}/run_corn_monthly_spike_official_pool_two_heads.py" \
    --csv "${CSV_PATH}" \
    --out-dir "${PART_DIR}" \
    --origin-mode monthly \
    --threshold-mode validation \
    --test-start 2017-01-01 \
    --monthly-cutoff-lag 1 \
    --min-train 24 \
    --val-size 12 \
    --lookbacks "${LOOKBACKS}" \
    --horizons "${HORIZONS}" \
    --origin-id-start "${ORIGIN_START}" \
    --origin-id-stop "${ORIGIN_STOP}" \
    --feature-sets "${FEATURE_SETS}" \
    --models "${MODEL}" \
    --heads cls,reg \
    --checkpoint \
    --resume-latest \
    --save-folds \
    --deep-epochs 6 \
    --deep-batch-size 16 \
    > "${LOG_PATH}" 2>&1
  local RC=$?

  local RUN_DIR
  RUN_DIR="$(find "${PART_DIR}" -mindepth 1 -maxdepth 1 -type d -print | sort | tail -1)"
  local STATUS
  if [[ "${RC}" == "0" ]]; then
    STATUS="done"
  elif [[ "${RC}" == "124" ]]; then
    STATUS="timeout"
  else
    STATUS="failed"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${MODEL}__${SUFFIX}" "${STATUS}" "${RC}" "${RUN_DIR}" >> "${OUT_ROOT}/part_status.tsv"
  echo "accel finished ${MODEL} suffix=${SUFFIX} status=${STATUS} rc=${RC} $(date --iso-8601=seconds)" | tee -a "${ACCEL_LOG}" "${OUT_ROOT}/run.log"
}

running_jobs=0
for SPEC in "${JOBS[@]}"; do
  run_job "${SPEC}" &
  running_jobs=$((running_jobs + 1))
  if (( running_jobs >= MAX_ACCEL_JOBS )); then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
done
wait

"${PYTHON_BIN}" "${MODEL_SCRIPT_DIR}/aggregate_longlookback_results.py" --root "${OUT_ROOT}" --out "${OUT_ROOT}/live" >> "${OUT_ROOT}/aggregate.log" 2>&1 || true
echo "accel stopped $(date --iso-8601=seconds)" | tee -a "${ACCEL_LOG}" "${OUT_ROOT}/run.log"
