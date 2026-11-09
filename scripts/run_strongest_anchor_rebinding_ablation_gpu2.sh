#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-artifacts/strongest_anchor_rebinding_ablation/${RUN_TAG}}"
mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/generated" "${ROOT_DIR}/reports"

latest_file() {
  find "$1" -type f -name "$2" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

TEST_ARTIFACT="${TEST_ARTIFACT:-$(latest_file artifacts/strongest_aod_anchor_evaluation test_variants.pt)}"
DP_CHECKPOINT="${DP_CHECKPOINT:-$(latest_file artifacts/checkpoints/delay_power_residual best.pt)}"
for required in "${TEST_ARTIFACT}" "${DP_CHECKPOINT}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty>}" >&2
    exit 2
  fi
done

ARTIFACT="${ROOT_DIR}/generated/cfg${CFG:-1}_copy_swap_no_dp.pt"
REPORT="${ROOT_DIR}/reports/summary.json"
QUEUE_LOG="${ROOT_DIR}/logs/queue.log"
STATUS_FILE="${ROOT_DIR}/logs/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
  fi
  printf 'artifact=%s\nreport=%s\nexit_code=%s\n' \
    "${ARTIFACT}" "${REPORT}" "${code}" | tee -a "${QUEUE_LOG}"
}
trap on_exit EXIT

run_stage() {
  local name="$1"
  shift
  printf '[%s] START %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee "${ROOT_DIR}/logs/${name}.log"
  printf '[%s] DONE %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
}

{
  printf 'test_artifact=%s\ndp_checkpoint=%s\ncfg=%s\ngpu=%s\n' \
    "${TEST_ARTIFACT}" "${DP_CHECKPOINT}" "${CFG:-1}" "${CUDA_VISIBLE_DEVICES}"
} | tee "${QUEUE_LOG}"

run_stage 01_generate_paired_ablation \
  "${PYTHON_BIN}" scripts/generate_strongest_anchor_rebinding_ablation.py \
    --artifact "${TEST_ARTIFACT}" \
    --dp-checkpoint "${DP_CHECKPOINT}" \
    --output "${ARTIFACT}" \
    --cfg "${CFG:-1}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --device cuda

run_stage 02_evaluate_paired_ablation \
  "${PYTHON_BIN}" scripts/evaluate_strongest_anchor_rebinding_ablation.py \
    --artifact "${ARTIFACT}" \
    --output "${REPORT}" \
    --device cuda \
    --minimum-samples-per-map "${MINIMUM_SAMPLES_PER_MAP:-8}" \
    --projection-dimension "${PROJECTION_DIMENSION:-64}" \
    --projection-seeds "${PROJECTION_SEEDS:-0,1,2}" \
    --swd-projections "${SWD_PROJECTIONS:-128}" \
    --weighted-quantiles "${WEIGHTED_QUANTILES:-256}" \
    --coverage-k "${COVERAGE_K:-3}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
    --maximum-los-direct-change-over-1deg-rate \
      "${MAXIMUM_LOS_DIRECT_CHANGE_OVER_1DEG_RATE:-0.01}" \
    --seed "${SEED:-2026}"

printf 'Summary: %s\n' "${REPORT}" | tee -a "${QUEUE_LOG}"
