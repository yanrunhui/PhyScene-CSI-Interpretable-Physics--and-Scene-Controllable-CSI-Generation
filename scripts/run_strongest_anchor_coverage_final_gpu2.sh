#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-artifacts/strongest_anchor_coverage_final/${RUN_TAG}}"
mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/generated" "${ROOT_DIR}/reports"

latest_file() {
  {
    find "$1" -type f -name "$2" -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -nr | head -1 | cut -d' ' -f2-
}

if [[ -z "${REBIND_ARTIFACT:-}" ]]; then
  REBIND_ARTIFACT="$(latest_file \
    artifacts/strongest_anchor_rebinding_ablation cfg1_copy_swap_no_dp.pt)"
fi
if [[ -z "${ANCHOR_CHECKPOINT:-}" ]]; then
  ANCHOR_CHECKPOINT="$({
    find artifacts/strongest_aod_anchor artifacts/checkpoints/strongest_aod_anchor_b3 \
      -type f -path '*/checkpoints/best.pt' -printf '%T@ %p\n' 2>/dev/null || true
    find artifacts/checkpoints/strongest_aod_anchor_b3 \
      -type f -name best.pt -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -nr | head -1 | cut -d' ' -f2-)"
fi
DP_CHECKPOINT="${DP_CHECKPOINT:-$(latest_file artifacts/checkpoints/delay_power_residual best.pt)}"
for required in "${REBIND_ARTIFACT}" "${ANCHOR_CHECKPOINT}" "${DP_CHECKPOINT}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty>}" >&2
    printf 'Set REBIND_ARTIFACT, ANCHOR_CHECKPOINT and DP_CHECKPOINT explicitly.\n' >&2
    exit 2
  fi
done

COVERAGE_REPORT="${ROOT_DIR}/reports/01_coverage_strata.json"
DIVERSITY_ARTIFACT="${ROOT_DIR}/generated/cfg1_shared_24_candidate_pool.pt"
DIVERSITY_REPORT="${ROOT_DIR}/reports/02_diversity_selection.json"
SUMMARY="${ROOT_DIR}/reports/summary.json"
QUEUE_LOG="${ROOT_DIR}/logs/queue.log"
STATUS_FILE="${ROOT_DIR}/logs/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
  fi
  printf 'coverage_report=%s\ndiversity_artifact=%s\ndiversity_report=%s\nexit_code=%s\n' \
    "${COVERAGE_REPORT}" "${DIVERSITY_ARTIFACT}" "${DIVERSITY_REPORT}" "${code}" \
    | tee -a "${QUEUE_LOG}"
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
  printf 'rebinding_artifact=%s\nanchor_checkpoint=%s\ndp_checkpoint=%s\n' \
    "${REBIND_ARTIFACT}" "${ANCHOR_CHECKPOINT}" "${DP_CHECKPOINT}"
  printf 'cfg=1\ngpu=%s\nroot_dir=%s\n' "${CUDA_VISIBLE_DEVICES}" "${ROOT_DIR}"
} | tee "${QUEUE_LOG}"

run_stage 01_coverage_strata \
  "${PYTHON_BIN}" scripts/diagnose_strongest_anchor_coverage_strata.py \
    --artifact "${REBIND_ARTIFACT}" \
    --output "${COVERAGE_REPORT}" \
    --variant dp_swap \
    --minimum-real-support "${MINIMUM_REAL_SUPPORT:-8}" \
    --minimum-samples-per-group "${MINIMUM_SAMPLES_PER_GROUP:-8}" \
    --projection-dimension "${PROJECTION_DIMENSION:-64}" \
    --projection-seeds "${PROJECTION_SEEDS:-0,1,2}" \
    --swd-projections "${SWD_PROJECTIONS:-128}" \
    --weighted-quantiles "${WEIGHTED_QUANTILES:-256}" \
    --coverage-k "${COVERAGE_K:-3}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
    --seed "${SEED:-2026}" \
    --device cuda

run_stage 02_generate_shared_candidate_pool \
  "${PYTHON_BIN}" scripts/generate_strongest_anchor_diversity_artifact.py \
    --source-artifact "${REBIND_ARTIFACT}" \
    --aod-checkpoint "${ANCHOR_CHECKPOINT}" \
    --dp-checkpoint "${DP_CHECKPOINT}" \
    --output "${DIVERSITY_ARTIFACT}" \
    --maximum-queries-per-map "${MAXIMUM_QUERIES_PER_MAP:-8}" \
    --query-batch-size "${QUERY_BATCH_SIZE:-8}" \
    --candidates-per-query 24 \
    --outputs-per-query 8 \
    --diversity-strength "${DIVERSITY_STRENGTH:-0.35}" \
    --guidance-scale 1 \
    --angle-tolerance-deg 10 \
    --minimum-real-support "${MINIMUM_REAL_SUPPORT:-8}" \
    --seed "${SEED:-2026}" \
    --device cuda

run_stage 03_evaluate_candidate_selection \
  "${PYTHON_BIN}" scripts/evaluate_strongest_anchor_diversity_selection.py \
    --artifact "${DIVERSITY_ARTIFACT}" \
    --output "${DIVERSITY_REPORT}" \
    --minimum-queries-per-map "${MINIMUM_QUERIES_PER_MAP:-4}" \
    --projection-dimension "${PROJECTION_DIMENSION:-64}" \
    --projection-seeds "${PROJECTION_SEEDS:-0,1,2}" \
    --swd-projections "${SWD_PROJECTIONS:-128}" \
    --weighted-quantiles "${WEIGHTED_QUANTILES:-256}" \
    --coverage-k "${COVERAGE_K:-3}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
    --minimum-hit-at-10 "${MINIMUM_HIT_AT_10:-0.99}" \
    --minimum-selection-success-rate "${MINIMUM_SELECTION_SUCCESS_RATE:-0.95}" \
    --seed "${SEED:-2026}" \
    --device cuda

run_stage 04_summarize \
  "${PYTHON_BIN}" scripts/summarize_strongest_anchor_coverage_final.py \
    --coverage-report "${COVERAGE_REPORT}" \
    --diversity-report "${DIVERSITY_REPORT}" \
    --output "${SUMMARY}"

printf 'Final summary: %s\n' "${SUMMARY}" | tee -a "${QUEUE_LOG}"
