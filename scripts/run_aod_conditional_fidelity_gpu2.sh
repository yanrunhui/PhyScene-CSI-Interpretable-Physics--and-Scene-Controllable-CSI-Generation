#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

discover() {
  { find "$1" -type f -name "$2" -print 2>/dev/null || true; } | sort | tail -1
}
AOD_CHECKPOINT="${AOD_CHECKPOINT:-$(discover artifacts/checkpoints/aod_conditioned_b3 best.pt)}"
DP_CHECKPOINT="${DP_CHECKPOINT:-$(discover artifacts/checkpoints/delay_power_residual best.pt)}"
LINK_CHECKPOINT="${LINK_CHECKPOINT:-$(discover artifacts/checkpoints/aod_link_global_residual best.pt)}"
CACHE="${CACHE:-$(discover artifacts/generated/delay_power_residual 'b3_cache_seed*.pt')}"
for required in "${AOD_CHECKPOINT}" "${DP_CHECKPOINT}" "${LINK_CHECKPOINT}" "${CACHE}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty>}" >&2
    exit 2
  fi
done

REPORT_DIR="${REPORT_DIR:-artifacts/reports/aod_conditional_fidelity/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/aod_conditional_fidelity/${RUN_TAG}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/generated/aod_conditional_fidelity/${RUN_TAG}}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "${ARTIFACT_DIR}"
QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] AoD conditional fidelity completed successfully.\n' "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] AoD conditional fidelity failed with exit code %s.\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

VAL_SUPPORT="${REPORT_DIR}/validation_support.json"
VAL_PLAN="${ARTIFACT_DIR}/validation_pair_plan.pt"
VAL_ARTIFACT="${ARTIFACT_DIR}/validation_variants.pt"
VAL_REPORT="${REPORT_DIR}/validation_fidelity.json"
TEST_SUPPORT="${REPORT_DIR}/test_support.json"
TEST_PLAN="${ARTIFACT_DIR}/test_pair_plan.pt"
TEST_ARTIFACT="${ARTIFACT_DIR}/test_variants.pt"
TEST_REPORT="${REPORT_DIR}/test_fidelity.json"
SUMMARY="${REPORT_DIR}/summary.json"

{
  printf 'project_root=%s\npython=%s\ncuda_visible_devices=%s\n' "${PROJECT_ROOT}" "${PYTHON_BIN}" "${CUDA_VISIBLE_DEVICES}"
  printf 'aod_checkpoint=%s\ndp_checkpoint=%s\nlink_checkpoint=%s\ncache=%s\n' "${AOD_CHECKPOINT}" "${DP_CHECKPOINT}" "${LINK_CHECKPOINT}" "${CACHE}"
  printf 'report_dir=%s\nartifact_dir=%s\nlog_dir=%s\n' "${REPORT_DIR}" "${ARTIFACT_DIR}" "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

run_stage() {
  local name="$1"
  shift
  printf '\n[%s] START %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] DONE  %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
}

support_stage() {
  local split="$1" report="$2" plan="$3"
  if [[ "${REUSE_EXISTING:-0}" == "1" && -f "${report}" && -f "${plan}" ]]; then
    printf '\n[%s] SKIP support_%s (reusing %s and %s)\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${split}" "${report}" "${plan}" | tee -a "${QUEUE_LOG}"
    return
  fi
  run_stage "01_support_${split}" "${PYTHON_BIN}" scripts/diagnose_map_condition_aod_support.py \
    --cache "${CACHE}" --split "${split}" --output-report "${report}" --output-plan "${plan}" \
    --path-count 5 --maximum-condition-distance "${CONDITION_RADIUS:-1.5}" \
    --maximum-angle-distance-deg "${ANGLE_RADIUS_DEG:-10}" --minimum-support "${MINIMUM_SUPPORT:-8}" \
    --maximum-pairs-per-map-los "${MAXIMUM_PAIRS_PER_MAP_LOS:-16}" --seed "${SEED:-2026}"
}

generate_stage() {
  local split="$1" plan="$2" output="$3"
  if [[ "${REUSE_EXISTING:-0}" == "1" && -f "${output}" ]]; then
    printf '\n[%s] SKIP generate_%s (reusing %s)\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${split}" "${output}" | tee -a "${QUEUE_LOG}"
    return
  fi
  run_stage "02_generate_${split}" "${PYTHON_BIN}" scripts/generate_aod_conditional_fidelity_artifact.py \
    --aod-checkpoint "${AOD_CHECKPOINT}" --dp-checkpoint "${DP_CHECKPOINT}" \
    --link-checkpoint "${LINK_CHECKPOINT}" --cache "${CACHE}" --pair-plan "${plan}" \
    --output "${output}" --batch-size "${BATCH_SIZE:-128}" --device cuda --seed "${SEED:-2026}"
}

evaluate_stage() {
  local split="$1" artifact="$2" output="$3" selection="${4:-}"
  if [[ "${REUSE_EXISTING:-0}" == "1" && -f "${output}" ]]; then
    printf '\n[%s] SKIP evaluate_%s (reusing %s)\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${split}" "${output}" | tee -a "${QUEUE_LOG}"
    return
  fi
  local arguments=()
  if [[ -n "${selection}" ]]; then arguments+=(--selection-from "${selection}"); fi
  run_stage "03_evaluate_${split}" "${PYTHON_BIN}" scripts/evaluate_aod_conditional_fidelity.py \
    --artifact "${artifact}" --output "${output}" "${arguments[@]}" --device cuda \
    --minimum-samples-per-map "${MINIMUM_SAMPLES_PER_MAP:-8}" \
    --projection-dimension "${PROJECTION_DIMENSION:-64}" --projection-seeds "${PROJECTION_SEEDS:-0,1,2}" \
    --swd-projections "${SWD_PROJECTIONS:-128}" --weighted-quantiles "${WEIGHTED_QUANTILES:-256}" \
    --coverage-k "${COVERAGE_K:-3}" --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" --seed "${SEED:-2026}"
}

support_stage validation "${VAL_SUPPORT}" "${VAL_PLAN}"
generate_stage validation "${VAL_PLAN}" "${VAL_ARTIFACT}"
evaluate_stage validation "${VAL_ARTIFACT}" "${VAL_REPORT}"
support_stage test "${TEST_SUPPORT}" "${TEST_PLAN}"
generate_stage test "${TEST_PLAN}" "${TEST_ARTIFACT}"
evaluate_stage test "${TEST_ARTIFACT}" "${TEST_REPORT}" "${VAL_REPORT}"
run_stage 04_summarize "${PYTHON_BIN}" scripts/summarize_aod_conditional_fidelity.py \
  --support-validation "${VAL_SUPPORT}" --support-test "${TEST_SUPPORT}" \
  --validation "${VAL_REPORT}" --test "${TEST_REPORT}" --output "${SUMMARY}"

printf '\nSummary: %s\nValidation: %s\nTest: %s\nLogs: %s\n' "${SUMMARY}" "${VAL_REPORT}" "${TEST_REPORT}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
