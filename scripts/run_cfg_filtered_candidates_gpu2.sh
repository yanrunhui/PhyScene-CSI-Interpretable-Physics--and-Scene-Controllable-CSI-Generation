#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
discover() {
  { find "$1" -type f -name "$2" -print 2>/dev/null || true; } | sort | tail -1
}
SOURCE_ARTIFACT="${SOURCE_ARTIFACT:-$(discover artifacts/generated/aod_conditional_fidelity test_variants.pt)}"
AOD_CHECKPOINT="${AOD_CHECKPOINT:-$(discover artifacts/checkpoints/aod_conditioned_b3 best.pt)}"
DP_CHECKPOINT="${DP_CHECKPOINT:-$(discover artifacts/checkpoints/delay_power_residual best.pt)}"
for required in "${SOURCE_ARTIFACT}" "${AOD_CHECKPOINT}" "${DP_CHECKPOINT}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty>}" >&2
    exit 2
  fi
done

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/generated/cfg_filtered_candidates/${RUN_TAG}}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/cfg_filtered_candidates/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/cfg_filtered_candidates/${RUN_TAG}}"
ARTIFACT="${ARTIFACT_DIR}/cfg2_cfg3_filtered.pt"
REPORT="${REPORT_DIR}/cfg2_cfg3_filtered.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${ARTIFACT_DIR}" "${REPORT_DIR}" "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] CFG filtered-candidate experiment completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] CFG filtered-candidate experiment failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\npython=%s\ncuda_visible_devices=%s\n' \
    "${PROJECT_ROOT}" "${PYTHON_BIN}" "${CUDA_VISIBLE_DEVICES}"
  printf 'source_artifact=%s\naod_checkpoint=%s\ndp_checkpoint=%s\n' \
    "${SOURCE_ARTIFACT}" "${AOD_CHECKPOINT}" "${DP_CHECKPOINT}"
  printf 'candidates_per_query=%s\noutputs_per_query=%s\ntolerance_deg=%s\n' \
    "${CANDIDATES_PER_QUERY:-16}" "${OUTPUTS_PER_QUERY:-8}" "${TOLERANCE_DEG:-10}"
  printf 'artifact=%s\nreport=%s\nsummary=%s\nlog_dir=%s\n' \
    "${ARTIFACT}" "${REPORT}" "${SUMMARY}" "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

run_stage() {
  local name="$1"
  shift
  printf '\n[%s] START %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] DONE  %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
}

run_stage 01_generate_filtered_candidates \
  "${PYTHON_BIN}" scripts/generate_cfg_filtered_candidates.py \
  --source-artifact "${SOURCE_ARTIFACT}" \
  --aod-checkpoint "${AOD_CHECKPOINT}" \
  --dp-checkpoint "${DP_CHECKPOINT}" \
  --output "${ARTIFACT}" \
  --candidates-per-query "${CANDIDATES_PER_QUERY:-16}" \
  --outputs-per-query "${OUTPUTS_PER_QUERY:-8}" \
  --tolerance-deg "${TOLERANCE_DEG:-10}" \
  --batch-size "${QUERY_BATCH_SIZE:-32}" \
  --device cuda --seed "${SEED:-2026}"

run_stage 02_evaluate_filtered_candidates \
  "${PYTHON_BIN}" scripts/evaluate_cfg_filtered_candidates.py \
  --artifact "${ARTIFACT}" --output "${REPORT}" --device cuda \
  --angle-bin-width-deg "${ANGLE_BIN_WIDTH_DEG:-60}" \
  --minimum-queries-per-map-angle "${MINIMUM_QUERIES_PER_MAP_ANGLE:-3}" \
  --coverage-k "${COVERAGE_K:-3}" \
  --projection-dimension "${PROJECTION_DIMENSION:-64}" \
  --projection-seeds "${PROJECTION_SEEDS:-0,1,2}" \
  --swd-projections "${SWD_PROJECTIONS:-128}" \
  --weighted-quantiles "${WEIGHTED_QUANTILES:-256}" \
  --synthesis-batch-size "${SYNTHESIS_BATCH_SIZE:-256}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
  --seed "${SEED:-2026}"

run_stage 03_summarize \
  "${PYTHON_BIN}" scripts/summarize_cfg_filtered_candidates.py \
  --report "${REPORT}" --output "${SUMMARY}"

printf '\nArtifact: %s\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${ARTIFACT}" "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
