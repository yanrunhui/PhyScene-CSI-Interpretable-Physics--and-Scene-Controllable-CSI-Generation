#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${ARTIFACT:-}" ]]; then
  ARTIFACT="$({
    find artifacts/strongest_aod_evaluation -type f \
      -name test_variants.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi
if [[ -z "${ARTIFACT}" || ! -f "${ARTIFACT}" ]]; then
  printf 'Strongest Test fidelity artifact not found: %s\n' \
    "${ARTIFACT:-<empty>}" >&2
  printf 'Set ARTIFACT=/path/to/test_variants.pt explicitly.\n' >&2
  exit 2
fi

REPORT_DIR="${REPORT_DIR:-artifacts/reports/strongest_aod_counterfactual_swaps/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/strongest_aod_counterfactual_swaps/${RUN_TAG}}"
REPORT="${REPORT:-${REPORT_DIR}/counterfactual_swaps.json}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
  fi
  printf '\nreport=%s\nlogs=%s\nexit_code=%s\n' \
    "${REPORT}" "${LOG_DIR}" "${code}" | tee -a "${QUEUE_LOG}"
}
trap on_exit EXIT

{
  printf 'project_root=%s\npython=%s\ncuda_visible_devices=%s\n' \
    "${PROJECT_ROOT}" "${PYTHON_BIN}" "${CUDA_VISIBLE_DEVICES}"
  printf 'artifact=%s\ncandidate_label=%s\nreport=%s\n' \
    "${ARTIFACT}" "${CANDIDATE_LABEL:-aod_cfg3_dp}" "${REPORT}"
} | tee "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/diagnose_strongest_aod_counterfactual_swaps.py \
  --artifact "${ARTIFACT}" --candidate-label "${CANDIDATE_LABEL:-aod_cfg3_dp}" \
  --output "${REPORT}" --device cuda \
  --minimum-samples-per-map "${MINIMUM_SAMPLES_PER_MAP:-8}" \
  --projection-dimension "${PROJECTION_DIMENSION:-64}" \
  --projection-seeds "${PROJECTION_SEEDS:-0,1,2}" \
  --swd-projections "${SWD_PROJECTIONS:-128}" \
  --weighted-quantiles "${WEIGHTED_QUANTILES:-256}" \
  --coverage-k "${COVERAGE_K:-3}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
  --seed "${SEED:-2026}" \
  2>&1 | tee "${LOG_DIR}/counterfactual_swaps.log"
