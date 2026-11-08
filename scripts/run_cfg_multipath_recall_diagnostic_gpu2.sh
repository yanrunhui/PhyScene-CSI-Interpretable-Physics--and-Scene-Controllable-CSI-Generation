#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if [[ -z "${ARTIFACT:-}" ]]; then
  ARTIFACT="$({
    find artifacts/generated/aod_conditional_fidelity -type f \
      -name test_variants.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi
if [[ -z "${ARTIFACT}" || ! -f "${ARTIFACT}" ]]; then
  printf 'Test conditional-fidelity artifact not found: %s\n' "${ARTIFACT:-<empty>}" >&2
  printf 'Set ARTIFACT=/path/to/test_variants.pt explicitly.\n' >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/cfg_multipath_recall/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/cfg_multipath_recall/${RUN_TAG}}"
REPORT="${REPORT_DIR}/cfg_multipath_recall.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] CFG multipath recall diagnostic completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] CFG multipath recall diagnostic failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'artifact=%s\n' "${ARTIFACT}"
  printf 'report=%s\n' "${REPORT}"
  printf 'summary=%s\n' "${SUMMARY}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START cfg_multipath_recall\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/diagnose_cfg_multipath_recall.py \
  --artifact "${ARTIFACT}" \
  --output "${REPORT}" \
  --device cuda \
  --minimum-samples-per-map "${MINIMUM_SAMPLES_PER_MAP:-8}" \
  --coverage-k "${COVERAGE_K:-3}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
  --seed "${SEED:-2026}" \
  2>&1 | tee "${LOG_DIR}/cfg_multipath_recall.log"
printf '[%s] DONE  cfg_multipath_recall\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/summarize_cfg_multipath_recall.py \
  --report "${REPORT}" --output "${SUMMARY}" \
  2>&1 | tee "${LOG_DIR}/summarize.log"

printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
