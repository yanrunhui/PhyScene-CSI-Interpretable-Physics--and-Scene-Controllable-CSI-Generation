#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

discover() {
  { find "$1" -type f -name "$2" -print 2>/dev/null || true; } | sort | tail -1
}

STRONGEST_AOD_CHECKPOINT="${STRONGEST_AOD_CHECKPOINT:-$(discover artifacts/checkpoints/strongest_aod_conditioned_b3 best.pt)}"
DP_CHECKPOINT="${DP_CHECKPOINT:-$(discover artifacts/checkpoints/delay_power_residual best.pt)}"
for required in "${STRONGEST_AOD_CHECKPOINT}" "${DP_CHECKPOINT}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty>}" >&2
    exit 2
  fi
done

REPORT_DIR="${REPORT_DIR:-artifacts/reports/strongest_aod_stage_diagnostics/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/strongest_aod_stage_diagnostics/${RUN_TAG}}"
REPORT="${REPORT:-${REPORT_DIR}/stage_diagnostics.json}"
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
  printf 'strongest_aod_checkpoint=%s\ndp_checkpoint=%s\n' \
    "${STRONGEST_AOD_CHECKPOINT}" "${DP_CHECKPOINT}"
  printf 'maps=%s\ntarget_angles=%s\nguidance_scales=%s\nlos_statuses=%s\n' \
    "${MAPS:-94,85,87,99,78,10,25}" \
    "${TARGET_ANGLES:--135,-45,0,45,90,135}" \
    "${GUIDANCE_SCALES:-2,3}" "${LOS_STATUSES:-0,1}"
  printf 'samples_per_cell=%s\nreport=%s\n' \
    "${SAMPLES_PER_CELL:-128}" "${REPORT}"
} | tee "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/diagnose_aod_cfg_and_stages.py \
  --checkpoint "${STRONGEST_AOD_CHECKPOINT}" \
  --dp-checkpoint "${DP_CHECKPOINT}" \
  --maps "${MAPS:-94,85,87,99,78,10,25}" \
  --target-angles="${TARGET_ANGLES:--135,-45,0,45,90,135}" \
  --guidance-scales "${GUIDANCE_SCALES:-2,3}" \
  --los-statuses "${LOS_STATUSES:-0,1}" \
  --samples-per-cell "${SAMPLES_PER_CELL:-128}" \
  --device cuda --seed "${SEED:-2026}" --output "${REPORT}" \
  2>&1 | tee "${LOG_DIR}/stage_diagnostics.log"
