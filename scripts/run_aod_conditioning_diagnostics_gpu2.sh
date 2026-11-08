#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${AOD_CHECKPOINT:-}" ]]; then
  AOD_CHECKPOINT="$(find artifacts/checkpoints/aod_conditioned_b3 -type f -name best.pt -print 2>/dev/null | sort | tail -1)"
fi
if [[ -z "${DP_CHECKPOINT:-}" ]]; then
  DP_CHECKPOINT="$(find artifacts/checkpoints/delay_power_residual -mindepth 2 -maxdepth 2 -type f -name best.pt -print 2>/dev/null | sort | tail -1)"
fi
if [[ -z "${LINK_CHECKPOINT:-}" ]]; then
  LINK_CHECKPOINT="$(find artifacts/checkpoints/aod_link_global_residual -mindepth 2 -maxdepth 2 -type f -name best.pt -print 2>/dev/null | sort | tail -1)"
fi

if [[ ! -f "${AOD_CHECKPOINT}" ]]; then
  printf 'AoD-conditioned checkpoint not found: %s\n' "${AOD_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${DP_CHECKPOINT}" ]]; then
  printf 'DP checkpoint not found: %s\n' "${DP_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${LINK_CHECKPOINT}" ]]; then
  printf 'Link-global checkpoint not found: %s\n' "${LINK_CHECKPOINT}" >&2
  exit 2
fi

REPORT_DIR="${REPORT_DIR:-artifacts/reports/aod_conditioning_diagnostics/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/aod_conditioning_diagnostics/${RUN_TAG}}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

GRADIENT_REPORT="${REPORT_DIR}/gradient_audit.json"
CFG_REPORT="${REPORT_DIR}/cfg_and_stages.json"
SUMMARY="${REPORT_DIR}/summary.json"

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'aod_checkpoint=%s\n' "${AOD_CHECKPOINT}"
  printf 'dp_checkpoint=%s\n' "${DP_CHECKPOINT}"
  printf 'link_checkpoint=%s\n' "${LINK_CHECKPOINT}"
  printf 'report_dir=%s\n' "${REPORT_DIR}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${LOG_DIR}/queue.log"

run_stage() {
  local name="$1"
  shift
  printf '\n[%s] START %s\n' "$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')" "${name}" | tee -a "${LOG_DIR}/queue.log"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] DONE  %s\n' "$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')" "${name}" | tee -a "${LOG_DIR}/queue.log"
}

export CUDA_VISIBLE_DEVICES

run_stage 01_gradient_audit \
  "${PYTHON_BIN}" scripts/diagnose_aod_loss_gradients.py \
    --checkpoint "${AOD_CHECKPOINT}" \
    --output "${GRADIENT_REPORT}" \
    --device cuda \
    --batch-size "${GRADIENT_BATCH_SIZE:-256}" \
    --batches "${GRADIENT_BATCHES:-8}" \
    --seed "${SEED:-2026}" \
    --weights online

run_stage 02_cfg_and_stages \
  "${PYTHON_BIN}" scripts/diagnose_aod_cfg_and_stages.py \
    --checkpoint "${AOD_CHECKPOINT}" \
    --dp-checkpoint "${DP_CHECKPOINT}" \
    --link-checkpoint "${LINK_CHECKPOINT}" \
    --output "${CFG_REPORT}" \
    --maps "${MAPS:-10,25,50,75,96}" \
    --target-angles="${TARGET_ANGLES:--179,-150,-90,-30,30,38.5,90,150,179}" \
    --guidance-scales "${GUIDANCE_SCALES:-1.0,1.5,2.0,3.0}" \
    --los-statuses "${LOS_STATUSES:-0,1}" \
    --samples-per-cell "${SAMPLES_PER_CELL:-128}" \
    --device cuda \
    --seed "${SEED:-2026}"

run_stage 03_summarize \
  "${PYTHON_BIN}" scripts/summarize_aod_conditioning_diagnostics.py \
    --cfg-report "${CFG_REPORT}" \
    --gradient-report "${GRADIENT_REPORT}" \
    --output "${SUMMARY}"

{
  printf '\nGradient report: %s\n' "${GRADIENT_REPORT}"
  printf 'CFG/stage report: %s\n' "${CFG_REPORT}"
  printf 'Summary: %s\n' "${SUMMARY}"
  printf 'Logs: %s\n' "${LOG_DIR}"
  printf '\n[%s] AoD conditioning diagnostics completed successfully.\n' "$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
} | tee -a "${LOG_DIR}/queue.log"
