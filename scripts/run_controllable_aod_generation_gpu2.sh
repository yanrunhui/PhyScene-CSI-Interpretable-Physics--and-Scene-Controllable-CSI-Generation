#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
AOD_MODE="${AOD_MODE:-mean_power_weighted}"
TARGET_AOD_DEG="${TARGET_AOD_DEG:-}"
if [[ "${AOD_MODE}" != "mean_power_weighted" && "${AOD_MODE}" != "strongest_path" ]]; then
  printf 'AOD_MODE must be mean_power_weighted or strongest_path; got %s\n' \
    "${AOD_MODE}" >&2
  exit 2
fi
if [[ -z "${TARGET_AOD_DEG}" ]]; then
  printf 'Set TARGET_AOD_DEG to the requested angle in degrees.\n' >&2
  exit 2
fi

latest_file() {
  {
    find "$1" -type f -name best.pt -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -nr | head -1 | cut -d' ' -f2-
}

if [[ -z "${MEAN_AOD_CHECKPOINT:-}" ]]; then
  MEAN_AOD_CHECKPOINT="$(latest_file artifacts/checkpoints/aod_conditioned_b3)"
fi
if [[ -z "${STRONGEST_AOD_CHECKPOINT:-}" ]]; then
  STRONGEST_AOD_CHECKPOINT="$({
    find artifacts/strongest_aod_anchor artifacts/checkpoints/strongest_aod_anchor_b3 \
      -type f -name best.pt -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -nr | head -1 | cut -d' ' -f2-)"
fi
if [[ -z "${DP_CHECKPOINT:-}" ]]; then
  DP_CHECKPOINT="$(latest_file artifacts/checkpoints/delay_power_residual)"
fi
SUPPORT_CACHE="${SUPPORT_CACHE:-artifacts/generated/delay_power_residual/b3_cache_seed2026.pt}"

if [[ "${AOD_MODE}" == "mean_power_weighted" ]]; then
  SELECTED_CHECKPOINT="${MEAN_AOD_CHECKPOINT}"
  CFG="2"
  OUTPUT_POLICY="fixed_16_to_8_natural_mean_filter"
else
  SELECTED_CHECKPOINT="${STRONGEST_AOD_CHECKPOINT}"
  CFG="1"
  OUTPUT_POLICY="direct_8_dp_swap_top1_verification"
fi
for required in "${SELECTED_CHECKPOINT}" "${DP_CHECKPOINT}" "${SUPPORT_CACHE}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty>}" >&2
    exit 2
  fi
done

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-artifacts/logs/controllable_aod_generation/${RUN_TAG}}"
OUTPUT="${OUTPUT:-artifacts/generated/controllable_aod/csi_${RUN_TAG}.pt}"
REPORT="${REPORT:-artifacts/reports/controllable_aod_generation/${RUN_TAG}/generation.json}"
mkdir -p "${LOG_DIR}" "$(dirname -- "${OUTPUT}")" "$(dirname -- "${REPORT}")"
QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
  fi
  printf 'artifact=%s\nreport=%s\nlogs=%s\nexit_code=%s\n' \
    "${OUTPUT}" "${REPORT}" "${LOG_DIR}" "${code}" | tee -a "${QUEUE_LOG}"
}
trap on_exit EXIT

{
  printf 'project_root=%s\ncuda_visible_devices=%s\n' \
    "${PROJECT_ROOT}" "${CUDA_VISIBLE_DEVICES}"
  printf 'aod_mode=%s\ntarget_aod_deg=%s\ncfg=%s\noutput_policy=%s\n' \
    "${AOD_MODE}" "${TARGET_AOD_DEG}" "${CFG}" "${OUTPUT_POLICY}"
  printf 'map_id=%s\nmean_aod_checkpoint=%s\nstrongest_anchor_checkpoint=%s\n' \
    "${MAP_ID:-10}" "${MEAN_AOD_CHECKPOINT:-<not found>}" \
    "${STRONGEST_AOD_CHECKPOINT:-<not found>}"
  printf 'selected_checkpoint=%s\ndp_checkpoint=%s\nsupport_cache=%s\n' \
    "${SELECTED_CHECKPOINT}" "${DP_CHECKPOINT}" "${SUPPORT_CACHE}"
  printf 'output=%s\nreport=%s\n' "${OUTPUT}" "${REPORT}"
} | tee "${QUEUE_LOG}"

COMMAND=(
  "${PYTHON_BIN}" scripts/generate_dual_mode_aod_csi.py
  --dp-checkpoint "${DP_CHECKPOINT}"
  --support-cache "${SUPPORT_CACHE}"
  --output "${OUTPUT}"
  --report "${REPORT}"
  --aod-mode "${AOD_MODE}"
  --target-aod-deg "${TARGET_AOD_DEG}"
  --map-id "${MAP_ID:-10}"
  --num-samples 8
  --total-power-db "${TOTAL_POWER_DB:--90}"
  --delay-spread-ns "${DELAY_SPREAD_NS:-30}"
  --k-factor-db "${K_FACTOR_DB:-0}"
  --los-status "${LOS_STATUS:-0}"
  --path-count "${PATH_COUNT:-5}"
  --candidate-generation-batch-size "${CANDIDATE_BATCH_SIZE:-512}"
  --natural-angle-tolerance-deg 10
  --seed "${SEED:-2026}"
  --device cuda
)
if [[ "${AOD_MODE}" == "mean_power_weighted" ]]; then
  COMMAND+=(--mean-aod-checkpoint "${MEAN_AOD_CHECKPOINT}")
else
  COMMAND+=(--strongest-aod-checkpoint "${STRONGEST_AOD_CHECKPOINT}")
fi

"${COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/generation.log"
