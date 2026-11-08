#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
# Frozen production policy. Research sweeps should call the Python entry point
# directly instead of mutating these validated defaults through environment state.
AOD_GUIDANCE_SCALE="2.0"
CANDIDATE_MULTIPLIER="2"
CANDIDATE_TOP_UP_SIZE="8"
MAXIMUM_CANDIDATE_MULTIPLIER="8"
NATURAL_TOLERANCE_DEG="10"
B3_CHECKPOINT="${B3_CHECKPOINT:-artifacts/checkpoints/b3_seen_map_baseline/best.pt}"
SUPPORT_CACHE="${SUPPORT_CACHE:-artifacts/generated/delay_power_residual/b3_cache_seed2026.pt}"
if [[ -z "${AOD_CHECKPOINT:-}" ]]; then
  AOD_CHECKPOINT="$({
    find artifacts/checkpoints/aod_conditioned_b3 -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi

if [[ -z "${DP_CHECKPOINT:-}" ]]; then
  DP_CHECKPOINT="$({
    find artifacts/checkpoints/delay_power_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi
if [[ -z "${LINK_CHECKPOINT:-}" ]]; then
  LINK_CHECKPOINT="$({
    find artifacts/checkpoints/aod_link_global_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi
TARGET_AOD_AZ_DEG="${TARGET_AOD_AZ_DEG:-}"
if [[ -n "${TARGET_AOD_AZ_DEG}" ]]; then
  GENERATION_MODE="six_condition_mean_aod"
  REQUIRED_FILES=("${AOD_CHECKPOINT}" "${DP_CHECKPOINT}" "${SUPPORT_CACHE}")
else
  GENERATION_MODE="five_condition"
  REQUIRED_FILES=("${B3_CHECKPOINT}" "${DP_CHECKPOINT}" "${LINK_CHECKPOINT}" "${SUPPORT_CACHE}")
fi
for required in "${REQUIRED_FILES[@]}"; do
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
    printf '\n[%s] Controllable AoD generation completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Controllable AoD generation failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'map_id=%s\n' "${MAP_ID:-10}"
  printf 'generation_mode=%s\n' "${GENERATION_MODE}"
  printf 'b3_checkpoint=%s\n' "${B3_CHECKPOINT}"
  printf 'aod_checkpoint=%s\n' "${AOD_CHECKPOINT:-<not found>}"
  printf 'dp_checkpoint=%s\n' "${DP_CHECKPOINT}"
  printf 'link_checkpoint=%s\n' "${LINK_CHECKPOINT:-<not found>}"
  printf 'samples=%s\n' "${NUM_SAMPLES:-8}"
  printf 'target_aod_az_deg=%s\n' "${TARGET_AOD_AZ_DEG:-<not specified>}"
  printf 'candidate_multiplier=%s\n' "${CANDIDATE_MULTIPLIER}"
  printf 'candidate_top_up_size=%s\n' "${CANDIDATE_TOP_UP_SIZE}"
  printf 'maximum_candidate_multiplier=%s\n' "${MAXIMUM_CANDIDATE_MULTIPLIER}"
  printf 'natural_tolerance_deg=%s\n' "${NATURAL_TOLERANCE_DEG}"
  printf 'aod_guidance_scale=%s\n' "${AOD_GUIDANCE_SCALE}"
  printf 'selection_policy=uniform_random_without_replacement\n'
  printf 'tolerance_relaxation=disabled\n'
  printf 'output=%s\n' "${OUTPUT}"
  printf 'report=%s\n' "${REPORT}"
} | tee "${QUEUE_LOG}"

COMMAND=(
  "${PYTHON_BIN}" scripts/generate_dual_mode_aod_csi.py
  --dp-checkpoint "${DP_CHECKPOINT}"
  --support-cache "${SUPPORT_CACHE}"
  --output "${OUTPUT}"
  --report "${REPORT}"
  --map-id "${MAP_ID:-10}"
  --num-samples "${NUM_SAMPLES:-8}"
  --total-power-db "${TOTAL_POWER_DB:--90}"
  --delay-spread-ns "${DELAY_SPREAD_NS:-30}"
  --k-factor-db "${K_FACTOR_DB:-0}"
  --los-status "${LOS_STATUS:-0}"
  --path-count "${PATH_COUNT:-5}"
  --candidate-multiplier "${CANDIDATE_MULTIPLIER}"
  --candidate-top-up-size "${CANDIDATE_TOP_UP_SIZE}"
  --maximum-candidate-multiplier "${MAXIMUM_CANDIDATE_MULTIPLIER}"
  --candidate-generation-batch-size "${CANDIDATE_BATCH_SIZE:-512}"
  --natural-angle-tolerance-deg "${NATURAL_TOLERANCE_DEG}"
  --aod-guidance-scale "${AOD_GUIDANCE_SCALE}"
  --seed "${SEED:-2026}"
  --device cuda
)
if [[ "${GENERATION_MODE}" == "five_condition" ]]; then
  COMMAND+=(--b3-checkpoint "${B3_CHECKPOINT}" --link-checkpoint "${LINK_CHECKPOINT}")
else
  COMMAND+=(--aod-checkpoint "${AOD_CHECKPOINT}" --target-aod-az-deg "${TARGET_AOD_AZ_DEG}")
fi
"${COMMAND[@]}" \
  2>&1 | tee "${LOG_DIR}/generation.log"

printf '\nArtifact: %s\nReport: %s\nLogs: %s\n' \
  "${OUTPUT}" "${REPORT}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
