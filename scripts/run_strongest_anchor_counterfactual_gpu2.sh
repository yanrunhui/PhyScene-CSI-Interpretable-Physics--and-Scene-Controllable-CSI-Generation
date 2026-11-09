#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-artifacts/strongest_anchor_counterfactual/${RUN_TAG}}"
mkdir -p "${ROOT_DIR}"

required_variables=(
  ANCHOR_CHECKPOINT
  SUPPORT_CACHE
  MAP_ID
  TOTAL_POWER_DB
  DELAY_SPREAD_NS
  LOS_STATUS
  STRONGEST_AOD_DEG
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    printf 'Required environment variable is not set: %s\n' "${variable}" >&2
    exit 2
  fi
done
for required_file in "${ANCHOR_CHECKPOINT}" "${SUPPORT_CACHE}"; do
  if [[ ! -f "${required_file}" ]]; then
    printf 'Required file not found: %s\n' "${required_file}" >&2
    exit 2
  fi
done

ARTIFACT="${ROOT_DIR}/generation.pt"
REPORT="${ROOT_DIR}/counterfactual_evaluation.json"
LOG="${ROOT_DIR}/run.log"

generate_args=(
  --checkpoint "${ANCHOR_CHECKPOINT}"
  --output "${ARTIFACT}"
  --map-id "${MAP_ID}"
  --num-samples "${NUM_SAMPLES:-1000}"
  --total-power-db "${TOTAL_POWER_DB}"
  --delay-spread-ns "${DELAY_SPREAD_NS}"
  --k-factor-db "${K_FACTOR_DB:-0.0}"
  --los-status "${LOS_STATUS}"
  --strongest-aod-az-deg "${STRONGEST_AOD_DEG}"
  --path-count "${PATH_COUNT:-5}"
  --aod-support-cache "${SUPPORT_CACHE}"
  --minimum-aod-map-support "${MINIMUM_AOD_MAP_SUPPORT:-8}"
  --maximum-condition-distance "${MAXIMUM_CONDITION_DISTANCE:-1.5}"
  --maximum-angle-distance-deg "${MAXIMUM_ANGLE_DISTANCE_DEG:-10.0}"
  --guidance-scale "${GUIDANCE_SCALE:-2.0}"
  --device cuda
  --seed "${SEED:-2026}"
)
if [[ -n "${KNOWN_LOS_DIRECT_AOD_DEG:-}" ]]; then
  generate_args+=(
    --known-los-direct-aod-deg "${KNOWN_LOS_DIRECT_AOD_DEG}"
    --los-direct-tolerance-deg "${LOS_DIRECT_TOLERANCE_DEG:-10.0}"
  )
fi

{
  printf '[%s] generating class-B candidate\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)"
  "${PYTHON_BIN}" scripts/generate_scene_id_diffusion.py "${generate_args[@]}"
  printf '[%s] evaluating cross-scene physical reasonableness\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)"
  "${PYTHON_BIN}" scripts/evaluate_strongest_anchor_counterfactual.py \
    --artifact "${ARTIFACT}" \
    --support-cache "${SUPPORT_CACHE}" \
    --output "${REPORT}" \
    --maximum-condition-distance "${MAXIMUM_CONDITION_DISTANCE:-1.5}" \
    --maximum-angle-distance-deg "${MAXIMUM_ANGLE_DISTANCE_DEG:-10.0}" \
    --minimum-cross-scene-reference "${MINIMUM_CROSS_SCENE_REFERENCE:-8}" \
    --minimum-hit-at-10 "${MINIMUM_HIT_AT_10:-0.80}"
  printf 'artifact=%s\nreport=%s\n' "${ARTIFACT}" "${REPORT}"
} 2>&1 | tee "${LOG}"
