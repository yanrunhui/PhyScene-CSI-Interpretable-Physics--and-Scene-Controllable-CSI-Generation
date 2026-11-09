#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
CFG_SCALES="${CFG_SCALES:-1.0 1.5 2.0 3.0}"
CACHE="${CACHE:-artifacts/generated/delay_power_residual/b3_cache_seed2026.pt}"

if [[ -z "${STRONGEST_AOD_CHECKPOINT:-}" ]]; then
  STRONGEST_AOD_CHECKPOINT="$({
    find artifacts/checkpoints/strongest_aod_conditioned_b3 -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi
if [[ -z "${DP_CHECKPOINT:-}" ]]; then
  DP_CHECKPOINT="$({
    find artifacts/checkpoints/delay_power_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi
for required in "${STRONGEST_AOD_CHECKPOINT}" "${DP_CHECKPOINT}" "${CACHE}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty>}" >&2
    exit 2
  fi
done

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${OUTPUT:-artifacts/reports/strongest_aod_cfg_validation/${RUN_TAG}/summary.json}"
LOG_DIR="${LOG_DIR:-artifacts/logs/strongest_aod_cfg_scan/${RUN_TAG}}"
mkdir -p "$(dirname -- "${OUTPUT}")" "${LOG_DIR}"

printf 'aod_checkpoint=%s\ndp_checkpoint=%s\ncache=%s\ncfg_scales=%s\noutput=%s\n' \
  "${STRONGEST_AOD_CHECKPOINT}" "${DP_CHECKPOINT}" "${CACHE}" \
  "${CFG_SCALES}" "${OUTPUT}" \
  | tee "${LOG_DIR}/scan.log"

"${PYTHON_BIN}" scripts/evaluate_strongest_aod_cfg_validation.py \
  --aod-checkpoint "${STRONGEST_AOD_CHECKPOINT}" \
  --dp-checkpoint "${DP_CHECKPOINT}" \
  --cache "${CACHE}" \
  --output "${OUTPUT}" \
  --guidance-scales ${CFG_SCALES} \
  --maximum-samples-per-map "${MAXIMUM_SAMPLES_PER_MAP:-64}" \
  --candidates-per-query "${CANDIDATES_PER_QUERY:-2}" \
  --outputs-per-query "${OUTPUTS_PER_QUERY:-1}" \
  --natural-angle-tolerance-deg "${NATURAL_TOLERANCE_DEG:-10}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --device cuda \
  --seed "${SEED:-2026}" \
  2>&1 | tee "${LOG_DIR}/validation.log"

printf '\nValidation summary: %s\nLogs: %s\n' "${OUTPUT}" "${LOG_DIR}" \
  | tee -a "${LOG_DIR}/scan.log"
