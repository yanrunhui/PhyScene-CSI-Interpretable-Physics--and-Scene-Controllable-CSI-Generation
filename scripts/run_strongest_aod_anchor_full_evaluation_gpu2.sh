#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${ANCHOR_CHECKPOINT:-}" ]]; then
  ANCHOR_CHECKPOINT="$({
    find artifacts/strongest_aod_anchor artifacts/checkpoints/strongest_aod_anchor_b3 \
      -type f -path '*/checkpoints/best.pt' -printf '%T@ %p\n' 2>/dev/null || true
    find artifacts/checkpoints/strongest_aod_anchor_b3 \
      -type f -name best.pt -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -nr | head -1 | cut -d' ' -f2-)"
fi
if [[ -z "${ANCHOR_CHECKPOINT}" || ! -f "${ANCHOR_CHECKPOINT}" ]]; then
  printf 'Strongest Anchor checkpoint not found: %s\n' \
    "${ANCHOR_CHECKPOINT:-<empty>}" >&2
  printf 'Set ANCHOR_CHECKPOINT=/path/to/checkpoints/best.pt\n' >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-artifacts/strongest_aod_anchor_evaluation/${RUN_TAG}}"
printf 'anchor_checkpoint=%s\nroot_dir=%s\ngpu=%s\n' \
  "${ANCHOR_CHECKPOINT}" "${ROOT_DIR}" "${CUDA_VISIBLE_DEVICES}"

STRONGEST_AOD_CHECKPOINT="${ANCHOR_CHECKPOINT}" \
ROOT_DIR="${ROOT_DIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
REQUIRE_STRONGEST_ANCHOR=1 \
PRODUCTION_CFG="${PRODUCTION_CFG:-1}" \
BASELINE_HIT_AT_10="${BASELINE_HIT_AT_10:-0.687}" \
BASELINE_NEAREST_TOP1="${BASELINE_NEAREST_TOP1:-0.345}" \
MINIMUM_HIT_IMPROVEMENT="${MINIMUM_HIT_IMPROVEMENT:-0.03}" \
MINIMUM_TOP1_IMPROVEMENT="${MINIMUM_TOP1_IMPROVEMENT:-0.03}" \
MINIMUM_16_TO_8_SUCCESS_RATE="${MINIMUM_16_TO_8_SUCCESS_RATE:-0.95}" \
MAXIMUM_DP_RANKING_REGRESSION="${MAXIMUM_DP_RANKING_REGRESSION:-0.0}" \
bash scripts/run_strongest_aod_full_evaluation_gpu2.sh
