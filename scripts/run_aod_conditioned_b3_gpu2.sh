#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/multipath_diffusion_aod_conditioned.yaml}"
SCENE_PLAN="${SCENE_PLAN:-artifacts/splits/scene_id_seen_map_v1.json}"
TRAIN_SEED="${TRAIN_SEED:-2026}"

if [[ -z "${INITIALIZE_FROM:-}" ]]; then
  INITIALIZE_FROM="$({
    find artifacts/checkpoints/b3_scene_id -mindepth 1 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
  if [[ -z "${INITIALIZE_FROM}" \
        && -f artifacts/checkpoints/b3_scene_id/best.pt ]]; then
    INITIALIZE_FROM="artifacts/checkpoints/b3_scene_id/best.pt"
  fi
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints/aod_conditioned_b3/seed${TRAIN_SEED}_${RUN_TAG}}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/aod_conditioned_b3/seed${TRAIN_SEED}_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/aod_conditioned_b3/seed${TRAIN_SEED}_${RUN_TAG}}"
SMOKE_ARTIFACT="${REPORT_DIR}/generation_smoke.pt"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${CHECKPOINT_DIR}" "${REPORT_DIR}" "${LOG_DIR}"

for required in "${CONFIG}" "${SCENE_PLAN}" "${INITIALIZE_FROM}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty auto-detected path>}" >&2
    exit 2
  fi
done

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

run_stage() {
  local name="$1"
  shift
  printf '\n[%s] START %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] DONE  %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
}

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] AoD-conditioned B3 fine-tuning completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] AoD-conditioned B3 fine-tuning failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'config=%s\n' "${CONFIG}"
  printf 'scene_plan=%s\n' "${SCENE_PLAN}"
  printf 'initialize_from=%s\n' "${INITIALIZE_FROM}"
  printf 'training_seed=%s\n' "${TRAIN_SEED}"
  printf 'checkpoint_dir=%s\n' "${CHECKPOINT_DIR}"
  printf 'report_dir=%s\n' "${REPORT_DIR}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

if [[ "${SKIP_DRY_RUN:-0}" != "1" ]]; then
  run_stage 01_dry_run \
    "${PYTHON_BIN}" scripts/train_scene_id_diffusion.py \
      --config "${CONFIG}" \
      --scene-plan "${SCENE_PLAN}" \
      --initialize-from "${INITIALIZE_FROM}" \
      --seed "${TRAIN_SEED}" \
      --dry-run \
      --dry-run-batch-size "${DRY_RUN_BATCH_SIZE:-16}"
fi

run_stage 02_train \
  "${PYTHON_BIN}" scripts/train_scene_id_diffusion.py \
    --config "${CONFIG}" \
    --scene-plan "${SCENE_PLAN}" \
    --initialize-from "${INITIALIZE_FROM}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --seed "${TRAIN_SEED}"

run_stage 03_generation_smoke \
  "${PYTHON_BIN}" scripts/generate_scene_id_diffusion.py \
    --checkpoint "${CHECKPOINT_DIR}/best.pt" \
    --output "${SMOKE_ARTIFACT}" \
    --map-id "${SMOKE_MAP_ID:-10}" \
    --num-samples "${SMOKE_SAMPLES:-64}" \
    --total-power-db "${SMOKE_POWER_DB:--80}" \
    --delay-spread-ns "${SMOKE_DELAY_SPREAD_NS:-50}" \
    --k-factor-db 0 \
    --los-status 0 \
    --path-count 5 \
    --mean-aod-az-deg "${SMOKE_MEAN_AOD_DEG:-90}" \
    --skip-support-check \
    --device cuda \
    --seed 2026

"${PYTHON_BIN}" scripts/summarize_aod_conditioned_training.py \
  "${CHECKPOINT_DIR}/best.pt" \
  --generation-artifact "${SMOKE_ARTIFACT}" | tee "${SUMMARY}"

printf '\nCheckpoint: %s\nSmoke artifact: %s\nSummary: %s\nLogs: %s\n' \
  "${CHECKPOINT_DIR}/best.pt" "${SMOKE_ARTIFACT}" "${SUMMARY}" "${LOG_DIR}" \
  | tee -a "${QUEUE_LOG}"
