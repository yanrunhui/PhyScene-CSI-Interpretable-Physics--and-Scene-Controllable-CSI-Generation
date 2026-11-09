#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/multipath_diffusion_strongest_aod_anchor.yaml}"
SCENE_PLAN="${SCENE_PLAN:-artifacts/splits/scene_id_seen_map_v1.json}"
TRAIN_SEED="${TRAIN_SEED:-2026}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-artifacts/strongest_aod_anchor/${RUN_TAG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"

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
for required in "${CONFIG}" "${SCENE_PLAN}" "${INITIALIZE_FROM}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty>}" >&2
    exit 2
  fi
done

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"
on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
  fi
  printf '\ncheckpoint=%s\nlogs=%s\nexit_code=%s\n' \
    "${CHECKPOINT_DIR}/best.pt" "${LOG_DIR}" "${code}" | tee -a "${QUEUE_LOG}"
}
trap on_exit EXIT

run_stage() {
  local name="$1"
  shift
  printf '\n[%s] START %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] DONE %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
}

TRAIN_ARGS=(
  --config "${CONFIG}"
  --scene-plan "${SCENE_PLAN}"
  --initialize-from "${INITIALIZE_FROM}"
  --seed "${TRAIN_SEED}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --epochs "${ANCHOR_EPOCHS:-20}"
  --learning-rate "${ANCHOR_LEARNING_RATE:-5.0e-5}"
)

{
  printf 'project_root=%s\npython=%s\ngpu=%s\n' \
    "${PROJECT_ROOT}" "${PYTHON_BIN}" "${CUDA_VISIBLE_DEVICES}"
  printf 'config=%s\nscene_plan=%s\ninitialize_from=%s\ncheckpoint_dir=%s\n' \
    "${CONFIG}" "${SCENE_PLAN}" "${INITIALIZE_FROM}" "${CHECKPOINT_DIR}"
  printf 'training_objective=diffusion_plus_existing_physics\nanchor_parameterization=enabled\n'
} | tee "${QUEUE_LOG}"

if [[ "${SKIP_DRY_RUN:-0}" != "1" ]]; then
  run_stage 01_dry_run \
    "${PYTHON_BIN}" scripts/train_scene_id_diffusion.py \
      "${TRAIN_ARGS[@]}" --dry-run \
      --dry-run-batch-size "${DRY_RUN_BATCH_SIZE:-32}"
fi

run_stage 02_train \
  "${PYTHON_BIN}" scripts/train_scene_id_diffusion.py "${TRAIN_ARGS[@]}"

printf '\nStrongest Anchor checkpoint: %s\n' \
  "${CHECKPOINT_DIR}/best.pt" | tee -a "${QUEUE_LOG}"
