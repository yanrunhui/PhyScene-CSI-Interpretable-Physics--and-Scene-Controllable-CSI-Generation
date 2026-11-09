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
LINK_CHECKPOINT="${LINK_CHECKPOINT:-$(discover artifacts/checkpoints/aod_link_global_residual best.pt)}"
CACHE="${CACHE:-$(discover artifacts/generated/delay_power_residual 'b3_cache_seed*.pt')}"
for required in \
  "${STRONGEST_AOD_CHECKPOINT}" "${DP_CHECKPOINT}" "${LINK_CHECKPOINT}" "${CACHE}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty>}" >&2
    exit 2
  fi
done

if [[ -z "${MEAN_FIDELITY_SUMMARY:-}" ]]; then
  MEAN_FIDELITY_SUMMARY="$(discover artifacts/reports/aod_conditional_fidelity summary.json)"
fi
if [[ -n "${MEAN_FIDELITY_SUMMARY}" && ! -f "${MEAN_FIDELITY_SUMMARY}" ]]; then
  printf 'Mean fidelity summary not found: %s\n' "${MEAN_FIDELITY_SUMMARY}" >&2
  exit 2
fi
if [[ -z "${BASELINE_STRONGEST_FIDELITY_SUMMARY:-}" ]]; then
  BASELINE_STRONGEST_FIDELITY_SUMMARY="$(discover \
    artifacts/strongest_aod_evaluation 07_fidelity_summary.json)"
fi
if [[ -n "${BASELINE_STRONGEST_FIDELITY_SUMMARY}" \
      && ! -f "${BASELINE_STRONGEST_FIDELITY_SUMMARY}" ]]; then
  printf 'Baseline Strongest fidelity summary not found: %s\n' \
    "${BASELINE_STRONGEST_FIDELITY_SUMMARY}" >&2
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-artifacts/strongest_aod_evaluation/${RUN_TAG}}"
REPORT_DIR="${REPORT_DIR:-${ROOT_DIR}/reports}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/generated}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
mkdir -p "${REPORT_DIR}" "${ARTIFACT_DIR}" "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Strongest-AoD full evaluation completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Strongest-AoD full evaluation failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

run_stage() {
  local name="$1"
  shift
  printf '\n[%s] START %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] DONE  %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
}

ANCHOR_ARGS=()
if [[ "${REQUIRE_STRONGEST_ANCHOR:-0}" == "1" ]]; then
  ANCHOR_ARGS+=(--require-strongest-anchor)
fi

CFG_REPORT="${REPORT_DIR}/01_cfg_validation.json"
STABILITY_REPORT="${REPORT_DIR}/02_candidate_16_to_8.json"
VAL_SUPPORT="${REPORT_DIR}/03_validation_support.json"
VAL_PLAN="${ARTIFACT_DIR}/validation_pair_plan.pt"
VAL_ARTIFACT="${ARTIFACT_DIR}/validation_variants.pt"
VAL_FIDELITY="${REPORT_DIR}/04_validation_fidelity.json"
TEST_SUPPORT="${REPORT_DIR}/05_test_support.json"
TEST_PLAN="${ARTIFACT_DIR}/test_pair_plan.pt"
TEST_ARTIFACT="${ARTIFACT_DIR}/test_variants.pt"
TEST_FIDELITY="${REPORT_DIR}/06_test_fidelity.json"
FIDELITY_SUMMARY="${REPORT_DIR}/07_fidelity_summary.json"
DP_TRANSITIONS="${REPORT_DIR}/08_dp_rank_transitions.json"
RECALL_REPORT="${REPORT_DIR}/09_multipath_recall.json"
RECALL_SUMMARY="${REPORT_DIR}/10_multipath_recall_summary.json"
FINAL_SUMMARY="${REPORT_DIR}/summary.json"

{
  printf 'project_root=%s\npython=%s\ncuda_visible_devices=%s\n' \
    "${PROJECT_ROOT}" "${PYTHON_BIN}" "${CUDA_VISIBLE_DEVICES}"
  printf 'strongest_aod_checkpoint=%s\ndp_checkpoint=%s\nlink_checkpoint=%s\ncache=%s\n' \
    "${STRONGEST_AOD_CHECKPOINT}" "${DP_CHECKPOINT}" "${LINK_CHECKPOINT}" "${CACHE}"
  printf 'mean_fidelity_summary=%s\nroot_dir=%s\n' \
    "${MEAN_FIDELITY_SUMMARY:-<not provided>}" "${ROOT_DIR}"
  printf 'baseline_strongest_fidelity_summary=%s\n' \
    "${BASELINE_STRONGEST_FIDELITY_SUMMARY:-<not provided>}"
} | tee "${QUEUE_LOG}"

run_stage 01_cfg_validation \
  "${PYTHON_BIN}" scripts/evaluate_strongest_aod_cfg_validation.py \
  --aod-checkpoint "${STRONGEST_AOD_CHECKPOINT}" \
  --dp-checkpoint "${DP_CHECKPOINT}" --cache "${CACHE}" \
  --output "${CFG_REPORT}" --guidance-scales 1 2 3 \
  --maximum-samples-per-map "${CFG_MAXIMUM_SAMPLES_PER_MAP:-64}" \
  --candidates-per-query 2 --outputs-per-query 1 \
  --natural-angle-tolerance-deg 10 --batch-size "${BATCH_SIZE:-64}" \
  "${ANCHOR_ARGS[@]}" \
  --device cuda --seed "${SEED:-2026}"

run_stage 02_candidate_16_to_8 \
  "${PYTHON_BIN}" scripts/evaluate_strongest_aod_cfg_validation.py \
  --aod-checkpoint "${STRONGEST_AOD_CHECKPOINT}" \
  --dp-checkpoint "${DP_CHECKPOINT}" --cache "${CACHE}" \
  --output "${STABILITY_REPORT}" --guidance-scales "${PRODUCTION_CFG:-3}" \
  --maximum-samples-per-map "${STABILITY_MAXIMUM_SAMPLES_PER_MAP:-16}" \
  --candidates-per-query 16 --outputs-per-query 8 \
  --natural-angle-tolerance-deg 10 --batch-size "${STABILITY_BATCH_SIZE:-16}" \
  "${ANCHOR_ARGS[@]}" \
  --device cuda --seed "${SEED:-2026}"

support_stage() {
  local split="$1" report="$2" plan="$3"
  run_stage "03_support_${split}" \
    "${PYTHON_BIN}" scripts/diagnose_map_condition_aod_support.py \
    --cache "${CACHE}" --split "${split}" --output-report "${report}" \
    --output-plan "${plan}" --condition-type strongest --path-count 5 \
    --maximum-condition-distance "${CONDITION_RADIUS:-1.5}" \
    --maximum-angle-distance-deg "${ANGLE_RADIUS_DEG:-10}" \
    --minimum-support "${MINIMUM_SUPPORT:-8}" \
    --maximum-pairs-per-map-los "${MAXIMUM_PAIRS_PER_MAP_LOS:-16}" \
    --seed "${SEED:-2026}"
}

generate_stage() {
  local split="$1" plan="$2" output="$3"
  run_stage "04_generate_${split}" \
    "${PYTHON_BIN}" scripts/generate_aod_conditional_fidelity_artifact.py \
    --aod-checkpoint "${STRONGEST_AOD_CHECKPOINT}" \
    --dp-checkpoint "${DP_CHECKPOINT}" --link-checkpoint "${LINK_CHECKPOINT}" \
    --cache "${CACHE}" --pair-plan "${plan}" --output "${output}" \
    --condition-type strongest --batch-size "${FIDELITY_BATCH_SIZE:-128}" \
    --device cuda --seed "${SEED:-2026}"
}

evaluate_stage() {
  local split="$1" artifact="$2" output="$3" selection="${4:-}"
  local selection_args=()
  if [[ -n "${selection}" ]]; then
    selection_args+=(--selection-from "${selection}")
  fi
  run_stage "05_evaluate_${split}" \
    "${PYTHON_BIN}" scripts/evaluate_aod_conditional_fidelity.py \
    --artifact "${artifact}" --output "${output}" "${selection_args[@]}" \
    --device cuda --minimum-samples-per-map "${MINIMUM_SAMPLES_PER_MAP:-8}" \
    --projection-dimension "${PROJECTION_DIMENSION:-64}" \
    --projection-seeds "${PROJECTION_SEEDS:-0,1,2}" \
    --swd-projections "${SWD_PROJECTIONS:-128}" \
    --weighted-quantiles "${WEIGHTED_QUANTILES:-256}" \
    --coverage-k "${COVERAGE_K:-3}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" --seed "${SEED:-2026}"
}

support_stage validation "${VAL_SUPPORT}" "${VAL_PLAN}"
generate_stage validation "${VAL_PLAN}" "${VAL_ARTIFACT}"
evaluate_stage validation "${VAL_ARTIFACT}" "${VAL_FIDELITY}"
support_stage test "${TEST_SUPPORT}" "${TEST_PLAN}"
generate_stage test "${TEST_PLAN}" "${TEST_ARTIFACT}"
evaluate_stage test "${TEST_ARTIFACT}" "${TEST_FIDELITY}" "${VAL_FIDELITY}"

run_stage 06_summarize_fidelity \
  "${PYTHON_BIN}" scripts/summarize_aod_conditional_fidelity.py \
  --support-validation "${VAL_SUPPORT}" --support-test "${TEST_SUPPORT}" \
  --validation "${VAL_FIDELITY}" --test "${TEST_FIDELITY}" \
  --output "${FIDELITY_SUMMARY}"

run_stage 07_dp_rank_transitions \
  "${PYTHON_BIN}" scripts/diagnose_strongest_dp_rank_transitions.py \
  --artifact "${TEST_ARTIFACT}" --output "${DP_TRANSITIONS}" --device cuda \
  --minimum-samples-per-map "${MINIMUM_SAMPLES_PER_MAP:-8}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" --seed "${SEED:-2026}"

run_stage 08_multipath_recall \
  "${PYTHON_BIN}" scripts/diagnose_cfg_multipath_recall.py \
  --artifact "${TEST_ARTIFACT}" --output "${RECALL_REPORT}" --device cuda \
  --minimum-samples-per-map "${MINIMUM_SAMPLES_PER_MAP:-8}" \
  --coverage-k "${COVERAGE_K:-3}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" --seed "${SEED:-2026}"

run_stage 09_summarize_recall \
  "${PYTHON_BIN}" scripts/summarize_cfg_multipath_recall.py \
  --report "${RECALL_REPORT}" --output "${RECALL_SUMMARY}"

summary_args=()
if [[ -n "${MEAN_FIDELITY_SUMMARY}" ]]; then
  summary_args+=(--mean-fidelity-summary "${MEAN_FIDELITY_SUMMARY}")
fi
if [[ -n "${BASELINE_STRONGEST_FIDELITY_SUMMARY}" ]]; then
  summary_args+=(
    --baseline-strongest-fidelity-summary \
    "${BASELINE_STRONGEST_FIDELITY_SUMMARY}"
  )
fi
run_stage 10_final_summary \
  "${PYTHON_BIN}" scripts/summarize_strongest_aod_evaluation.py \
  --cfg "${CFG_REPORT}" --candidate-stability "${STABILITY_REPORT}" \
  --fidelity "${FIDELITY_SUMMARY}" --recall "${RECALL_SUMMARY}" \
  --dp-transitions "${DP_TRANSITIONS}" \
  --production-cfg "${PRODUCTION_CFG:-3}" \
  --minimum-16-to-8-success-rate "${MINIMUM_16_TO_8_SUCCESS_RATE:-0.95}" \
  --baseline-unfiltered-hit-at-10 "${BASELINE_HIT_AT_10:-0.687}" \
  --minimum-hit-at-10-improvement "${MINIMUM_HIT_IMPROVEMENT:-0.03}" \
  --baseline-target-nearest-top1-rate "${BASELINE_NEAREST_TOP1:-0.345}" \
  --minimum-target-nearest-top1-improvement "${MINIMUM_TOP1_IMPROVEMENT:-0.03}" \
  --maximum-dp-ranking-regression "${MAXIMUM_DP_RANKING_REGRESSION:-0.01}" \
  --maximum-tx-spatial-regression "${MAXIMUM_TX_SPATIAL_REGRESSION:-0.0}" \
  --maximum-recall-regression "${MAXIMUM_RECALL_REGRESSION:-0.01}" \
  --maximum-power-mae-db "${MAXIMUM_POWER_MAE_DB:-0.05}" \
  --maximum-delay-spread-mae-ns "${MAXIMUM_DS_MAE_NS:-0.5}" \
  --maximum-k-factor-mae-db "${MAXIMUM_K_MAE_DB:-0.1}" \
  --minimum-anchor-pre-repair-hit-at-10 \
    "${MINIMUM_ANCHOR_PRE_REPAIR_HIT_AT_10:-0.80}" \
  --output "${FINAL_SUMMARY}" "${summary_args[@]}"

printf '\nSummary: %s\nReports: %s\nArtifacts: %s\nLogs: %s\n' \
  "${FINAL_SUMMARY}" "${REPORT_DIR}" "${ARTIFACT_DIR}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
