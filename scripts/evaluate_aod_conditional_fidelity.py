#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.evaluation.aod_support import (
    circular_distance_degrees,
    conditioned_aod_measurement,
    strongest_aod,
)
from physics_csi.evaluation.scene_diagnostics import paired_map_summary
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump
from scripts.evaluate_csi_absolute_quality import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    METRIC_NAMES,
    Evaluator,
)


MEAN_CANDIDATE_LABELS = (
    "aod_cfg1_dp",
    "aod_cfg2_dp",
    "aod_cfg3_dp",
    "aod_cfg2_dp_link",
    "aod_cfg3_dp_link",
)
BASELINE_LABEL = "baseline_b3_dp_link"
FIDELITY_SELECTION_WEIGHTS = {
    "aod_error_deg": 0.30,
    "path_level_power_aod_sliced_wasserstein": 0.15,
    "link_level_power_aod_sliced_wasserstein": 0.15,
    "power_weighted_aod_sliced_wasserstein": 0.10,
    "tx_spatial_correlation_relative_frobenius": 0.15,
    "phase_invariant_full_csi_projected_sliced_wasserstein": 0.10,
    "pdp_sliced_wasserstein": 0.05,
}


def _to_device(payload: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }


def _control_metrics(
    payload: Mapping[str, torch.Tensor],
    target_angle: torch.Tensor,
    support_count: torch.Tensor | None,
    minimum_support: int,
    condition_type: str,
) -> dict[str, Any]:
    actual, valid, confidence = conditioned_aod_measurement(
        payload, condition_type=condition_type
    )
    error = circular_distance_degrees(actual, target_angle)
    error = error[valid]
    if not error.numel():
        raise ValueError("controlled AoD measurement has no valid samples")
    result: dict[str, Any] = {
        "samples": int(error.numel()),
        "aod_error_deg": float(error.mean()),
        "p95_aod_error_deg": float(torch.quantile(error, 0.95)),
        "hit_at_5deg": float((error <= 5.0).float().mean()),
        "hit_at_10deg": float((error <= 10.0).float().mean()),
        "controlled_aod_valid_rate": float(valid.float().mean()),
        "path_count_accuracy": float(
            (
                payload["path_mask"].bool().sum(-1)
                == payload["path_mask"].shape[-1]
            )
            .float()
            .mean()
        ),
    }
    result[f"{condition_type}_aod_error_deg"] = result["aod_error_deg"]
    if condition_type == "strongest":
        measurement = strongest_aod(payload)
        finite_gap = measurement.top2_power_gap_db[
            measurement.valid & torch.isfinite(measurement.top2_power_gap_db)
        ]
        path_mask = payload["path_mask"].bool()
        per_path_error = circular_distance_degrees(
            payload["aod_az_deg"].float(), target_angle.float().unsqueeze(-1)
        ).masked_fill(~path_mask, torch.inf)
        target_nearest_index = per_path_error.argmin(-1)
        target_nearest_power = payload["path_powers_linear"].float().gather(
            1, target_nearest_index.unsqueeze(-1)
        ).squeeze(-1)
        target_nearest_rank = 1 + (
            (payload["path_powers_linear"].float() > target_nearest_power.unsqueeze(-1))
            & path_mask
        ).sum(-1)
        nearest_valid = measurement.valid & path_mask.any(-1)
        result["strongest_near_tie_rate"] = float(
            measurement.near_tie.float().mean()
        )
        result["strongest_top2_power_gap_db_mean"] = (
            float(finite_gap.mean()) if finite_gap.numel() else None
        )
        result["target_nearest_path_top1_rate"] = float(
            (
                target_nearest_index[nearest_valid]
                == measurement.path_index[nearest_valid]
            )
            .float()
            .mean()
        )
        result["target_nearest_path_power_rank_mean"] = float(
            target_nearest_rank[nearest_valid].float().mean()
        )
        result["target_nearest_path_error_deg_mean"] = float(
            per_path_error.gather(
                1, target_nearest_index.unsqueeze(-1)
            )[nearest_valid].mean()
        )
        anchor_pre_repair = payload.get("strongest_anchor_pre_repair_aod_deg")
        if anchor_pre_repair is not None:
            anchor_pre_repair_error = circular_distance_degrees(
                anchor_pre_repair.float(), target_angle.float()
            )
        else:
            anchor_offset = payload.get("strongest_anchor_predicted_offset_deg")
            anchor_pre_repair_error = (
                anchor_offset.float().abs() if anchor_offset is not None else None
            )
        if anchor_pre_repair_error is not None:
            finite_anchor = anchor_pre_repair_error[
                torch.isfinite(anchor_pre_repair_error)
            ]
        else:
            finite_anchor = torch.empty(0, device=target_angle.device)
        if finite_anchor.numel():
            result["anchor_pre_repair_error_deg_mean"] = float(
                finite_anchor.mean()
            )
            result["anchor_pre_repair_error_deg_p95"] = float(
                torch.quantile(finite_anchor, 0.95)
            )
            result["anchor_pre_repair_hit_at_10deg"] = float(
                (finite_anchor <= 10.0).float().mean()
            )
            result["anchor_structural_correction_over_10deg_rate"] = float(
                (finite_anchor > 10.0).float().mean()
            )
    elif bool(valid.any()):
        result["mean_aod_concentration_mean"] = float(confidence[valid].mean())
    result["expected_candidates_per_accepted_at_10deg"] = 1.0 / max(
        float(result["hit_at_10deg"]), 1e-12
    )
    feasible = payload.get("constraint_feasible")
    result["physical_feasible_rate"] = (
        float(feasible.float().mean()) if feasible is not None else None
    )
    if support_count is not None:
        result["generated_high_support_rate"] = float(
            (support_count >= minimum_support).float().mean()
        )
        result["generated_support_count_mean"] = float(support_count.float().mean())
        result["generated_support_count_p05"] = float(
            torch.quantile(support_count.float(), 0.05)
        )
    actual_conditions = payload.get("actual_conditions")
    target_conditions = payload.get("conditions")
    validity = payload.get("condition_validity")
    if actual_conditions is not None and target_conditions is not None and validity is not None:
        names = ("power_db", "delay_spread_ns", "k_factor_db", "los_status")
        absolute = (actual_conditions.float() - target_conditions.float()).abs()
        result["physical_condition_mae"] = {
            name: float(absolute[validity[:, index].bool(), index].mean())
            if bool(validity[:, index].any())
            else None
            for index, name in enumerate(names)
        }
    return result


def _macro(records: list[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in labels:
        current = [record for record in records if label in record["models"]]
        if not current:
            continue
        fidelity = {
            metric: sum(record["models"][label][metric] for record in current)
            / len(current)
            for metric in METRIC_NAMES
        }
        control_keys = (
            "aod_error_deg",
            "mean_aod_error_deg",
            "strongest_aod_error_deg",
            "p95_aod_error_deg",
            "hit_at_5deg",
            "hit_at_10deg",
            "path_count_accuracy",
            "physical_feasible_rate",
            "generated_high_support_rate",
            "generated_support_count_mean",
            "controlled_aod_valid_rate",
            "strongest_near_tie_rate",
            "strongest_top2_power_gap_db_mean",
            "target_nearest_path_top1_rate",
            "target_nearest_path_power_rank_mean",
            "target_nearest_path_error_deg_mean",
            "anchor_pre_repair_error_deg_mean",
            "anchor_pre_repair_error_deg_p95",
            "anchor_pre_repair_hit_at_10deg",
            "anchor_structural_correction_over_10deg_rate",
            "mean_aod_concentration_mean",
        )
        control = {
            key: sum(
                float(record["control"][label][key]) for record in current
            )
            / len(current)
            for key in control_keys
            if all(
                record["control"][label].get(key) is not None for record in current
            )
        }
        result[label] = {
            "maps_or_map_los_groups": len(current),
            "fidelity": fidelity,
            "control": control,
        }
        if "hit_at_10deg" in control:
            result[label]["control"][
                "expected_candidates_per_accepted_at_10deg"
            ] = 1.0 / max(float(control["hit_at_10deg"]), 1e-12)
        physical_fields = sorted(
            {
                field
                for record in current
                for field in record["control"][label].get(
                    "physical_condition_mae", {}
                )
            }
        )
        if physical_fields:
            result[label]["control"]["physical_condition_mae"] = {
                field: sum(
                    float(record["control"][label]["physical_condition_mae"][field])
                    for record in current
                    if record["control"][label]["physical_condition_mae"].get(field)
                    is not None
                )
                / max(
                    sum(
                        record["control"][label]["physical_condition_mae"].get(field)
                        is not None
                        for record in current
                    ),
                    1,
                )
                for field in physical_fields
            }
    return result


def _rank_selection(
    macro: Mapping[str, Any],
    candidate_labels: tuple[str, ...] = MEAN_CANDIDATE_LABELS,
) -> dict[str, Any]:
    available = [label for label in candidate_labels if label in macro]
    if not available:
        return {
            "selected_label": None,
            "gate_status": "unavailable_no_supported_map_groups",
            "eligible_labels": [],
            "weighted_rank_score_lower_is_better": {},
            "per_metric_rank": {},
            "predeclared_weights": FIDELITY_SELECTION_WEIGHTS,
            "minimum_hit_at_10deg": 0.80,
            "minimum_physical_feasible_rate": 0.99,
        }
    eligible = [
        label
        for label in available
        if macro[label]["control"].get("hit_at_10deg", 0.0) >= 0.80
        and macro[label]["control"].get("physical_feasible_rate", 0.0) >= 0.99
    ]
    if not eligible:
        eligible = available
        gate_status = "no_candidate_met_hit10_and_feasibility_gate"
    else:
        gate_status = "eligible_candidates_met_hit10_and_feasibility_gate"
    scores = {label: 0.0 for label in eligible}
    ranks: dict[str, dict[str, int]] = {}
    for metric, weight in FIDELITY_SELECTION_WEIGHTS.items():
        values = {
            label: (
                macro[label]["control"][metric]
                if metric == "aod_error_deg"
                else macro[label]["fidelity"][metric]
            )
            for label in eligible
        }
        ordered = sorted(values, key=values.get)
        metric_ranks = {label: rank for rank, label in enumerate(ordered)}
        ranks[metric] = metric_ranks
        denominator = max(len(eligible) - 1, 1)
        for label in eligible:
            scores[label] += weight * metric_ranks[label] / denominator
    selected = min(
        eligible,
        key=lambda label: (
            scores[label], macro[label]["control"]["aod_error_deg"]
        ),
    )
    return {
        "selected_label": selected,
        "gate_status": gate_status,
        "eligible_labels": eligible,
        "weighted_rank_score_lower_is_better": scores,
        "per_metric_rank": ranks,
        "predeclared_weights": FIDELITY_SELECTION_WEIGHTS,
        "minimum_hit_at_10deg": 0.80,
        "minimum_physical_feasible_rate": 0.99,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AoD-conditioned CFG/Link variants against independent real donors."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-from", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum-samples-per-map", type=int, default=8)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--projection-seeds", default="0,1,2")
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--weighted-quantiles", type=int, default=256)
    parser.add_argument("--coverage-k", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    device = resolve_device(args.device)
    source = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if source.get("model_kind") not in {
        "aod_conditioned_cfg_link_fidelity_comparison",
        "aod_conditioned_cfg_fidelity_comparison",
    }:
        raise ValueError("unsupported conditional-fidelity artifact")
    condition_type = source.get("condition_type", "mean")
    candidate_labels = tuple(
        label
        for label in MEAN_CANDIDATE_LABELS
        if label in source["payloads"]
    )
    if not candidate_labels:
        raise ValueError("conditional-fidelity artifact has no CFG candidates")
    labels = ("real_query", BASELINE_LABEL, *candidate_labels)
    payloads = {
        "reference": _to_device(source["payloads"]["reference"], device),
        **{
            label: _to_device(source["payloads"][label], device) for label in labels
        },
    }
    target = source.get("target_aod_deg", source.get("target_mean_aod_deg"))
    if not isinstance(target, torch.Tensor):
        raise ValueError("conditional-fidelity artifact has no controlled AoD target")
    target = target.to(device)
    generated_support = {
        label: values.to(device)
        for label, values in source["generated_real_support_count"].items()
    }
    minimum_support = int(source["support_protocol"]["minimum_real_neighbors"])
    evaluator = Evaluator(
        payloads,
        projection_dimension=args.projection_dimension,
        projection_seeds=tuple(
            int(value) for value in args.projection_seeds.split(",") if value
        ),
        projections=args.swd_projections,
        quantiles=args.weighted_quantiles,
        coverage_k=args.coverage_k,
    )
    map_ids = payloads["reference"]["map_id"].long()
    los = payloads["real_query"]["conditions"][:, 3] >= 0.5
    records: list[dict[str, Any]] = []
    map_los_records: list[dict[str, Any]] = []
    for map_id_tensor in torch.unique(map_ids, sorted=True):
        map_id = int(map_id_tensor)
        indices = torch.nonzero(map_ids == map_id, as_tuple=False).flatten()
        if indices.numel() < args.minimum_samples_per_map:
            continue
        models = {
            label: evaluator.evaluate(label, indices, indices)
            for label in labels
        }
        # Restrict control metrics to this Map rather than the full aligned artifact.
        control = {
            label: _control_metrics(
                {key: value[indices] for key, value in payloads[label].items()},
                target[indices],
                generated_support.get(label)[indices]
                if label in generated_support
                else None,
                minimum_support,
                condition_type,
            )
            for label in labels
        }
        records.append(
            {"map_id": map_id, "samples": int(indices.numel()), "models": models, "control": control}
        )
        for los_value in (0, 1):
            local = indices[los[indices] == bool(los_value)]
            if local.numel() < max(4, args.minimum_samples_per_map // 2):
                continue
            map_los_records.append(
                {
                    "map_id": map_id,
                    "los_status": los_value,
                    "samples": int(local.numel()),
                    "models": {
                        label: evaluator.evaluate(label, local, local, local=True)
                        for label in labels
                    },
                    "control": {
                        label: _control_metrics(
                            {key: value[local] for key, value in payloads[label].items()},
                            target[local],
                            generated_support.get(label)[local]
                            if label in generated_support
                            else None,
                            minimum_support,
                            condition_type,
                        )
                        for label in labels
                    },
                }
            )
        print(
            json.dumps(
                {"event": "conditional_fidelity_map_complete", "map_id": map_id}
            ),
            flush=True,
        )

    if not records:
        raise RuntimeError("no Map has enough supported pairs for fidelity evaluation")
    all_labels = labels
    macro = _macro(records, all_labels)
    by_los = {
        str(los_value): _macro(
            [record for record in map_los_records if record["los_status"] == los_value],
            all_labels,
        )
        for los_value in (0, 1)
    }
    comparisons: dict[str, Any] = {}
    for label in candidate_labels:
        section: dict[str, Any] = {}
        for metric in (
            "path_level_power_aod_sliced_wasserstein",
            "link_level_power_aod_sliced_wasserstein",
            "power_weighted_aod_sliced_wasserstein",
            "tx_spatial_correlation_relative_frobenius",
            "phase_invariant_full_csi_projected_mmd_rbf",
            "phase_invariant_full_csi_projected_sliced_wasserstein",
            "pdp_sliced_wasserstein",
            "multipath_recall",
            "multipath_coverage",
        ):
            first = [record["models"][label][metric] for record in records]
            second = [record["models"][BASELINE_LABEL][metric] for record in records]
            if metric in HIGHER_IS_BETTER:
                first, second = [-value for value in first], [-value for value in second]
            section[metric] = paired_map_summary(
                first,
                second,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + len(section),
            )
        comparisons[f"{label}_vs_{BASELINE_LABEL}"] = section

    validation_selection = None
    frozen_selection = None
    if source["split"] == "validation":
        unified_selection = _rank_selection(macro, candidate_labels)
        if unified_selection["selected_label"] is None:
            raise RuntimeError(
                "Validation has no supported Map groups for unified model selection"
            )
        los_selection: dict[str, Any] = {}
        for los_value in ("0", "1"):
            current = _rank_selection(by_los[los_value], candidate_labels)
            if current["selected_label"] is None:
                current = dict(current)
                current["selected_label"] = unified_selection["selected_label"]
                current["fallback_to_unified"] = True
                current["fallback_reason"] = (
                    "No supported Map-LoS groups survived the minimum-sample protocol; "
                    "the pre-Test unified Validation selection is used instead."
                )
            else:
                current["fallback_to_unified"] = False
            los_selection[los_value] = current
        validation_selection = {
            "unified": unified_selection,
            "by_los": los_selection,
        }
    if args.selection_from:
        with Path(args.selection_from).open("r", encoding="utf-8") as handle:
            selection_source = json.load(handle)
        frozen_selection = selection_source.get("validation_selection")
        if not frozen_selection:
            raise ValueError("--selection-from has no validation_selection")

    report = {
        "format_version": 1,
        "evaluation": "map_condition_aod_conditional_fidelity",
        "condition_type": condition_type,
        "split": source["split"],
        "artifact": str(Path(args.artifact).resolve()),
        "metric_direction": {
            "lower_is_better": sorted(LOWER_IS_BETTER),
            "higher_is_better": sorted(HIGHER_IS_BETTER),
        },
        "protocol": {
            "independent_reference": (
                "Generation targets come from query real links; fidelity references are "
                "nearby donors from a disjoint real half."
            ),
            "macro_unit": "Map",
            "support_protocol": source["support_protocol"],
            "all_residual_csi_resynthesized": True,
            "selection_policy": (
                "Validation-only weighted ranks with a predeclared Hit@10 and physical "
                "feasibility gate; Test consumes the frozen Validation selection."
            ),
            "candidate_labels": list(candidate_labels),
        },
        "normalization_audit": evaluator.audit,
        "macro_per_map": macro,
        "macro_per_map_by_los": by_los,
        "candidate_vs_original_baseline": comparisons,
        "validation_selection": validation_selection,
        "frozen_validation_selection_used_for_test": frozen_selection,
        "per_map": records,
        "per_map_los": map_los_records,
        "cautions": [
            "The current checkpoint used joint condition dropout, so CFG is not AoD-only.",
            "High-support evaluation establishes in-distribution fidelity; it does not validate rare or unsupported target combinations.",
            "One generation seed tests configuration quality, not independent training-seed stability.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "aod_conditional_fidelity_evaluation_complete",
                "output": str(Path(args.output).resolve()),
                "split": source["split"],
                "maps": len(records),
                "validation_selection": validation_selection,
                "frozen_selection": frozen_selection,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
