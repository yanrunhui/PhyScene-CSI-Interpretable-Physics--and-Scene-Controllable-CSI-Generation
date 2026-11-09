#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPATIAL_METRICS = (
    "path_level_power_aod_sliced_wasserstein",
    "link_level_power_aod_sliced_wasserstein",
    "power_weighted_aod_sliced_wasserstein",
    "tx_spatial_correlation_relative_frobenius",
    "phase_invariant_full_csi_projected_sliced_wasserstein",
    "pdp_sliced_wasserstein",
    "multipath_recall",
    "multipath_coverage",
)


def _load(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _cfg_key(values: dict[str, Any], scale: float) -> str:
    key = f"{scale:g}"
    if key not in values.get("cfg", {}):
        raise ValueError(f"CFG report has no scale {key}")
    return key


def _selected_test_result(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    label = summary["validation_selection"]["unified"]["selected_label"]
    result = summary["test_key_results"].get(label)
    if not isinstance(result, dict):
        raise ValueError(f"fidelity summary has no frozen selected Test result: {label}")
    return label, result


def _optional_mean_comparison(
    strongest: dict[str, Any], mean_summary_path: str | None
) -> dict[str, Any]:
    if not mean_summary_path:
        return {
            "status": "not_provided",
            "gate": None,
            "reason": "Set --mean-fidelity-summary for the Strongest-vs-Mean spatial gate.",
        }
    mean_summary = _load(mean_summary_path)
    mean_label, mean_result = _selected_test_result(mean_summary)
    strong_tx = float(
        strongest["fidelity"]["tx_spatial_correlation_relative_frobenius"]
    )
    mean_tx = float(
        mean_result["fidelity"]["tx_spatial_correlation_relative_frobenius"]
    )
    return {
        "status": "evaluated",
        "mean_selected_label": mean_label,
        "strongest_tx_spatial_correlation_relative_frobenius": strong_tx,
        "mean_tx_spatial_correlation_relative_frobenius": mean_tx,
        "strongest_minus_mean": strong_tx - mean_tx,
        "gate": strong_tx <= mean_tx,
        "rule": "Lower is better; Strongest must not exceed frozen Mean Test error.",
    }


def _optional_strongest_baseline_comparison(
    current: dict[str, Any], baseline_summary_path: str | None
) -> dict[str, Any]:
    if not baseline_summary_path:
        return {
            "status": "not_provided",
            "reason": (
                "Set --baseline-strongest-fidelity-summary to enforce Tx and "
                "Multipath Recall non-regression."
            ),
        }
    baseline_summary = _load(baseline_summary_path)
    if baseline_summary.get("condition_type") != "strongest":
        raise ValueError("baseline fidelity summary is not Strongest-AoD")
    current_label = current["label"]
    baseline_result = baseline_summary["test_key_results"].get(current_label)
    if not isinstance(baseline_result, dict):
        raise ValueError(
            f"baseline Strongest summary has no Test result for {current_label}"
        )
    current_result = current["result"]
    metrics = {}
    for metric in (
        "tx_spatial_correlation_relative_frobenius",
        "multipath_recall",
        "multipath_coverage",
        "power_weighted_aod_sliced_wasserstein",
    ):
        current_value = float(current_result["fidelity"][metric])
        baseline_value = float(baseline_result["fidelity"][metric])
        metrics[metric] = {
            "current": current_value,
            "baseline": baseline_value,
            "current_minus_baseline": current_value - baseline_value,
        }
    return {
        "status": "evaluated",
        "label": current_label,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine the frozen Strongest-AoD evaluation protocol into one decision."
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--candidate-stability", required=True)
    parser.add_argument("--fidelity", required=True)
    parser.add_argument("--recall", required=True)
    parser.add_argument("--dp-transitions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--production-cfg", type=float, default=3.0)
    parser.add_argument("--minimum-16-to-8-success-rate", type=float, default=0.95)
    parser.add_argument("--baseline-unfiltered-hit-at-10", type=float, default=0.687)
    parser.add_argument("--minimum-hit-at-10-improvement", type=float, default=0.03)
    parser.add_argument(
        "--baseline-target-nearest-top1-rate", type=float, default=0.345
    )
    parser.add_argument("--minimum-target-nearest-top1-improvement", type=float, default=0.03)
    parser.add_argument("--maximum-dp-ranking-regression", type=float, default=0.01)
    parser.add_argument("--mean-fidelity-summary")
    parser.add_argument("--baseline-strongest-fidelity-summary")
    parser.add_argument("--maximum-tx-spatial-regression", type=float, default=0.0)
    parser.add_argument("--maximum-recall-regression", type=float, default=0.01)
    parser.add_argument("--maximum-power-mae-db", type=float, default=0.05)
    parser.add_argument("--maximum-delay-spread-mae-ns", type=float, default=0.5)
    parser.add_argument("--maximum-k-factor-mae-db", type=float, default=0.1)
    parser.add_argument(
        "--minimum-anchor-pre-repair-hit-at-10", type=float, default=0.80
    )
    args = parser.parse_args()

    cfg = _load(args.cfg)
    stability = _load(args.candidate_stability)
    fidelity = _load(args.fidelity)
    recall = _load(args.recall)
    dp_transitions = _load(args.dp_transitions) if args.dp_transitions else None
    for name, values in (("cfg", cfg), ("stability", stability)):
        if values.get("analysis") != "strongest_aod_cfg_validation":
            raise ValueError(f"{name} is not a Strongest-AoD CFG report")
    if fidelity.get("condition_type") != "strongest":
        raise ValueError("fidelity summary is not Strongest-AoD")
    if recall.get("condition_type") != "strongest":
        raise ValueError("recall summary is not Strongest-AoD")

    cfg_key = _cfg_key(cfg, args.production_cfg)
    stability_key = _cfg_key(stability, args.production_cfg)
    cfg_result = cfg["cfg"][cfg_key]
    stability_result = stability["cfg"][stability_key]
    selected_label, selected_test = _selected_test_result(fidelity)
    production_label = f"aod_cfg{args.production_cfg:g}_dp"
    production_test = fidelity["test_key_results"].get(production_label)
    if not isinstance(production_test, dict):
        raise ValueError(
            f"fidelity summary has no production CFG result: {production_label}"
        )
    stability_rate = float(stability_result["natural_filter_query_success_rate"])
    recall_decision = recall["decision"]
    recall_gate = recall_decision["classification"] != "evidence_of_within_condition_mode_loss"
    spatial_comparison = _optional_mean_comparison(
        production_test, args.mean_fidelity_summary
    )
    strongest_baseline = _optional_strongest_baseline_comparison(
        {"label": production_label, "result": production_test},
        args.baseline_strongest_fidelity_summary,
    )
    production_control = production_test["control"]
    minimum_hit = (
        args.baseline_unfiltered_hit_at_10
        + args.minimum_hit_at_10_improvement
    )
    minimum_top1 = (
        args.baseline_target_nearest_top1_rate
        + args.minimum_target_nearest_top1_improvement
    )
    dp_section = None
    if dp_transitions is not None:
        if dp_transitions.get("diagnostic") != "strongest_aod_dp_rank_transitions":
            raise ValueError("--dp-transitions has an unsupported diagnostic type")
        dp_section = dp_transitions["cfg"][cfg_key]["pooled"]
    physical_mae = production_control.get("physical_condition_mae", {})
    physical_feasible_rate = production_control.get("physical_feasible_rate")
    physical_values_available = (
        physical_feasible_rate is not None
        and all(
            physical_mae.get(name) is not None
            for name in ("power_db", "delay_spread_ns", "k_factor_db")
        )
    )
    baseline_metrics = strongest_baseline.get("metrics", {})
    tx_baseline = baseline_metrics.get(
        "tx_spatial_correlation_relative_frobenius"
    )
    recall_baseline = baseline_metrics.get("multipath_recall")
    candidate_name = "strongest_anchor"

    gates = {
        f"{candidate_name}_hit_at_10_improvement": {
            "pass": float(production_control["hit_at_10deg"]) >= minimum_hit,
            "observed": float(production_control["hit_at_10deg"]),
            "minimum": minimum_hit,
            "baseline": args.baseline_unfiltered_hit_at_10,
            "rule": "Frozen Test Hit@10 must improve over the unfiltered Strongest baseline.",
        },
        "target_nearest_path_becomes_top1": {
            "pass": float(
                production_control.get("target_nearest_path_top1_rate", 0.0)
            ) >= minimum_top1,
            "observed": production_control.get("target_nearest_path_top1_rate"),
            "minimum": minimum_top1,
            "baseline": args.baseline_target_nearest_top1_rate,
            "rule": "The target-nearest path Top1 rate must improve over baseline.",
        },
        "candidate_16_to_8": {
            "pass": stability_rate >= args.minimum_16_to_8_success_rate,
            "observed_success_rate": stability_rate,
            "minimum_success_rate": args.minimum_16_to_8_success_rate,
        },
        "multipath_recall": {
            "pass": recall_gate,
            "classification": recall_decision["classification"],
            "rule": "Reject stable evidence of within-condition mode loss.",
        },
        "tx_spatial_nonregression_vs_strongest_baseline": {
            "pass": (
                float(tx_baseline["current_minus_baseline"])
                <= args.maximum_tx_spatial_regression
                if tx_baseline is not None
                else None
            ),
            "comparison": tx_baseline,
            "maximum_allowed_regression": args.maximum_tx_spatial_regression,
        },
        "multipath_recall_nonregression_vs_strongest_baseline": {
            "pass": (
                float(recall_baseline["current_minus_baseline"])
                >= -args.maximum_recall_regression
                if recall_baseline is not None
                else None
            ),
            "comparison": recall_baseline,
            "maximum_allowed_regression": args.maximum_recall_regression,
        },
        "physical_condition_accuracy": {
            "pass": (
                (
                    float(physical_feasible_rate) >= 0.99
                    and float(physical_mae["power_db"])
                    <= args.maximum_power_mae_db
                    and float(physical_mae["delay_spread_ns"])
                    <= args.maximum_delay_spread_mae_ns
                    and float(physical_mae["k_factor_db"])
                    <= args.maximum_k_factor_mae_db
                )
                if physical_values_available
                else None
            ),
            "observed_mae": physical_mae,
            "physical_feasible_rate": physical_feasible_rate,
            "maximum_mae": {
                "power_db": args.maximum_power_mae_db,
                "delay_spread_ns": args.maximum_delay_spread_mae_ns,
                "k_factor_db": args.maximum_k_factor_mae_db,
            },
        },
        "tx_spatial_vs_mean": {
            "pass": spatial_comparison["gate"],
            "rule": "Strongest must not be worse than frozen Mean on Test Tx spatial correlation.",
        },
        "dp_does_not_break_hit_at_10": {
            "pass": (
                float(dp_section["transition"]["hit_at_10_net_change"])
                >= -args.maximum_dp_ranking_regression
                if dp_section is not None
                else None
            ),
            "observed_after_minus_before": (
                dp_section["transition"]["hit_at_10_net_change"]
                if dp_section is not None
                else None
            ),
            "maximum_allowed_regression": args.maximum_dp_ranking_regression,
        },
        "dp_does_not_break_target_nearest_top1": {
            "pass": (
                float(
                    dp_section["transition"][
                        "target_nearest_top1_net_change"
                    ]
                )
                >= -args.maximum_dp_ranking_regression
                if dp_section is not None
                else None
            ),
            "observed_after_minus_before": (
                dp_section["transition"]["target_nearest_top1_net_change"]
                if dp_section is not None
                else None
            ),
            "maximum_allowed_regression": args.maximum_dp_ranking_regression,
        },
    }
    anchor_pre_repair_hit = production_control.get(
        "anchor_pre_repair_hit_at_10deg"
    )
    if anchor_pre_repair_hit is not None:
        gates["anchor_learned_before_structural_repair"] = {
            "pass": float(anchor_pre_repair_hit)
            >= args.minimum_anchor_pre_repair_hit_at_10,
            "observed": float(anchor_pre_repair_hit),
            "minimum": args.minimum_anchor_pre_repair_hit_at_10,
            "mean_pre_repair_error_deg": production_control.get(
                "anchor_pre_repair_error_deg_mean"
            ),
            "p95_pre_repair_error_deg": production_control.get(
                "anchor_pre_repair_error_deg_p95"
            ),
            "rule": (
                "The learned final-Top1 relative angle must already be near zero "
                "before the structural Anchor repair; otherwise the model is "
                "behaving like a post-generation angle edit."
            ),
        }
    anchor_target_nearest_diagnostic = None
    if candidate_name == "strongest_anchor":
        anchor_target_nearest_diagnostic = gates.pop(
            "dp_does_not_break_target_nearest_top1", None
        )
    known_gates = [section["pass"] for section in gates.values() if section["pass"] is not None]
    pending = [name for name, section in gates.items() if section["pass"] is None]
    eligible = all(known_gates) and not pending
    result = {
        "format_version": 1,
        "evaluation": "strongest_aod_full_protocol",
        "production_cfg": args.production_cfg,
        "cfg_validation": cfg_result,
        "candidate_16_to_8_validation": stability_result,
        "frozen_validation_selection": fidelity["validation_selection"],
        "selected_test_label": selected_label,
        "selected_test_control": selected_test["control"],
        "production_test_label": production_label,
        "production_test_control": production_control,
        "selected_test_spatial_fidelity": {
            metric: selected_test["fidelity"][metric] for metric in SPATIAL_METRICS
        },
        "production_test_spatial_fidelity": {
            metric: production_test["fidelity"][metric]
            for metric in SPATIAL_METRICS
        },
        "multipath_recall_decision": recall_decision,
        "dp_rank_transitions_at_production_cfg": dp_section,
        "strongest_vs_mean": spatial_comparison,
        "candidate_name": candidate_name,
        "candidate_vs_original_strongest": strongest_baseline,
        "gates": gates,
        "anchor_diagnostics": {
            "dp_target_nearest_top1": anchor_target_nearest_diagnostic,
            "interpretation": (
                "Diagnostic only for Anchor: coincident target-near paths make "
                "nearest-path identity ambiguous. Promotion uses final-power Top1 "
                "AoD control and requires the paired Copy/Swap/No-DP ablation."
                if candidate_name == "strongest_anchor"
                else None
            ),
            "required_followup": (
                "strongest_anchor_rebinding_ablation"
                if candidate_name == "strongest_anchor"
                else None
            ),
        },
        "pending_gates": pending,
        "decision": (
            "eligible_pending_rebinding_ablation"
            if eligible and candidate_name == "strongest_anchor"
            else "eligible_to_replace_mean" if eligible else
            "rejected" if not all(known_gates) else
            "pending_required_comparisons"
        ),
        "sources": {
            "cfg": str(Path(args.cfg).resolve()),
            "candidate_stability": str(Path(args.candidate_stability).resolve()),
            "fidelity": str(Path(args.fidelity).resolve()),
            "recall": str(Path(args.recall).resolve()),
            "dp_transitions": (
                str(Path(args.dp_transitions).resolve())
                if args.dp_transitions
                else None
            ),
            "mean_fidelity_summary": (
                str(Path(args.mean_fidelity_summary).resolve())
                if args.mean_fidelity_summary else None
            ),
            "baseline_strongest_fidelity_summary": (
                str(Path(args.baseline_strongest_fidelity_summary).resolve())
                if args.baseline_strongest_fidelity_summary
                else None
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
