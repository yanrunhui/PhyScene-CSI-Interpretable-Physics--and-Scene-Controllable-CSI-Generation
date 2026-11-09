#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.evaluation.aod_support import (
    circular_distance_degrees,
    strongest_aod,
)
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump
from scripts.evaluate_csi_absolute_quality import Evaluator


VARIANTS = (
    "direct_8",
    "random_16_to_8",
    "diversity_16_to_8",
    "random_24_to_8",
    "diversity_24_to_8",
)
METRICS = {
    "multipath_precision": "higher",
    "multipath_density": "higher",
    "multipath_recall": "higher",
    "multipath_coverage": "higher",
    "multipath_rare_reference_coverage": "higher",
    "tx_spatial_correlation_relative_frobenius": "lower",
    "path_level_power_aod_sliced_wasserstein": "lower",
    "link_level_power_aod_sliced_wasserstein": "lower",
    "power_weighted_aod_sliced_wasserstein": "lower",
}
COMPARISONS = (
    ("random_16_to_8", "direct_8"),
    ("diversity_16_to_8", "direct_8"),
    ("diversity_16_to_8", "random_16_to_8"),
    ("random_24_to_8", "direct_8"),
    ("diversity_24_to_8", "direct_8"),
    ("diversity_24_to_8", "random_24_to_8"),
)


def _to_device(
    payload: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    samples = int(payload["path_mask"].shape[0])
    return {
        key: value.to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (samples,)
    }


def _bootstrap_delta(
    values: list[float], *, samples: int, seed: int, direction: str
) -> dict[str, Any]:
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        tensor.numel(), (samples, tensor.numel()), generator=generator
    )
    distribution = tensor[indices].mean(-1)
    interval = torch.quantile(
        distribution, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    lower, upper = float(interval[0]), float(interval[1])
    if direction == "higher":
        evidence = (
            "improvement" if lower > 0.0 else
            "regression" if upper < 0.0 else
            "inconclusive"
        )
    else:
        evidence = (
            "improvement" if upper < 0.0 else
            "regression" if lower > 0.0 else
            "inconclusive"
        )
    return {
        "groups": int(tensor.numel()),
        "candidate_minus_baseline": float(tensor.mean()),
        "bootstrap_95_percent_ci": [lower, upper],
        "direction": f"{direction}_is_better",
        "evidence": evidence,
    }


def _macro(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    labels = ("real_real", *VARIANTS)
    return {
        label: {
            metric: sum(record["models"][label][metric] for record in records)
            / len(records)
            for metric in METRICS
        }
        for label in labels
    }


def _comparisons(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pair_number, (candidate, baseline) in enumerate(COMPARISONS):
        result[f"{candidate}_vs_{baseline}"] = {
            metric: _bootstrap_delta(
                [
                    record["models"][candidate][metric]
                    - record["models"][baseline][metric]
                    for record in records
                ],
                samples=bootstrap_samples,
                seed=seed + pair_number * 100 + metric_number,
                direction=direction,
            )
            for metric_number, (metric, direction) in enumerate(METRICS.items())
        }
    return result


def _control_metrics(
    payload: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
) -> dict[str, Any]:
    query_index = payload["query_index"].long()
    measured = strongest_aod(payload)
    error = circular_distance_degrees(measured.angle_deg, targets[query_index])
    feasible = payload.get("constraint_feasible")
    valid = measured.valid
    if isinstance(feasible, torch.Tensor):
        valid = valid & feasible.bool()
    return {
        "samples": int(error.numel()),
        "final_top1_error_deg_mean": float(error.mean()),
        "final_top1_error_deg_p95": float(torch.quantile(error, 0.95)),
        "final_top1_hit_at_10deg": float((valid & (error <= 10.0)).float().mean()),
        "constraint_feasible_rate": (
            float(feasible.float().mean())
            if isinstance(feasible, torch.Tensor)
            else None
        ),
    }


def _evaluate_groups(
    evaluator: Evaluator,
    payloads: Mapping[str, Mapping[str, torch.Tensor]],
    query_selection: torch.Tensor,
    *,
    minimum_queries: int,
    group_name: str,
    local: bool,
) -> list[dict[str, Any]]:
    query_map_ids = payloads["real_query"]["map_id"].long()
    records: list[dict[str, Any]] = []
    for map_tensor in torch.unique(query_map_ids[query_selection], sorted=True):
        reference_indices = torch.nonzero(
            query_selection & (query_map_ids == map_tensor), as_tuple=False
        ).flatten()
        if reference_indices.numel() < minimum_queries:
            continue
        models: dict[str, dict[str, float]] = {
            "real_real": evaluator.evaluate(
                "real_query", reference_indices, reference_indices, local=local
            )
        }
        for variant in VARIANTS:
            query_index = payloads[variant]["query_index"].long()
            selected_query = torch.zeros_like(query_selection)
            selected_query[reference_indices] = True
            candidate_indices = torch.nonzero(
                selected_query[query_index], as_tuple=False
            ).flatten()
            models[variant] = evaluator.evaluate(
                variant, candidate_indices, reference_indices, local=local
            )
        records.append(
            {
                "group": group_name,
                "map_id": int(map_tensor),
                "queries": int(reference_indices.numel()),
                "candidate_samples_per_variant": int(reference_indices.numel()) * 8,
                "models": models,
            }
        )
    return records


def _slice_summary(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    if not records:
        return {"groups": 0, "status": "insufficient_samples"}
    return {
        "groups": len(records),
        "queries": sum(record["queries"] for record in records),
        "macro_per_map": _macro(records),
        "paired_per_map": _comparisons(
            records, bootstrap_samples=bootstrap_samples, seed=seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct 8, random 16/24-to-8 and mild-diversity 16/24-to-8 "
            "from one locked Strongest Anchor candidate pool."
        )
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum-queries-per-map", type=int, default=4)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--projection-seeds", default="0,1,2")
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--weighted-quantiles", type=int, default=256)
    parser.add_argument("--coverage-k", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--minimum-hit-at-10", type=float, default=0.99)
    parser.add_argument("--minimum-selection-success-rate", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.minimum_queries_per_map < 4 or args.bootstrap_samples < 100:
        raise ValueError("minimum queries must be >=4 and bootstrap samples >=100")

    device = resolve_device(args.device)
    source = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if source.get("model_kind") != "strongest_anchor_diversity_selection":
        raise ValueError("artifact is not a Strongest Anchor diversity selection artifact")
    if int(source.get("cfg", -1)) != 1:
        raise ValueError("the locked diversity evaluation requires CFG1")
    payloads = {
        label: _to_device(source["payloads"][label], device)
        for label in ("reference", "real_query", *VARIANTS)
    }
    query_count = int(source["target_aod_deg"].numel())
    if any(payloads[label]["path_mask"].shape[0] != query_count for label in ("reference", "real_query")):
        raise ValueError("real query and reference payloads are not query-aligned")
    for variant in VARIANTS:
        expected = query_count * int(source["protocol"]["outputs_per_query"])
        if payloads[variant]["path_mask"].shape[0] != expected:
            raise ValueError(f"{variant} does not contain exactly 8 outputs per query")
    projection_seeds = tuple(
        int(value) for value in args.projection_seeds.split(",") if value
    )
    if not projection_seeds:
        raise ValueError("--projection-seeds requires at least one integer")
    evaluator = Evaluator(
        payloads,
        projection_dimension=args.projection_dimension,
        projection_seeds=projection_seeds,
        projections=args.swd_projections,
        quantiles=args.weighted_quantiles,
        coverage_k=args.coverage_k,
    )

    real_query = payloads["real_query"]
    all_queries = torch.ones(query_count, dtype=torch.bool, device=device)
    los = real_query["conditions"][:, 3] >= 0.5
    slices = {
        "all": all_queries,
        "nlos": ~los,
        "los": los,
    }
    slice_reports: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for slice_number, (name, selection) in enumerate(slices.items()):
        records = _evaluate_groups(
            evaluator,
            payloads,
            selection,
            minimum_queries=args.minimum_queries_per_map,
            group_name=name,
            local=name != "all",
        )
        slice_reports[name] = _slice_summary(
            records,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + slice_number * 1000,
        ) | {"selected_queries": int(selection.sum())}
        if name == "all":
            all_records = records
    if not all_records:
        raise RuntimeError("no Map has enough query samples for diversity evaluation")

    targets = source["target_aod_deg"].to(device).float()
    control = {
        variant: _control_metrics(payloads[variant], targets)
        for variant in VARIANTS
    }
    paired = slice_reports["all"]["paired_per_map"]
    guardrails = (
        "multipath_precision",
        "multipath_density",
        "multipath_recall",
        "tx_spatial_correlation_relative_frobenius",
        "path_level_power_aod_sliced_wasserstein",
        "link_level_power_aod_sliced_wasserstein",
        "power_weighted_aod_sliced_wasserstein",
    )
    decisions: dict[str, Any] = {}
    for budget in (16, 24):
        candidate = f"diversity_{budget}_to_8"
        baseline = f"random_{budget}_to_8"
        comparison = paired[f"{candidate}_vs_{baseline}"]
        coverage_evidence = comparison["multipath_coverage"]["evidence"]
        regressions = [
            metric for metric in guardrails
            if comparison[metric]["evidence"] == "regression"
        ]
        hit_pass = control[candidate]["final_top1_hit_at_10deg"] >= args.minimum_hit_at_10
        selection_success_rate = float(
            source["selection_audit"][candidate]["query_success_rate"]
        )
        selection_success_pass = (
            selection_success_rate >= args.minimum_selection_success_rate
        )
        decisions[str(budget)] = {
            "promote": (
                coverage_evidence == "improvement"
                and not regressions
                and hit_pass
                and selection_success_pass
            ),
            "coverage_evidence_vs_same_budget_random": coverage_evidence,
            "stable_guardrail_regressions": regressions,
            "final_top1_control_pass": hit_pass,
            "selection_success_rate": selection_success_rate,
            "selection_success_pass": selection_success_pass,
            "minimum_selection_success_rate": args.minimum_selection_success_rate,
        }
    promoted = next(
        (budget for budget in (16, 24) if decisions[str(budget)]["promote"]),
        None,
    )
    decision = (
        f"promote_mild_diversity_{promoted}_to_8"
        if promoted is not None
        else "keep_direct_random_8"
    )
    report = {
        "format_version": 1,
        "evaluation": "strongest_anchor_candidate_diversity",
        "cfg": 1,
        "queries": query_count,
        "maps": len(all_records),
        "protocol": source["protocol"] | {
            "macro_unit": "Map",
            "confidence_interval": "paired Map bootstrap",
            "causal_control": (
                "each diversity selector is compared with random selection from the "
                "same candidate-budget prefix of the same generated pool"
            ),
            "promotion_rule": (
                "stable Coverage improvement versus same-budget random, no stable "
                "regression in Precision/Density/Recall/Tx/Power-AoD, and Hit@10 gate"
            ),
        },
        "selection_audit": source["selection_audit"],
        "control": control,
        "slices": slice_reports,
        "budget_decisions": decisions,
        "decision": decision,
        "sources": source["sources"] | {"artifact": str(Path(args.artifact).resolve())},
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "strongest_anchor_diversity_evaluation_complete",
                "output": str(Path(args.output).resolve()),
                "decision": decision,
                "budget_decisions": decisions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
