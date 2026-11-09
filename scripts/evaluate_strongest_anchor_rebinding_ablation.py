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


VARIANTS = ("no_dp", "dp_copy", "dp_swap")
PRIMARY_METRICS = {
    "tx_spatial_correlation_relative_frobenius": "lower",
    "path_level_power_aod_sliced_wasserstein": "lower",
    "link_level_power_aod_sliced_wasserstein": "lower",
    "power_weighted_aod_sliced_wasserstein": "lower",
    "multipath_recall": "higher",
    "multipath_coverage": "higher",
    "multipath_rare_reference_coverage": "higher",
}


def _to_device(
    payload: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    samples = int(payload["path_mask"].shape[0])
    return {
        key: value.to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (samples,)
    }


def _select(
    payload: Mapping[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in payload.items()}


def _bootstrap_delta(
    values: list[float], *, samples: int, seed: int, direction: str
) -> dict[str, Any]:
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    selection = torch.randint(
        tensor.numel(), (samples, tensor.numel()), generator=generator
    )
    distribution = tensor[selection].mean(-1)
    interval = torch.quantile(
        distribution, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    lower, upper = float(interval[0]), float(interval[1])
    if direction == "lower":
        evidence = (
            "improvement" if upper < 0.0 else "regression" if lower > 0.0 else "inconclusive"
        )
    else:
        evidence = (
            "improvement" if lower > 0.0 else "regression" if upper < 0.0 else "inconclusive"
        )
    return {
        "groups": int(tensor.numel()),
        "candidate_minus_baseline": float(tensor.mean()),
        "bootstrap_95_percent_ci": [lower, upper],
        "direction": f"{direction}_is_better",
        "evidence": evidence,
    }


def _anchor_metrics(
    payload: Mapping[str, torch.Tensor], target: torch.Tensor
) -> dict[str, Any]:
    measured = strongest_aod(payload)
    error = circular_distance_degrees(measured.angle_deg, target)
    tracked = payload.get("strongest_anchor_path_index")
    tracked_correct = (
        tracked.long() == measured.path_index
        if isinstance(tracked, torch.Tensor) and tracked.shape == measured.path_index.shape
        else torch.zeros_like(measured.valid)
    )
    added = payload.get("strongest_anchor_added_target_match_count")
    changed = payload.get("strongest_anchor_identity_changed")
    involves_direct = payload.get("strongest_anchor_rebind_involves_direct_path")
    return {
        "samples": int(error.numel()),
        "final_top1_error_deg_mean": float(error.mean()),
        "final_top1_error_deg_p95": float(torch.quantile(error, 0.95)),
        "final_top1_hit_at_5deg": float((measured.valid & (error <= 5.0)).float().mean()),
        "final_top1_hit_at_10deg": float((measured.valid & (error <= 10.0)).float().mean()),
        "tracked_anchor_is_final_top1_rate": float(tracked_correct.float().mean()),
        "top1_identity_changed_by_dp_rate": (
            float(changed.float().mean()) if isinstance(changed, torch.Tensor) else 0.0
        ),
        "rebind_added_target_match_rate": (
            float((added > 0).float().mean()) if isinstance(added, torch.Tensor) else 0.0
        ),
        "rebind_added_target_match_count_mean": (
            float(added.float().mean()) if isinstance(added, torch.Tensor) else 0.0
        ),
        "rebind_involves_direct_path_rate": (
            float(involves_direct.float().mean())
            if isinstance(involves_direct, torch.Tensor)
            else 0.0
        ),
    }


def _direct_angles(
    payload: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    direct = payload["direct_path_mask"].bool() & payload["path_mask"].bool()
    valid = direct.any(-1)
    index = direct.float().argmax(-1)
    angle = payload["aod_az_deg"].gather(1, index.unsqueeze(-1)).squeeze(-1)
    return angle, valid


def _geometry_metrics(
    candidate: Mapping[str, torch.Tensor],
    no_dp: Mapping[str, torch.Tensor],
    real_query: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, Any]:
    candidate_angle, candidate_valid = _direct_angles(candidate)
    no_dp_angle, no_dp_valid = _direct_angles(no_dp)
    real_angle, real_valid = _direct_angles(real_query)
    los = no_dp["conditions"][:, 3] >= 0.5
    preservation_valid = los & candidate_valid & no_dp_valid
    proxy_valid = los & candidate_valid & real_valid
    preservation_valid = preservation_valid[indices]
    proxy_valid = proxy_valid[indices]
    preservation = circular_distance_degrees(
        candidate_angle[indices], no_dp_angle[indices]
    )[preservation_valid]
    proxy = circular_distance_degrees(
        candidate_angle[indices], real_angle[indices]
    )[proxy_valid]
    return {
        "los_samples": int(los[indices].sum()),
        "direct_path_preservation_samples": int(preservation.numel()),
        "direct_aod_change_vs_no_dp_mean_deg": (
            float(preservation.mean()) if preservation.numel() else None
        ),
        "direct_aod_change_vs_no_dp_p95_deg": (
            float(torch.quantile(preservation, 0.95)) if preservation.numel() else None
        ),
        "direct_aod_changed_over_1deg_rate": (
            float((preservation > 1.0).float().mean()) if preservation.numel() else None
        ),
        "paired_real_direct_aod_error_mean_deg_diagnostic_only": (
            float(proxy.mean()) if proxy.numel() else None
        ),
        "note": (
            "The paired-real direct AoD is diagnostic only because link geometry/position "
            "is not an explicit model condition. Change versus No-DP measures damage "
            "introduced by DP/rebinding itself."
        ),
    }


def _macro(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        variant: {
            metric: sum(record["models"][variant][metric] for record in records)
            / len(records)
            for metric in PRIMARY_METRICS
        }
        for variant in VARIANTS
    }


def _comparisons(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pairs = (
        ("dp_copy", "no_dp"),
        ("dp_swap", "no_dp"),
        ("dp_swap", "dp_copy"),
    )
    for pair_number, (candidate, baseline) in enumerate(pairs):
        section: dict[str, Any] = {}
        for metric_number, (metric, direction) in enumerate(PRIMARY_METRICS.items()):
            delta = [
                record["models"][candidate][metric]
                - record["models"][baseline][metric]
                for record in records
            ]
            section[metric] = _bootstrap_delta(
                delta,
                samples=bootstrap_samples,
                seed=seed + pair_number * 100 + metric_number,
                direction=direction,
            )
        result[f"{candidate}_vs_{baseline}"] = section
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired Strongest Anchor No-DP, DP+Copy and DP+AoD-Swap "
            "variants, including DP-changed and LoS geometry slices."
        )
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum-samples-per-map", type=int, default=8)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--projection-seeds", default="0,1,2")
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--weighted-quantiles", type=int, default=256)
    parser.add_argument("--coverage-k", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-final-hit-at-10", type=float, default=0.99)
    parser.add_argument("--minimum-tracked-anchor-correct-rate", type=float, default=0.999)
    parser.add_argument("--maximum-swap-added-target-match-rate", type=float, default=0.0)
    parser.add_argument("--maximum-los-direct-change-over-1deg-rate", type=float, default=0.01)
    args = parser.parse_args()
    if args.minimum_samples_per_map < 2 or args.bootstrap_samples < 100:
        raise ValueError("minimum samples must be >=2 and bootstrap samples >=100")

    device = resolve_device(args.device)
    source = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if source.get("model_kind") != "strongest_anchor_rebinding_ablation":
        raise ValueError("artifact is not a Strongest Anchor rebinding ablation")
    payloads = {
        label: _to_device(source["payloads"][label], device)
        for label in ("reference", "real_query", *VARIANTS)
    }
    target = source["target_aod_deg"].to(device).float()
    samples = int(target.numel())
    if any(payloads[label]["path_mask"].shape[0] != samples for label in payloads):
        raise ValueError("ablation payloads are not sample-aligned")
    projection_seeds = tuple(
        int(value) for value in args.projection_seeds.split(",") if value
    )
    if not projection_seeds:
        raise ValueError("--projection-seeds requires at least one integer")
    evaluator = Evaluator(
        {"reference": payloads["reference"], **{name: payloads[name] for name in VARIANTS}},
        projection_dimension=args.projection_dimension,
        projection_seeds=projection_seeds,
        projections=args.swd_projections,
        quantiles=args.weighted_quantiles,
        coverage_k=args.coverage_k,
    )
    map_ids = payloads["no_dp"]["map_id"].long()
    los = payloads["no_dp"]["conditions"][:, 3] >= 0.5
    changed = source["dp_top1_changed"].to(device).bool()
    records: list[dict[str, Any]] = []
    los_records: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for map_tensor in torch.unique(map_ids, sorted=True):
        map_id = int(map_tensor)
        indices = torch.nonzero(map_ids == map_tensor, as_tuple=False).flatten()
        if indices.numel() < args.minimum_samples_per_map:
            continue
        record = {
            "map_id": map_id,
            "samples": int(indices.numel()),
            "models": {
                variant: evaluator.evaluate(variant, indices, indices)
                for variant in VARIANTS
            },
        }
        records.append(record)
        for los_value in (0, 1):
            local = indices[los[indices] == bool(los_value)]
            if local.numel() < max(4, args.minimum_samples_per_map // 2):
                continue
            los_records[los_value].append(
                {
                    "map_id": map_id,
                    "samples": int(local.numel()),
                    "models": {
                        variant: evaluator.evaluate(
                            variant, local, local, local=True
                        )
                        for variant in VARIANTS
                    },
                }
            )
        print(json.dumps({"event": "ablation_map_complete", "map_id": map_id}), flush=True)
    if not records:
        raise RuntimeError("no Map has enough samples for the ablation")

    all_indices = torch.arange(samples, device=device)
    changed_indices = torch.nonzero(changed, as_tuple=False).flatten()
    changed_slices: dict[str, Any] = {
        "samples": int(changed_indices.numel()),
        "rate": float(changed.float().mean()),
        "by_los": {},
    }
    if changed_indices.numel() >= 2:
        changed_slices["fidelity"] = {
            variant: evaluator.evaluate(variant, changed_indices, changed_indices)
            for variant in VARIANTS
        }
        changed_slices["anchor"] = {
            variant: _anchor_metrics(
                _select(payloads[variant], changed_indices), target[changed_indices]
            )
            for variant in VARIANTS
        }
        changed_slices["los_geometry"] = {
            variant: _geometry_metrics(
                payloads[variant], payloads["no_dp"], payloads["real_query"], changed_indices
            )
            for variant in VARIANTS
        }
    for los_value in (0, 1):
        local = changed_indices[los[changed_indices] == bool(los_value)]
        changed_slices["by_los"][str(los_value)] = {
            "samples": int(local.numel()),
            "anchor": (
                {
                    variant: _anchor_metrics(
                        _select(payloads[variant], local), target[local]
                    )
                    for variant in VARIANTS
                }
                if local.numel()
                else {}
            ),
        }

    anchor = {
        variant: _anchor_metrics(payloads[variant], target) for variant in VARIANTS
    }
    geometry = {
        variant: _geometry_metrics(
            payloads[variant], payloads["no_dp"], payloads["real_query"], all_indices
        )
        for variant in VARIANTS
    }
    macro = _macro(records)
    comparisons = _comparisons(
        records, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    by_los = {
        str(value): {
            "maps_or_groups": len(los_records[value]),
            "macro": _macro(los_records[value]) if los_records[value] else {},
            "comparisons": (
                _comparisons(
                    los_records[value],
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + 1000 + value * 100,
                )
                if los_records[value]
                else {}
            ),
        }
        for value in (0, 1)
    }
    swap_geometry_rate = geometry["dp_swap"]["direct_aod_changed_over_1deg_rate"]
    gates = {
        "swap_final_top1_control": {
            "pass": anchor["dp_swap"]["final_top1_hit_at_10deg"]
            >= args.minimum_final_hit_at_10,
            "observed": anchor["dp_swap"]["final_top1_hit_at_10deg"],
            "minimum": args.minimum_final_hit_at_10,
        },
        "swap_tracked_anchor_is_final_top1": {
            "pass": anchor["dp_swap"]["tracked_anchor_is_final_top1_rate"]
            >= args.minimum_tracked_anchor_correct_rate,
            "observed": anchor["dp_swap"]["tracked_anchor_is_final_top1_rate"],
            "minimum": args.minimum_tracked_anchor_correct_rate,
        },
        "swap_does_not_add_target_duplicates": {
            "pass": anchor["dp_swap"]["rebind_added_target_match_rate"]
            <= args.maximum_swap_added_target_match_rate,
            "observed": anchor["dp_swap"]["rebind_added_target_match_rate"],
            "maximum": args.maximum_swap_added_target_match_rate,
            "note": "Natural pre-existing coincident directions are allowed.",
        },
        "swap_los_direct_geometry_preservation": {
            "pass": swap_geometry_rate is not None
            and swap_geometry_rate <= args.maximum_los_direct_change_over_1deg_rate,
            "observed_rate_over_1deg": swap_geometry_rate,
            "maximum": args.maximum_los_direct_change_over_1deg_rate,
        },
    }
    swap_vs_no_dp = comparisons["dp_swap_vs_no_dp"]
    stable_dp_benefit = any(
        swap_vs_no_dp[name]["evidence"] == "improvement"
        for name in (
            "tx_spatial_correlation_relative_frobenius",
            "power_weighted_aod_sliced_wasserstein",
            "multipath_recall",
            "multipath_coverage",
        )
    ) and not any(
        swap_vs_no_dp[name]["evidence"] == "regression"
        for name in (
            "tx_spatial_correlation_relative_frobenius",
            "multipath_recall",
            "multipath_coverage",
        )
    )
    gates_pass = all(section["pass"] for section in gates.values())
    report = {
        "format_version": 1,
        "evaluation": "strongest_anchor_rebinding_ablation",
        "cfg": source["cfg"],
        "samples": samples,
        "maps": len(records),
        "protocol": source["protocol"] | {
            "macro_unit": "Map",
            "confidence_interval": "paired Map bootstrap",
            "primary_anchor_contract": (
                "circular AoD error of final-power Top1 versus target; uniqueness "
                "of target-near paths is not required"
            ),
        },
        "anchor": anchor,
        "los_geometry": geometry,
        "macro_per_map": macro,
        "paired_per_map": comparisons,
        "macro_per_map_by_los": by_los,
        "dp_top1_changed_subset": changed_slices,
        "per_map": records,
        "gates": gates,
        "dp_swap_has_stable_benefit_vs_no_dp": stable_dp_benefit,
        "decision": (
            "prefer_dp_swap"
            if gates_pass and stable_dp_benefit
            else "prefer_no_dp"
            if gates_pass
            else "reject_dp_swap"
        ),
        "sources": source["sources"] | {"artifact": str(Path(args.artifact).resolve())},
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "strongest_anchor_rebinding_ablation_evaluation_complete",
                "output": str(Path(args.output).resolve()),
                "decision": report["decision"],
                "dp_top1_changed_subset_samples": int(changed_indices.numel()),
                "gates": gates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
