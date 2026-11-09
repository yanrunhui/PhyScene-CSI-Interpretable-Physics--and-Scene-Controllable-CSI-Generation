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

from physics_csi.evaluation.aod_support import circular_distance_degrees, strongest_aod
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump
from scripts.evaluate_csi_absolute_quality import Evaluator


METRICS = {
    "multipath_precision": "higher",
    "multipath_density": "higher",
    "multipath_recall": "higher",
    "multipath_coverage": "higher",
    "multipath_rare_reference_coverage": "higher",
    "path_level_power_aod_sliced_wasserstein": "lower",
    "link_level_power_aod_sliced_wasserstein": "lower",
    "power_weighted_aod_sliced_wasserstein": "lower",
    "tx_spatial_correlation_relative_frobenius": "lower",
}


def _to_device(payload: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    samples = int(payload["path_mask"].shape[0])
    return {
        key: value.to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (samples,)
    }


def _relative_aod_spread(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    measured = strongest_aod(payload)
    mask = payload["path_mask"].bool()
    power = payload["path_powers_linear"].float().clamp_min(0.0) * mask
    weight = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
    distance = circular_distance_degrees(
        payload["aod_az_deg"].float(), measured.angle_deg.unsqueeze(-1)
    )
    return torch.sqrt((weight * distance.square()).sum(-1))


def _relative_structure(
    payload: Mapping[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, float]:
    selected = {
        key: value[indices]
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
        and value.shape[:1] == (payload["path_mask"].shape[0],)
    }
    measured = strongest_aod(selected)
    mask = selected["path_mask"].bool()
    rows = torch.arange(mask.shape[0], device=mask.device)
    secondary = mask.clone()
    secondary[rows, measured.path_index] = False
    power = selected["path_powers_linear"].float().clamp_min(0.0) * mask
    weight = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
    relative_aod = circular_distance_degrees(
        selected["aod_az_deg"].float(), measured.angle_deg.unsqueeze(-1)
    )
    spread = torch.sqrt((weight * relative_aod.square()).sum(-1))
    far = secondary & (relative_aod > 75.0)
    far_power = (weight * far).sum(-1)
    top_delay = selected["path_delays_ns"].gather(
        1, measured.path_index.unsqueeze(-1)
    ).squeeze(-1)
    delay_offset = (
        selected["path_delays_ns"].float() - top_delay.unsqueeze(-1)
    ).abs()
    weighted_delay_offset = (weight * delay_offset).sum(-1)
    return {
        "relative_aod_spread_mean_deg": float(spread.mean()),
        "relative_aod_spread_p90_deg": float(torch.quantile(spread, 0.90)),
        "wide_secondary_path_rate": float(far.any(-1).float().mean()),
        "power_fraction_beyond_75deg_mean": float(far_power.mean()),
        "power_weighted_delay_offset_from_top1_ns_mean": float(
            weighted_delay_offset.mean()
        ),
        "top1_top2_power_gap_db_mean": float(measured.top2_power_gap_db.mean()),
    }


def _bootstrap(values: list[float], samples: int, seed: int) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        tensor.numel(), (samples, tensor.numel()), generator=generator
    )
    distribution = tensor[indices].mean(-1)
    interval = torch.quantile(
        distribution, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return [float(interval[0]), float(interval[1])]


def _summarize(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    if not records:
        return {"groups": 0, "status": "insufficient_samples"}
    metrics: dict[str, Any] = {}
    for number, (metric, direction) in enumerate(METRICS.items()):
        candidate = [record["candidate"][metric] for record in records]
        floor = [record["real_real"][metric] for record in records]
        delta = [first - second for first, second in zip(candidate, floor)]
        interval = _bootstrap(delta, bootstrap_samples, seed + number)
        if direction == "higher":
            evidence = (
                "candidate_better" if interval[0] > 0.0 else
                "candidate_worse" if interval[1] < 0.0 else "inconclusive"
            )
        else:
            evidence = (
                "candidate_better" if interval[1] < 0.0 else
                "candidate_worse" if interval[0] > 0.0 else "inconclusive"
            )
        metrics[metric] = {
            "candidate_macro_mean": sum(candidate) / len(candidate),
            "real_real_floor_macro_mean": sum(floor) / len(floor),
            "candidate_minus_real_real": sum(delta) / len(delta),
            "bootstrap_95_percent_ci": interval,
            "direction": f"{direction}_is_better",
            "evidence": evidence,
            "candidate_worse_group_rate": sum(
                (value < 0.0 if direction == "higher" else value > 0.0)
                for value in delta
            ) / len(delta),
        }
    structure: dict[str, Any] = {}
    for number, metric in enumerate(records[0]["relative_structure"]["candidate"]):
        candidate = [
            record["relative_structure"]["candidate"][metric] for record in records
        ]
        real = [
            record["relative_structure"]["real_query"][metric] for record in records
        ]
        delta = [first - second for first, second in zip(candidate, real)]
        structure[metric] = {
            "candidate_macro_mean": sum(candidate) / len(candidate),
            "paired_real_macro_mean": sum(real) / len(real),
            "candidate_minus_paired_real": sum(delta) / len(delta),
            "bootstrap_95_percent_ci": _bootstrap(
                delta, bootstrap_samples, seed + 100 + number
            ),
            "interpretation": "descriptive; zero is paired-real agreement",
        }
    return {
        "groups": len(records),
        "samples": sum(record["samples"] for record in records),
        "metrics": metrics,
        "relative_anchor_structure": structure,
    }


def _evaluate_stratum(
    evaluator: Evaluator,
    map_ids: torch.Tensor,
    selection: torch.Tensor,
    *,
    minimum_samples: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for map_tensor in torch.unique(map_ids[selection], sorted=True):
        indices = torch.nonzero(
            selection & (map_ids == map_tensor), as_tuple=False
        ).flatten()
        if indices.numel() < minimum_samples:
            continue
        records.append(
            {
                "map_id": int(map_tensor),
                "samples": int(indices.numel()),
                "candidate": evaluator.evaluate("candidate", indices, indices, local=True),
                "real_real": evaluator.evaluate("real_query", indices, indices, local=True),
                "relative_structure": {
                    "candidate": _relative_structure(
                        evaluator.payloads["candidate"], indices
                    ),
                    "real_query": _relative_structure(
                        evaluator.payloads["real_query"], indices
                    ),
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Strongest Anchor Coverage against a paired Real-Real floor "
            "by Map, LoS, relative AoD spread and Top1/Top2 gap."
        )
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("dp_swap", "dp_copy", "no_dp"), default="dp_swap")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum-samples-per-group", type=int, default=8)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--projection-seeds", default="0,1,2")
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--weighted-quantiles", type=int, default=256)
    parser.add_argument("--coverage-k", type=int, default=3)
    parser.add_argument("--minimum-real-support", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.minimum_samples_per_group < 4 or args.bootstrap_samples < 100:
        raise ValueError("minimum group samples must be >=4 and bootstrap samples >=100")
    if args.minimum_real_support < 1:
        raise ValueError("minimum real support must be positive")

    device = resolve_device(args.device)
    source = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if source.get("model_kind") != "strongest_anchor_rebinding_ablation":
        raise ValueError("artifact is not a Strongest Anchor rebinding ablation")
    test_artifact_path = Path(source.get("sources", {}).get("test_artifact", ""))
    if not test_artifact_path.is_file():
        raise FileNotFoundError(
            "the linked Test artifact is required to verify class-A support: "
            f"{test_artifact_path}"
        )
    test_artifact = torch.load(
        test_artifact_path, map_location="cpu", weights_only=False
    )
    real_support = test_artifact.get("query_real_support_count")
    if not isinstance(real_support, torch.Tensor):
        raise ValueError("linked Test artifact has no query_real_support_count")
    payloads = {
        "reference": _to_device(source["payloads"]["reference"], device),
        "real_query": _to_device(source["payloads"]["real_query"], device),
        "candidate": _to_device(source["payloads"][args.variant], device),
    }
    samples = int(payloads["candidate"]["path_mask"].shape[0])
    if any(payload["path_mask"].shape[0] != samples for payload in payloads.values()):
        raise ValueError("coverage artifact payloads must be sample-aligned")
    if real_support.shape != (samples,):
        raise ValueError("query_real_support_count is not aligned with the ablation")
    if bool((real_support < args.minimum_real_support).any()):
        raise ValueError("coverage diagnosis is restricted to class-A supported requests")
    seeds = tuple(int(value) for value in args.projection_seeds.split(",") if value)
    if not seeds:
        raise ValueError("--projection-seeds requires at least one integer")
    evaluator = Evaluator(
        payloads,
        projection_dimension=args.projection_dimension,
        projection_seeds=seeds,
        projections=args.swd_projections,
        quantiles=args.weighted_quantiles,
        coverage_k=args.coverage_k,
    )
    real = payloads["real_query"]
    map_ids = real["map_id"].long()
    los = real["conditions"][:, 3] >= 0.5
    measurement = strongest_aod(real)
    spread = _relative_aod_spread(real)
    gap = measurement.top2_power_gap_db
    direct = real["direct_path_mask"].bool()
    strongest_is_direct = direct.gather(
        1, measurement.path_index.unsqueeze(-1)
    ).squeeze(-1)
    k_factor = real["conditions"][:, 2]
    k_valid = real["condition_validity"][:, 2].bool()
    all_samples = torch.ones(samples, dtype=torch.bool, device=device)
    strata = {
        "all": all_samples,
        "los_nlos/nlos": ~los,
        "los_nlos/los": los,
        "relative_aod_spread/compact_le_30deg": spread <= 30.0,
        "relative_aod_spread/medium_30_to_75deg": (spread > 30.0) & (spread <= 75.0),
        "relative_aod_spread/wide_gt_75deg": spread > 75.0,
        "top1_top2_gap/near_tie_le_0_5db": gap <= 0.5,
        "top1_top2_gap/moderate_0_5_to_3db": (gap > 0.5) & (gap < 3.0),
        "top1_top2_gap/dominant_ge_3db": gap >= 3.0,
        "los_identity/direct_is_strongest": los & strongest_is_direct,
        "los_identity/scatter_is_strongest": los & ~strongest_is_direct,
        "los_k_factor/k_lt_0db": los & k_valid & (k_factor < 0.0),
        "los_k_factor/k_0_to_6db": los & k_valid & (k_factor >= 0.0) & (k_factor < 6.0),
        "los_k_factor/k_ge_6db": los & k_valid & (k_factor >= 6.0),
    }
    results: dict[str, Any] = {}
    per_map: list[dict[str, Any]] = []
    for number, (name, selection) in enumerate(strata.items()):
        records = _evaluate_stratum(
            evaluator,
            map_ids,
            selection,
            minimum_samples=args.minimum_samples_per_group,
        )
        results[name] = _summarize(
            records,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + number * 100,
        ) | {"selected_samples": int(selection.sum())}
        if name == "all":
            per_map = records
    coverage = results["all"]["metrics"]["multipath_coverage"]
    diagnosis = (
        "broad_stable_coverage_contraction"
        if coverage["evidence"] == "candidate_worse"
        and coverage["candidate_worse_group_rate"] >= 0.75
        else "localized_or_inconclusive_coverage_loss"
    )
    report = {
        "format_version": 1,
        "diagnostic": "strongest_anchor_coverage_strata",
        "variant": args.variant,
        "cfg": source["cfg"],
        "samples": samples,
        "maps": int(torch.unique(map_ids).numel()),
        "diagnosis": diagnosis,
        "protocol": {
            "candidate": f"CFG{source['cfg']} Strongest Anchor {args.variant}",
            "real_real_floor": "paired held-out real query versus independent real donor",
            "macro_unit": "Map within each named stratum",
            "stratification_source": "held-out real query only",
            "relative_aod_spread": "power-weighted circular RMS around real strongest path",
            "support_class": "A: Map-conditioned real support is at or above the locked threshold",
            "minimum_real_support": args.minimum_real_support,
        },
        "real_support_count": {
            "minimum": int(real_support.min()),
            "mean": float(real_support.float().mean()),
            "maximum": int(real_support.max()),
        },
        "strata": results,
        "per_map": per_map,
        "sources": source["sources"] | {"artifact": str(Path(args.artifact).resolve())},
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "strongest_anchor_coverage_strata_complete",
                "output": str(Path(args.output).resolve()),
                "diagnosis": diagnosis,
                "overall_coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
