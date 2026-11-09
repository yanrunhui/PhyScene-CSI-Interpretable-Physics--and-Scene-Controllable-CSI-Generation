#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.delay_power_residual import DelayPowerResidualCacheDataset
from physics_csi.evaluation.aod_support import (
    conditioned_aod_measurement,
    match_independent_supported_pairs,
)
from physics_csi.utils.io import atomic_json_dump


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean()),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit joint Map, physical-condition and controlled-AoD support in RayVerse."
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--path-count", type=int, default=5)
    parser.add_argument("--maximum-condition-distance", type=float, default=1.5)
    parser.add_argument("--maximum-angle-distance-deg", type=float, default=10.0)
    parser.add_argument("--minimum-support", type=int, default=8)
    parser.add_argument("--maximum-pairs-per-map-los", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--condition-type", choices=["mean", "strongest"], default="mean"
    )
    args = parser.parse_args()
    if args.path_count < 1:
        raise ValueError("path-count must be positive")
    if args.maximum_condition_distance <= 0 or args.maximum_angle_distance_deg <= 0:
        raise ValueError("support radii must be positive")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    dataset = DelayPowerResidualCacheDataset(cache, args.split)
    reference = dataset.reference
    condition_std = torch.as_tensor(
        cache.get("b3_statistics", {}).get("condition_std", []),
        dtype=torch.float32,
    )
    if condition_std.numel() != 4:
        raise ValueError("cache has no valid four-field condition_std")
    query, donor, support, pairing = match_independent_supported_pairs(
        reference,
        condition_std=condition_std,
        maximum_condition_distance=args.maximum_condition_distance,
        maximum_angle_distance_deg=args.maximum_angle_distance_deg,
        minimum_support=args.minimum_support,
        path_count=args.path_count,
        maximum_pairs_per_map_los=args.maximum_pairs_per_map_los,
        seed=args.seed,
        condition_type=args.condition_type,
    )
    angle, valid_angle, confidence = conditioned_aod_measurement(
        reference, condition_type=args.condition_type
    )
    path_count = reference["path_mask"].bool().sum(-1)
    los = reference["conditions"][:, 3] >= 0.5
    map_ids = reference["map_id"].long()
    angle_bin = torch.floor((angle + 180.0) / 30.0).long().clamp(0, 11)

    distributions: dict[str, Any] = {}
    for map_id in torch.unique(map_ids, sorted=True).tolist():
        map_section: dict[str, Any] = {}
        for los_value in (0, 1):
            selected = (map_ids == map_id) & (los == bool(los_value))
            if not bool(selected.any()):
                continue
            map_section[str(los_value)] = {
                "samples": int(selected.sum()),
                "path_count_histogram": {
                    str(key): value
                    for key, value in sorted(
                        Counter(path_count[selected].tolist()).items()
                    )
                },
                "controlled_aod_histogram_30deg": {
                    f"[{lower},{lower + 30})": int(
                        ((angle_bin == number) & selected & valid_angle).sum()
                    )
                    for number, lower in enumerate(range(-180, 180, 30))
                },
                "controlled_aod_valid_rate": float(valid_angle[selected].float().mean()),
                "controlled_aod_confidence": _quantiles(
                    confidence[selected & valid_angle & torch.isfinite(confidence)]
                )
                if bool(
                    (selected & valid_angle & torch.isfinite(confidence)).any()
                )
                else {},
            }
        distributions[f"map_{map_id}"] = map_section

    selected_l = path_count == args.path_count
    high_support = selected_l & (support >= args.minimum_support)
    by_los: dict[str, Any] = {}
    for los_value in (0, 1):
        current = selected_l & (los == bool(los_value))
        current_support = support[current]
        by_los[str(los_value)] = {
            "samples": int(current.sum()),
            "support_count": _quantiles(current_support) if current_support.numel() else {},
            "high_support_samples": int((high_support & current).sum()),
            "high_support_rate": float(
                (support[current] >= args.minimum_support).float().mean()
            )
            if current_support.numel()
            else 0.0,
        }
    map_pair_counts = Counter(map_ids[query].tolist())
    report = {
        "format_version": 1,
        "diagnostic": "map_condition_aod_joint_support",
        "condition_type": args.condition_type,
        "cache": str(Path(args.cache).resolve()),
        "split": args.split,
        "samples": len(dataset),
        "protocol": {
            "path_count_for_fidelity": args.path_count,
            "maximum_standardized_condition_distance": args.maximum_condition_distance,
            "maximum_circular_angle_distance_deg": args.maximum_angle_distance_deg,
            "minimum_real_neighbors": args.minimum_support,
            "maximum_pairs_per_map_los": args.maximum_pairs_per_map_los,
            "condition_type": args.condition_type,
            "strongest_near_tie_exclusion_gap_db": (
                0.5 if args.condition_type == "strongest" else None
            ),
            "exact_matching": ["Map", "LoS", "K-factor validity", "Path Count"],
        },
        "support_summary_for_selected_path_count": {
            "all": {
                "samples": int(selected_l.sum()),
                "support_count": _quantiles(support[selected_l]),
                "high_support_samples": int(high_support.sum()),
                "high_support_rate": float(high_support[selected_l].float().mean()),
            },
            "by_los": by_los,
        },
        "independent_pairing": pairing,
        "pairs_per_map": {
            f"map_{map_id}": count for map_id, count in sorted(map_pair_counts.items())
        },
        "per_map_los_angle_and_path_count_distribution": distributions,
        "interpretation": {
            "support_count": (
                "Number of other real links with the same Map/LoS/K-validity/L, "
                "nearby normalized Power-DS-K and nearby circular controlled AoD."
            ),
            "independence": (
                "Query conditions and real donors come from disjoint random halves; "
                "the donor is not the real sample that supplied the generation target."
            ),
        },
    }
    plan = {
        "format_version": 1,
        "kind": "map_condition_aod_supported_pair_plan",
        "condition_type": args.condition_type,
        "split": args.split,
        "cache": str(Path(args.cache).resolve()),
        "cache_scene_plan_fingerprint": cache.get("scene_plan_fingerprint"),
        "query_indices": query,
        "donor_indices": donor,
        "query_support_count": support[query],
        "condition_std": condition_std,
        "protocol": report["protocol"],
        "seed": args.seed,
    }
    output_plan = Path(args.output_plan)
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_plan.with_suffix(output_plan.suffix + ".tmp")
    torch.save(plan, temporary)
    temporary.replace(output_plan)
    atomic_json_dump(report, args.output_report)
    print(
        json.dumps(
            {
                "event": "map_condition_aod_support_complete",
                "report": str(Path(args.output_report).resolve()),
                "plan": str(output_plan.resolve()),
                "pairs": int(query.numel()),
                "maps": len(map_pair_counts),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
