#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.evaluation.aod_support import (
    circular_distance_degrees,
    conditioned_aod_measurement,
)
from physics_csi.evaluation.multipath_recall import (
    FeatureStandardizer,
    FeatureView,
    manifold_metrics,
    multipath_view_features,
    reference_partition_metrics,
)
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


LABELS = ("real_query", "aod_cfg1_dp", "aod_cfg2_dp", "aod_cfg3_dp")
VIEWS: tuple[FeatureView, ...] = (
    "absolute",
    "target_centered",
    "self_centered",
    "without_aod",
)
MANIFOLD_METRICS = (
    "precision",
    "density",
    "recall",
    "coverage",
    "rare_reference_coverage",
    "pairwise_log_ratio_abs",
)
PARTITION_METRICS = (
    "angle_core_le_5deg_recall",
    "angle_shoulder_5_to_10deg_recall",
    "angle_outside_10deg_recall",
    "density_dense_0_to_50pct_recall",
    "density_middle_50_to_90pct_recall",
    "density_rare_top_10pct_recall",
    "angle_core_le_5deg_coverage",
    "angle_shoulder_5_to_10deg_coverage",
    "density_rare_top_10pct_coverage",
)


def _sample_payload(
    payload: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    required = (
        "path_mask",
        "path_delays_ns",
        "path_powers_linear",
        "path_phases_rad",
        "aoa_az_deg",
        "aoa_el_deg",
        "aod_az_deg",
        "aod_el_deg",
        "map_id",
        "conditions",
        "condition_validity",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"payload is missing required tensors: {missing}")
    sample_count = int(payload["path_mask"].shape[0])
    return {
        key: payload[key].to(device)
        for key in required
        if payload[key].shape[:1] == (sample_count,)
    }


def _bootstrap_delta(
    cfg3: list[float],
    cfg2: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    first = torch.tensor(cfg3, dtype=torch.float64)
    second = torch.tensor(cfg2, dtype=torch.float64)
    if not first.numel() or first.shape != second.shape:
        raise ValueError("paired CFG metrics must be non-empty and equally sized")
    delta = first - second
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        delta.numel(),
        (bootstrap_samples, delta.numel()),
        generator=generator,
    )
    bootstrap = delta[indices].mean(-1)
    interval = torch.quantile(
        bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    lower, upper = float(interval[0]), float(interval[1])
    if lower > 0.0:
        evidence = "cfg3_higher"
    elif upper < 0.0:
        evidence = "cfg3_lower"
    else:
        evidence = "inconclusive"
    tolerance = 1e-12
    wins = int((delta > tolerance).sum())
    losses = int((delta < -tolerance).sum())
    ties = int(delta.numel() - wins - losses)
    return {
        "maps_or_groups": int(delta.numel()),
        "cfg2_macro_mean": float(second.mean()),
        "cfg3_macro_mean": float(first.mean()),
        "cfg3_minus_cfg2": float(delta.mean()),
        "bootstrap_95_percent_ci": [lower, upper],
        "cfg3_wins": wins,
        "cfg3_losses": losses,
        "ties": ties,
        "cfg3_win_rate_excluding_ties": (
            wins / (wins + losses) if wins + losses else None
        ),
        "evidence": evidence,
    }


def _evaluate_group(
    *,
    indices: torch.Tensor,
    features: Mapping[str, Mapping[FeatureView, torch.Tensor]],
    reference_angle_error: torch.Tensor,
    coverage_k: int,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for label in LABELS:
        models[label] = {}
        for view in VIEWS:
            result = manifold_metrics(
                features[label][view][indices],
                features["reference"][view][indices],
                requested_k=coverage_k,
            )
            models[label][view] = {
                "manifold": result.metrics,
                "reference_partitions": reference_partition_metrics(
                    result,
                    reference_angle_error_deg=reference_angle_error[indices],
                ),
            }
    return models


def _macro(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label in LABELS:
        output[label] = {}
        for view in VIEWS:
            manifold = {
                metric: sum(
                    record["models"][label][view]["manifold"][metric]
                    for record in records
                )
                / len(records)
                for metric in MANIFOLD_METRICS
            }
            partitions: dict[str, float] = {}
            for metric in PARTITION_METRICS:
                values = [
                    record["models"][label][view]["reference_partitions"].get(
                        metric
                    )
                    for record in records
                ]
                finite = [
                    float(value)
                    for value in values
                    if value is not None and math.isfinite(float(value))
                ]
                if finite:
                    partitions[metric] = sum(finite) / len(finite)
            output[label][view] = {
                "groups": len(records),
                "manifold": manifold,
                "reference_partitions": partitions,
            }
    return output


def _comparisons(
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    comparison_number = 0
    for view in VIEWS:
        output[view] = {"manifold": {}, "reference_partitions": {}}
        for section, metrics in (
            ("manifold", MANIFOLD_METRICS),
            ("reference_partitions", PARTITION_METRICS),
        ):
            for metric in metrics:
                paired = []
                for record in records:
                    first = record["models"]["aod_cfg3_dp"][view][section].get(
                        metric
                    )
                    second = record["models"]["aod_cfg2_dp"][view][section].get(
                        metric
                    )
                    if first is None or second is None:
                        continue
                    if not math.isfinite(float(first)) or not math.isfinite(float(second)):
                        continue
                    paired.append((float(first), float(second)))
                if paired:
                    output[view][section][metric] = _bootstrap_delta(
                        [first for first, _ in paired],
                        [second for _, second in paired],
                        bootstrap_samples=bootstrap_samples,
                        seed=seed + comparison_number,
                    )
                comparison_number += 1
    return output


def _decision(
    comparisons: Mapping[str, Any], *, condition_type: str
) -> dict[str, Any]:
    absolute = comparisons["absolute"]["manifold"]["recall"]
    target_centered = comparisons["target_centered"]["manifold"]["recall"]
    self_centered = comparisons["self_centered"]["manifold"]["recall"]
    without_aod = comparisons["without_aod"]["manifold"]["recall"]
    core = comparisons["absolute"]["reference_partitions"].get(
        "angle_core_le_5deg_recall"
    )
    shoulder = comparisons["absolute"]["reference_partitions"].get(
        "angle_shoulder_5_to_10deg_recall"
    )

    def lower(section: Mapping[str, Any] | None) -> bool:
        return bool(section) and section["bootstrap_95_percent_ci"][1] < 0.0

    def compatible(section: Mapping[str, Any]) -> bool:
        low, high = section["bootstrap_95_percent_ci"]
        return low <= 0.0 <= high

    if not lower(absolute):
        classification = "no_stable_cfg3_recall_loss_detected"
        explanation = (
            "The paired per-Map bootstrap does not establish lower absolute CFG3 recall."
        )
    elif lower(self_centered) or lower(without_aod):
        classification = "evidence_of_within_condition_mode_loss"
        explanation = (
            f"CFG3 remains lower after removing global {condition_type}-AoD orientation, or even after "
            "removing AoD features; this is evidence of lost internal multipath support."
        )
    elif (
        lower(shoulder)
        and (core is None or compatible(core))
        and compatible(self_centered)
        and compatible(without_aod)
    ):
        classification = "consistent_with_reasonable_target_condition_narrowing"
        explanation = (
            "The loss is concentrated in real references near the target-angle shoulder "
            "and disappears after removing global orientation, consistent with sharper "
            "conditioning rather than internal mode collapse."
        )
    else:
        classification = "inconclusive_narrowing_vs_mode_loss"
        explanation = (
            "Absolute recall is lower, but the centered/non-AoD and angle-band evidence "
            "does not isolate a single mechanism."
        )
    return {
        "classification": classification,
        "explanation": explanation,
        "absolute_recall": absolute,
        "target_centered_recall": target_centered,
        "self_centered_recall": self_centered,
        "without_aod_recall": without_aod,
        "angle_core_recall": core,
        "angle_shoulder_recall": shoulder,
        "decision_rule": (
            "A negative CFG3-CFG2 CI after self-centering or removing AoD is treated as "
            "within-condition mode loss. A loss confined to the 5-10 degree shoulder "
            "with centered views compatible with zero is treated as target narrowing."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether CFG3 multipath recall loss is target-angle narrowing or "
            "within-condition mode loss using paired CFG2/CFG3 samples."
        )
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum-samples-per-map", type=int, default=8)
    parser.add_argument("--coverage-k", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.minimum_samples_per_map < 2:
        raise ValueError("minimum-samples-per-map must be at least two")
    if args.coverage_k < 1:
        raise ValueError("coverage-k must be positive")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")

    device = resolve_device(args.device)
    source = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if source.get("model_kind") not in {
        "aod_conditioned_cfg_link_fidelity_comparison",
        "aod_conditioned_cfg_fidelity_comparison",
    }:
        raise ValueError("artifact is not an AoD conditional-fidelity comparison")
    if source.get("split") != "test":
        raise ValueError("recall diagnosis must use the independent test artifact")
    payloads = {
        label: _sample_payload(source["payloads"][label], device)
        for label in ("reference", *LABELS)
    }
    sample_count = int(payloads["reference"]["path_mask"].shape[0])
    if any(
        int(payloads[label]["path_mask"].shape[0]) != sample_count
        for label in LABELS
    ):
        raise ValueError("CFG2, CFG3 and real payloads must be sample-aligned")
    condition_type = source.get("condition_type", "mean")
    target = source.get("target_aod_deg", source.get("target_mean_aod_deg"))
    if not isinstance(target, torch.Tensor):
        raise ValueError("artifact has no controlled AoD target")
    target = target.to(device).float()
    if target.shape != (sample_count,):
        raise ValueError("target AoD must align with every sample")

    raw_features: dict[str, dict[FeatureView, torch.Tensor]] = {
        label: {
            view: multipath_view_features(
                payloads[label],
                view=view,
                target_mean_aod_deg=target,
                condition_type=condition_type,
            )
            for view in VIEWS
        }
        for label in ("reference", *LABELS)
    }
    standardizers = {
        view: FeatureStandardizer.fit(raw_features["reference"][view])
        for view in VIEWS
    }
    features = {
        label: {
            view: standardizers[view].transform(raw_features[label][view])
            for view in VIEWS
        }
        for label in ("reference", *LABELS)
    }
    reference_aod, reference_valid, _ = conditioned_aod_measurement(
        payloads["reference"], condition_type=condition_type
    )
    reference_angle_error = circular_distance_degrees(reference_aod, target)
    reference_angle_error = torch.where(
        reference_valid,
        reference_angle_error,
        torch.full_like(reference_angle_error, torch.inf),
    )
    map_ids = payloads["reference"]["map_id"].long()
    los = payloads["real_query"]["conditions"][:, 3] >= 0.5

    per_map: list[dict[str, Any]] = []
    per_map_los: list[dict[str, Any]] = []
    for map_tensor in torch.unique(map_ids, sorted=True):
        map_id = int(map_tensor.item())
        indices = torch.nonzero(map_ids == map_tensor, as_tuple=False).flatten()
        if indices.numel() < args.minimum_samples_per_map:
            continue
        per_map.append(
            {
                "map_id": map_id,
                "samples": int(indices.numel()),
                "models": _evaluate_group(
                    indices=indices,
                    features=features,
                    reference_angle_error=reference_angle_error,
                    coverage_k=args.coverage_k,
                ),
            }
        )
        for los_value in (0, 1):
            local = indices[los[indices] == bool(los_value)]
            if local.numel() < max(4, args.minimum_samples_per_map // 2):
                continue
            per_map_los.append(
                {
                    "map_id": map_id,
                    "los_status": los_value,
                    "samples": int(local.numel()),
                    "models": _evaluate_group(
                        indices=local,
                        features=features,
                        reference_angle_error=reference_angle_error,
                        coverage_k=args.coverage_k,
                    ),
                }
            )
        print(
            json.dumps(
                {"event": "cfg_recall_map_complete", "map_id": map_id},
                sort_keys=True,
            ),
            flush=True,
        )
    if not per_map:
        raise RuntimeError("no Map has enough aligned samples for recall diagnosis")

    macro = _macro(per_map)
    comparisons = _comparisons(
        per_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    by_los: dict[str, Any] = {}
    for los_value in (0, 1):
        records = [
            record
            for record in per_map_los
            if record["los_status"] == los_value
        ]
        if records:
            by_los[str(los_value)] = {
                "macro": _macro(records),
                "cfg3_vs_cfg2": _comparisons(
                    records,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + 1000 + los_value * 100,
                ),
            }

    report = {
        "format_version": 1,
        "diagnostic": "cfg2_cfg3_multipath_recall_decomposition",
        "condition_type": condition_type,
        "artifact": str(Path(args.artifact).resolve()),
        "samples": sample_count,
        "maps": len(per_map),
        "protocol": {
            "paired_cfg_noise": bool(
                source.get("sampling", {}).get("paired_noise_across_cfg", False)
            ),
            "condition_matching": (
                f"CFG2 and CFG3 use identical Map, physical conditions, target {condition_type} AoD "
                "and paired noise. The independent real donor is nearby under the saved "
                "Map-condition-AoD support protocol rather than necessarily identical."
            ),
            "independent_real_reference": True,
            "macro_unit": "Map",
            "coverage_k": args.coverage_k,
            "feature_views": {
                "absolute": "All link-level path parameters including absolute AoD.",
                "target_centered": "Subtract requested Mean AoD from every path AoD.",
                "self_centered": (
                    "Subtract each link's own power-weighted Mean AoD."
                    if condition_type == "mean"
                    else "Subtract each link's own final-power strongest-path AoD."
                ),
                "without_aod": "Remove all AoD coordinates while retaining other path parameters.",
            },
        },
        "normalization": {
            view: {
                "dimensions": int(standardizers[view].mean.shape[1]),
                "scale_floor": standardizers[view].scale_floor,
                "dimensions_using_floor": standardizers[view].dimensions_using_floor,
                "source": "complete independent real reference, fitted once",
            }
            for view in VIEWS
        },
        "macro_per_map": macro,
        "cfg3_vs_cfg2_paired_per_map": comparisons,
        "decision": _decision(comparisons, condition_type=condition_type),
        "by_los": by_los,
        "per_map": per_map,
        "per_map_los": per_map_los,
        "cautions": [
            "This diagnoses one trained checkpoint and one paired generation seed.",
            "CFG is joint-condition guidance in the current checkpoint, not AoD-only guidance.",
            "Manifold metrics depend on the declared feature representation, k and reference normalization.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "cfg_multipath_recall_diagnostic_complete",
                "output": str(Path(args.output).resolve()),
                "maps": len(per_map),
                "decision": report["decision"]["classification"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
