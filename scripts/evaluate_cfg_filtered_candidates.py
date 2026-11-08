#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.evaluation.filtered_candidates import circular_angle_bin
from physics_csi.evaluation.multipath_recall import (
    FeatureStandardizer,
    manifold_metrics,
    multipath_view_features,
)
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump
from scripts.evaluate_csi_absolute_quality import Evaluator


LABELS = ("cfg2_filtered", "cfg3_filtered")
LOWER_IS_BETTER = {
    "path_level_power_aod_sliced_wasserstein",
    "link_level_power_aod_sliced_wasserstein",
    "power_weighted_aod_sliced_wasserstein",
    "tx_spatial_correlation_relative_frobenius",
    "raw_full_csi_projected_mmd_rbf",
    "raw_full_csi_projected_sliced_wasserstein",
    "phase_invariant_full_csi_projected_mmd_rbf",
    "phase_invariant_full_csi_projected_sliced_wasserstein",
    "pdp_sliced_wasserstein",
}
HIGHER_IS_BETTER = {
    "multipath_precision",
    "multipath_density",
    "multipath_recall",
    "multipath_coverage",
    "multipath_rare_reference_coverage",
    "self_centered_multipath_precision",
    "self_centered_multipath_density",
    "self_centered_multipath_recall",
    "self_centered_multipath_coverage",
}
METRICS = tuple(sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER))


def _reference_payload(
    payload: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    sample_count = int(payload["path_mask"].shape[0])
    return {
        key: value.to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,)
    }


def _flatten_selected(
    payload: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[dict[str, torch.Tensor], int, int]:
    queries, outputs = payload["path_mask"].shape[:2]
    flattened = {
        key: value.reshape(queries * outputs, *value.shape[2:]).to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:2] == (queries, outputs)
    }
    return flattened, int(queries), int(outputs)


def _synthesize_in_batches(
    payload: Mapping[str, torch.Tensor],
    synthesizer: MIMOOFDMSynthesizer,
    *,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    samples = int(payload["path_mask"].shape[0])
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        chunks.append(
            synthesizer(
                payload["path_delays_ns"][start:stop],
                payload["path_powers_linear"][start:stop],
                payload["path_phases_rad"][start:stop],
                payload["aoa_az_deg"][start:stop],
                payload["aoa_el_deg"][start:stop],
                payload["aod_az_deg"][start:stop],
                payload["aod_el_deg"][start:stop],
                payload["path_mask"][start:stop],
            )
        )
    return torch.cat(chunks)


def _mean_slot_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: sum(item[metric] for item in values) / len(values)
        for metric in METRICS
    }


def _bootstrap_map_comparison(
    cfg2: list[float],
    cfg3: list[float],
    *,
    lower_is_better: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    first = torch.tensor(cfg2, dtype=torch.float64)
    second = torch.tensor(cfg3, dtype=torch.float64)
    if not first.numel() or first.shape != second.shape:
        raise ValueError("paired Map metrics must be non-empty and aligned")
    improvement = second - first if lower_is_better else first - second
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        improvement.numel(),
        (bootstrap_samples, improvement.numel()),
        generator=generator,
    )
    bootstrap = improvement[indices].mean(-1)
    interval = torch.quantile(
        bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    lower, upper = float(interval[0]), float(interval[1])
    evidence = (
        "cfg2_better"
        if lower > 0.0
        else "cfg2_worse"
        if upper < 0.0
        else "inconclusive"
    )
    return {
        "maps": int(first.numel()),
        "cfg2_macro_mean": float(first.mean()),
        "cfg3_macro_mean": float(second.mean()),
        "oriented_improvement_positive_favors_cfg2": float(improvement.mean()),
        "bootstrap_95_percent_ci": [lower, upper],
        "cfg2_map_wins": int((improvement > 1e-12).sum()),
        "cfg2_map_losses": int((improvement < -1e-12).sum()),
        "evidence": evidence,
        "metric_direction": "lower_is_better" if lower_is_better else "higher_is_better",
    }


def _audit(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.float().flatten()
    if not values.numel():
        return {
            "count": 0,
            "mean": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate unbiased 16-to-8 CFG2/CFG3 Mean-AoD filtering."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--angle-bin-width-deg", type=float, default=60.0)
    parser.add_argument("--minimum-queries-per-map-angle", type=int, default=3)
    parser.add_argument("--coverage-k", type=int, default=3)
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--projection-seeds", default="0,1,2")
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--weighted-quantiles", type=int, default=256)
    parser.add_argument("--synthesis-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.minimum_queries_per_map_angle < 2:
        raise ValueError("minimum queries per Map-angle must be at least two")

    device = resolve_device(args.device)
    source = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if source.get("model_kind") != "cfg2_cfg3_filtered_candidate_comparison":
        raise ValueError("unsupported filtered-candidate artifact")
    source_artifact = torch.load(
        source["source_artifact"], map_location="cpu", weights_only=False
    )
    if source_artifact.get("split") != "test":
        raise ValueError("source artifact is not the independent Test split")
    reference = _reference_payload(source_artifact["payloads"]["reference"], device)
    real_query = _reference_payload(source_artifact["payloads"]["real_query"], device)
    target = source_artifact["target_mean_aod_deg"].to(device).float()
    queries = int(reference["path_mask"].shape[0])

    selected: dict[str, dict[str, torch.Tensor]] = {}
    outputs = None
    for source_label, label in (("cfg2", "cfg2_filtered"), ("cfg3", "cfg3_filtered")):
        payload, count, current_outputs = _flatten_selected(
            source["variants"][source_label], device
        )
        if count != queries:
            raise ValueError("selected candidates do not align with source queries")
        if outputs is not None and outputs != current_outputs:
            raise ValueError("CFG2 and CFG3 have different output counts")
        outputs = current_outputs
        selected[label] = payload
    assert outputs is not None

    synthesizer = MIMOOFDMSynthesizer(
        SystemConfig.from_mapping(source["system"])
    ).to(device).eval()
    with torch.inference_mode():
        for label in LABELS:
            selected[label]["csi"] = _synthesize_in_batches(
                selected[label], synthesizer, batch_size=args.synthesis_batch_size
            )
    payloads = {"reference": reference, **selected}
    evaluator = Evaluator(
        payloads,
        projection_dimension=args.projection_dimension,
        projection_seeds=tuple(
            int(item) for item in args.projection_seeds.split(",") if item
        ),
        projections=args.swd_projections,
        quantiles=args.weighted_quantiles,
        coverage_k=args.coverage_k,
    )

    reference_self = multipath_view_features(reference, view="self_centered")
    self_standardizer = FeatureStandardizer.fit(reference_self)
    self_features = {
        "reference": self_standardizer.transform(reference_self),
        **{
            label: self_standardizer.transform(
                multipath_view_features(selected[label], view="self_centered")
            )
            for label in LABELS
        },
    }
    success = {
        "cfg2_filtered": source["selection_audit"]["cfg2"]["success"].to(device),
        "cfg3_filtered": source["selection_audit"]["cfg3"]["success"].to(device),
    }
    common_success = success["cfg2_filtered"] & success["cfg3_filtered"]
    map_ids = reference["map_id"].long()
    angle_bin, angle_center = circular_angle_bin(
        target, width_deg=args.angle_bin_width_deg
    )

    success_groups: list[dict[str, Any]] = []
    fidelity_groups: list[dict[str, Any]] = []
    for map_tensor in torch.unique(map_ids, sorted=True):
        map_id = int(map_tensor.item())
        map_mask = map_ids == map_tensor
        for bin_tensor in torch.unique(angle_bin[map_mask], sorted=True):
            indices = torch.nonzero(
                map_mask & (angle_bin == bin_tensor), as_tuple=False
            ).flatten()
            center = float(angle_center[indices[0]])
            group_success = {
                label: float(success[label][indices].float().mean())
                for label in LABELS
            }
            common = indices[common_success[indices]]
            success_groups.append(
                {
                    "map_id": map_id,
                    "angle_bin": int(bin_tensor.item()),
                    "angle_center_deg": center,
                    "queries": int(indices.numel()),
                    "success_rate": group_success,
                    "common_success_queries": int(common.numel()),
                }
            )
            if common.numel() < args.minimum_queries_per_map_angle:
                continue
            models: dict[str, dict[str, float]] = {}
            for label in LABELS:
                slot_metrics: list[dict[str, float]] = []
                for slot in range(outputs):
                    candidate_indices = common * outputs + slot
                    evaluated = evaluator.evaluate(
                        label, candidate_indices, common, local=True
                    )
                    self_result = manifold_metrics(
                        self_features[label][candidate_indices],
                        self_features["reference"][common],
                        requested_k=args.coverage_k,
                    )
                    slot_metrics.append(
                        {
                            **{metric: evaluated[metric] for metric in METRICS if metric in evaluated},
                            **{
                                f"self_centered_multipath_{metric}": value
                                for metric, value in self_result.metrics.items()
                                if f"self_centered_multipath_{metric}" in METRICS
                            },
                        }
                    )
                models[label] = _mean_slot_metrics(slot_metrics)
            fidelity_groups.append(
                {
                    "map_id": map_id,
                    "angle_bin": int(bin_tensor.item()),
                    "angle_center_deg": center,
                    "queries": int(common.numel()),
                    "models": models,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "cfg_filtered_map_angle_complete",
                        "map_id": map_id,
                        "angle_center_deg": center,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not fidelity_groups:
        raise RuntimeError("no Map-angle group has enough common successful queries")

    by_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in fidelity_groups:
        by_map[record["map_id"]].append(record)
    per_map: list[dict[str, Any]] = []
    for map_id, records in sorted(by_map.items()):
        per_map.append(
            {
                "map_id": map_id,
                "map_angle_groups": len(records),
                "models": {
                    label: {
                        metric: sum(record["models"][label][metric] for record in records)
                        / len(records)
                        for metric in METRICS
                    }
                    for label in LABELS
                },
            }
        )

    macro_per_map = {
        label: {
            metric: sum(record["models"][label][metric] for record in per_map)
            / len(per_map)
            for metric in METRICS
        }
        for label in LABELS
    }
    comparisons = {
        metric: _bootstrap_map_comparison(
            [record["models"]["cfg2_filtered"][metric] for record in per_map],
            [record["models"]["cfg3_filtered"][metric] for record in per_map],
            lower_is_better=metric in LOWER_IS_BETTER,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index,
        )
        for index, metric in enumerate(METRICS)
    }
    success_summary: dict[str, Any] = {}
    for source_label, label in (("cfg2", "cfg2_filtered"), ("cfg3", "cfg3_filtered")):
        current_success = source["selection_audit"][source_label]["success"].float()
        eligible = source["selection_audit"][source_label]["eligible_count"].float()
        selected_error = source["selection_audit"][source_label][
            "selected_error_deg"
        ]
        valid_error = selected_error[torch.isfinite(selected_error)]
        success_summary[label] = {
            "queries": queries,
            "successful_queries": int(current_success.sum()),
            "output_success_rate": float(current_success.mean()),
            "eligible_candidates_out_of_16": _audit(eligible),
            "selected_angle_error_deg": _audit(valid_error),
        }

    report = {
        "format_version": 1,
        "experiment": "cfg2_cfg3_unbiased_16_to_8_filtered_comparison",
        "artifact": str(Path(args.artifact).resolve()),
        "protocol": source["protocol"]
        | {
            "angle_bin_width_deg": args.angle_bin_width_deg,
            "minimum_queries_per_map_angle": args.minimum_queries_per_map_angle,
            "fidelity_evaluation": (
                "Each of the eight accepted output slots is compared separately with "
                "the same independent real donors, then averaged. This avoids an 8:1 "
                "candidate/reference sample-count artifact. Fidelity uses only queries "
                "where both CFG2 and CFG3 successfully returned eight outputs."
            ),
            "bootstrap_unit": "Map after averaging Map-angle groups within Map",
        },
        "success": success_summary,
        "common_success_rate": float(common_success.float().mean()),
        "map_angle_success": success_groups,
        "fidelity_map_angle_groups": len(fidelity_groups),
        "maps_in_fidelity_macro": len(per_map),
        "macro_per_map": macro_per_map,
        "cfg2_vs_cfg3_cluster_bootstrap": comparisons,
        "per_map": per_map,
        "per_map_angle": fidelity_groups,
        "normalization_audit": evaluator.audit
        | {
            "self_centered_multipath": {
                "dimensions": int(self_standardizer.mean.shape[1]),
                "scale_floor": self_standardizer.scale_floor,
                "dimensions_using_floor": self_standardizer.dimensions_using_floor,
                "source": "complete independent real reference",
            }
        },
        "cautions": [
            "The theoretical binomial success rate assumes identically distributed independent candidates; per-condition empirical rates are reported instead.",
            "Map-angle groups sharing a Map are correlated, so bootstrap inference uses Map as the cluster unit.",
            "This compares generation sampling for one trained checkpoint; it is not a training-seed study.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "cfg_filtered_candidate_evaluation_complete",
                "output": str(Path(args.output).resolve()),
                "cfg2_success_rate": success_summary["cfg2_filtered"]["output_success_rate"],
                "cfg3_success_rate": success_summary["cfg3_filtered"]["output_success_rate"],
                "map_angle_groups": len(fidelity_groups),
                "maps": len(per_map),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
