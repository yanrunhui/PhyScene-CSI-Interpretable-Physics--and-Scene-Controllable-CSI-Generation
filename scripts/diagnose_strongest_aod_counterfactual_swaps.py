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

from physics_csi.data.physical_labels import compute_physical_conditions
from physics_csi.evaluation.aod_support import (
    circular_distance_degrees,
    strongest_aod,
)
from physics_csi.physics.constraints import first_active_path_mask
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump
from scripts.evaluate_csi_absolute_quality import Evaluator


VARIANTS = ("original_cfg3_dp", "power_swap", "aod_swap")
KEY_METRICS = (
    "path_level_power_aod_sliced_wasserstein",
    "link_level_power_aod_sliced_wasserstein",
    "power_weighted_aod_sliced_wasserstein",
    "tx_spatial_correlation_relative_frobenius",
    "pdp_sliced_wasserstein",
    "multipath_recall",
    "multipath_coverage",
)
HIGHER_IS_BETTER = {"multipath_recall", "multipath_coverage"}
CONTROL_METRICS = (
    "mean_error_deg",
    "p95_error_deg",
    "hit_at_5_deg",
    "hit_at_10_deg",
)
HIGHER_CONTROL_IS_BETTER = {"hit_at_5_deg", "hit_at_10_deg"}


def _to_device(
    payload: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    sample_count = int(payload["path_mask"].shape[0])
    return {
        key: value.to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,)
    }


def _clone_payload(
    payload: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in payload.items()}


def _swap_columns(
    values: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    apply: torch.Tensor,
) -> torch.Tensor:
    result = values.clone()
    rows = torch.nonzero(apply, as_tuple=False).flatten()
    if not rows.numel():
        return result
    first_column = first[rows]
    second_column = second[rows]
    first_values = values[rows, first_column].clone()
    second_values = values[rows, second_column].clone()
    result[rows, first_column] = second_values
    result[rows, second_column] = first_values
    return result


def _direct_path_mask(payload: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if "direct_path_mask" in payload:
        return payload["direct_path_mask"].bool()
    los = payload["conditions"][:, 3:4] >= 0.5
    return first_active_path_mask(payload["path_mask"]) & los


def _physical_conditions(
    payload: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_physical_conditions(
        payload["path_powers_linear"].float(),
        payload["path_delays_ns"].float(),
        payload["path_mask"].bool(),
        _direct_path_mask(payload),
    )


def _decorate(
    payload: dict[str, torch.Tensor], synthesizer: MIMOOFDMSynthesizer
) -> None:
    conditions, validity = _physical_conditions(payload)
    payload["counterfactual_actual_conditions"] = conditions
    payload["counterfactual_actual_condition_validity"] = validity
    payload["csi"] = synthesizer(
        payload["path_delays_ns"],
        payload["path_powers_linear"],
        payload["path_phases_rad"],
        payload["aoa_az_deg"],
        payload["aoa_el_deg"],
        payload["aod_az_deg"],
        payload["aod_el_deg"],
        payload["path_mask"],
    )


def _make_counterfactuals(
    original: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    synthesizer: MIMOOFDMSynthesizer,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    strongest = strongest_aod(original)
    mask = original["path_mask"].bool()
    error = circular_distance_degrees(
        original["aod_az_deg"].float(), target.unsqueeze(-1)
    ).masked_fill(~mask, torch.inf)
    nearest = error.argmin(-1)
    swap = strongest.valid & (nearest != strongest.path_index)

    power_swap = _clone_payload(original)
    power_swap["path_powers_linear"] = _swap_columns(
        original["path_powers_linear"],
        strongest.path_index,
        nearest,
        swap,
    )
    _decorate(power_swap, synthesizer)

    aod_swap = _clone_payload(original)
    for key in ("aod_az_deg", "aod_el_deg"):
        aod_swap[key] = _swap_columns(
            original[key], strongest.path_index, nearest, swap
        )
    _decorate(aod_swap, synthesizer)

    original_copy = _clone_payload(original)
    _decorate(original_copy, synthesizer)
    return {
        "original_cfg3_dp": original_copy,
        "power_swap": power_swap,
        "aod_swap": aod_swap,
    }, {
        "strongest_index": strongest.path_index,
        "target_nearest_index": nearest,
        "swap_applied": swap,
        "target_nearest_error_deg": error.gather(
            1, nearest.unsqueeze(-1)
        ).squeeze(-1),
    }


def _control_metrics(
    payload: Mapping[str, torch.Tensor], target: torch.Tensor
) -> dict[str, Any]:
    measured = strongest_aod(payload)
    error = circular_distance_degrees(measured.angle_deg, target)
    valid_error = error[measured.valid]
    finite_gap = measured.top2_power_gap_db[
        measured.valid & torch.isfinite(measured.top2_power_gap_db)
    ]
    return {
        "samples": int(error.numel()),
        "valid_rate": float(measured.valid.float().mean()),
        "mean_error_deg": float(valid_error.mean()),
        "p95_error_deg": float(torch.quantile(valid_error, 0.95)),
        "hit_at_5_deg": float((valid_error <= 5.0).float().mean()),
        "hit_at_10_deg": float((valid_error <= 10.0).float().mean()),
        "near_tie_rate": float(measured.near_tie.float().mean()),
        "top2_power_gap_db_mean": (
            float(finite_gap.mean()) if finite_gap.numel() else None
        ),
    }


def _physical_delta(
    original: Mapping[str, torch.Tensor],
    variant: Mapping[str, torch.Tensor],
) -> dict[str, float | None]:
    first, first_validity = _physical_conditions(original)
    second, second_validity = _physical_conditions(variant)
    names = ("total_power_db", "delay_spread_ns", "k_factor_db", "los_status")
    result: dict[str, float | None] = {}
    for index, name in enumerate(names):
        valid = first_validity[:, index] & second_validity[:, index]
        result[f"{name}_mae"] = (
            float((second[valid, index] - first[valid, index]).abs().mean())
            if bool(valid.any())
            else None
        )
        result[f"{name}_signed_mean_change"] = (
            float((second[valid, index] - first[valid, index]).mean())
            if bool(valid.any())
            else None
        )
    return result


def _macro(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in VARIANTS:
        result[label] = {
            "groups": len(records),
            "control": {
                key: sum(float(record["control"][label][key]) for record in records)
                / len(records)
                for key in (
                    "valid_rate",
                    "mean_error_deg",
                    "p95_error_deg",
                    "hit_at_5_deg",
                    "hit_at_10_deg",
                    "near_tie_rate",
                    "top2_power_gap_db_mean",
                )
                if all(record["control"][label].get(key) is not None for record in records)
            },
            "fidelity": {
                metric: sum(float(record["fidelity"][label][metric]) for record in records)
                / len(records)
                for metric in KEY_METRICS
            },
        }
    return result


def _paired_delta(
    counterfactual: list[float],
    original: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
    higher_is_better: bool,
) -> dict[str, Any]:
    first = torch.tensor(counterfactual, dtype=torch.float64)
    second = torch.tensor(original, dtype=torch.float64)
    raw_delta = first - second
    oriented_delta = -raw_delta if higher_is_better else raw_delta
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        oriented_delta.numel(),
        (bootstrap_samples, oriented_delta.numel()),
        generator=generator,
    )
    bootstrap = oriented_delta[indices].mean(-1)
    interval = torch.quantile(
        bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return {
        "groups": int(raw_delta.numel()),
        "original_macro_mean": float(second.mean()),
        "counterfactual_macro_mean": float(first.mean()),
        "counterfactual_minus_original_raw": float(raw_delta.mean()),
        "oriented_delta_negative_is_better": float(oriented_delta.mean()),
        "oriented_bootstrap_95_percent_ci": [
            float(interval[0]),
            float(interval[1]),
        ],
        "counterfactual_wins": int((oriented_delta < 0.0).sum()),
        "counterfactual_losses": int((oriented_delta > 0.0).sum()),
    }


def _comparisons(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    number = 0
    for label in ("power_swap", "aod_swap"):
        output[label] = {"control": {}, "fidelity": {}}
        for metric in CONTROL_METRICS:
            section = _paired_delta(
                [record["control"][label][metric] for record in records],
                [
                    record["control"]["original_cfg3_dp"][metric]
                    for record in records
                ],
                bootstrap_samples=bootstrap_samples,
                seed=seed + number,
                higher_is_better=metric in HIGHER_CONTROL_IS_BETTER,
            )
            section["direction"] = (
                "higher_is_better"
                if metric in HIGHER_CONTROL_IS_BETTER
                else "lower_is_better"
            )
            output[label]["control"][metric] = section
            number += 1
        for metric in KEY_METRICS:
            counterfactual = [record["fidelity"][label][metric] for record in records]
            original = [
                record["fidelity"]["original_cfg3_dp"][metric]
                for record in records
            ]
            section = _paired_delta(
                counterfactual,
                original,
                bootstrap_samples=bootstrap_samples,
                seed=seed + number,
                higher_is_better=metric in HIGHER_IS_BETTER,
            )
            section["direction"] = (
                "higher_is_better" if metric in HIGHER_IS_BETTER else "lower_is_better"
            )
            section["reported_delta_note"] = (
                "Raw delta preserves metric direction; oriented negative means the "
                "counterfactual is better."
            )
            output[label]["fidelity"][metric] = section
            number += 1
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterfactual Power-Swap and AoD-Swap diagnosis for Strongest AoD."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-label", default="aod_cfg3_dp")
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
    if args.minimum_samples_per_map < 2 or args.bootstrap_samples < 100:
        raise ValueError("map minimum must be >=2 and bootstrap samples >=100")

    device = resolve_device(args.device)
    source = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if source.get("model_kind") not in {
        "aod_conditioned_cfg_fidelity_comparison",
        "aod_conditioned_cfg_link_fidelity_comparison",
    }:
        raise ValueError("artifact is not an AoD conditional-fidelity comparison")
    if source.get("condition_type") != "strongest" or source.get("split") != "test":
        raise ValueError("counterfactual diagnosis requires the Strongest-AoD Test artifact")
    if args.candidate_label not in source.get("payloads", {}):
        raise ValueError(f"artifact has no candidate payload {args.candidate_label}")

    reference = _to_device(source["payloads"]["reference"], device)
    original = _to_device(source["payloads"][args.candidate_label], device)
    target = source.get("target_aod_deg")
    if not isinstance(target, torch.Tensor):
        raise ValueError("artifact has no target_aod_deg tensor")
    target = target.to(device).float()
    system = SystemConfig.from_mapping(source["system"])
    synthesizer = MIMOOFDMSynthesizer(system).to(device).eval()
    with torch.inference_mode():
        variants, swap_audit = _make_counterfactuals(
            original, target, synthesizer
        )
        payloads = {"reference": reference, **variants}
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

        map_ids = reference["map_id"].long()
        los = original["conditions"][:, 3] >= 0.5
        records: list[dict[str, Any]] = []
        los_records: list[dict[str, Any]] = []
        for map_tensor in torch.unique(map_ids, sorted=True):
            map_id = int(map_tensor)
            indices = torch.nonzero(map_ids == map_tensor, as_tuple=False).flatten()
            if indices.numel() < args.minimum_samples_per_map:
                continue
            records.append(
                {
                    "map_id": map_id,
                    "samples": int(indices.numel()),
                    "control": {
                        label: _control_metrics(
                            {key: value[indices] for key, value in variants[label].items()},
                            target[indices],
                        )
                        for label in VARIANTS
                    },
                    "fidelity": {
                        label: evaluator.evaluate(label, indices, indices)
                        for label in VARIANTS
                    },
                }
            )
            for los_value in (0, 1):
                local = indices[los[indices] == bool(los_value)]
                if local.numel() < max(4, args.minimum_samples_per_map // 2):
                    continue
                los_records.append(
                    {
                        "map_id": map_id,
                        "los_status": los_value,
                        "samples": int(local.numel()),
                        "control": {
                            label: _control_metrics(
                                {key: value[local] for key, value in variants[label].items()},
                                target[local],
                            )
                            for label in VARIANTS
                        },
                        "fidelity": {
                            label: evaluator.evaluate(
                                label, local, local, local=True
                            )
                            for label in VARIANTS
                        },
                    }
                )
            print(
                json.dumps(
                    {"event": "strongest_counterfactual_map_complete", "map_id": map_id}
                ),
                flush=True,
            )
    if not records:
        raise RuntimeError("no Map has enough samples for counterfactual evaluation")

    by_los = {
        str(value): _macro(
            [record for record in los_records if record["los_status"] == value]
        )
        for value in (0, 1)
        if any(record["los_status"] == value for record in los_records)
    }
    report = {
        "format_version": 1,
        "diagnostic": "strongest_aod_power_and_aod_swap_counterfactual",
        "artifact": str(Path(args.artifact).resolve()),
        "candidate_label": args.candidate_label,
        "samples": int(target.numel()),
        "maps": len(records),
        "protocol": {
            "power_swap": (
                "Swap the final-power strongest path's scalar power with the path whose "
                "AoD azimuth is nearest to the target; all path angles stay fixed."
            ),
            "aod_swap": (
                "Swap AoD azimuth and elevation between those two paths; powers, delays, "
                "phases and AoA stay fixed."
            ),
            "swap_only_when_indices_differ": True,
            "csi_resynthesized_after_every_counterfactual": True,
            "held_out_reference": "same independent Test donors as the fidelity artifact",
            "macro_unit": "Map",
        },
        "swap_audit": {
            "swap_applied_samples": int(swap_audit["swap_applied"].sum()),
            "swap_applied_rate": float(swap_audit["swap_applied"].float().mean()),
            "target_nearest_error_deg_mean": float(
                swap_audit["target_nearest_error_deg"].mean()
            ),
            "target_nearest_error_deg_p95": float(
                torch.quantile(swap_audit["target_nearest_error_deg"], 0.95)
            ),
        },
        "physical_side_effects_vs_original": {
            label: _physical_delta(variants["original_cfg3_dp"], variants[label])
            for label in ("power_swap", "aod_swap")
        },
        "macro_per_map": _macro(records),
        "macro_per_map_by_los": by_los,
        "counterfactual_vs_original_paired_per_map": _comparisons(
            records,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "normalization_audit": evaluator.audit,
        "per_map": records,
        "per_map_los": los_records,
        "interpretation": {
            "ranking_hypothesis_supported_if": (
                "Power Swap sharply improves Strongest Hit@10 while its Tx/Power-AoD/Recall "
                "metrics worsen or fail to improve."
            ),
            "postprocessing_not_a_solution_if": (
                "Either swap improves control but degrades held-out spatial fidelity or recall."
            ),
        },
        "cautions": [
            "These are causal counterfactual diagnostics, not valid production transforms.",
            "Power Swap preserves total power but can change DS, K-factor and PDP.",
            "AoD Swap preserves the AoD set but changes Power/Delay/AoD path association.",
        ],
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "strongest_aod_counterfactual_complete",
                "output": str(Path(args.output).resolve()),
                "samples": int(target.numel()),
                "maps": len(records),
                "swap_rate": report["swap_audit"]["swap_applied_rate"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
