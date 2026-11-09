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
    strongest_aod,
)
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


CFG_SCALES = (1, 2, 3)
PHYSICAL_NAMES = ("power_db", "delay_spread_ns", "k_factor_db", "los_status")


def _sample_payload(
    payload: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    sample_count = int(payload["path_mask"].shape[0])
    return {
        key: value.to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,)
    }


def _measure(
    payload: Mapping[str, torch.Tensor], target: torch.Tensor
) -> dict[str, torch.Tensor]:
    measured = strongest_aod(payload)
    error = circular_distance_degrees(measured.angle_deg, target)
    path_mask = payload["path_mask"].bool()
    per_path_error = circular_distance_degrees(
        payload["aod_az_deg"].float(), target.float().unsqueeze(-1)
    ).masked_fill(~path_mask, torch.inf)
    nearest_index = per_path_error.argmin(-1)
    nearest_error = per_path_error.gather(
        1, nearest_index.unsqueeze(-1)
    ).squeeze(-1)
    powers = payload["path_powers_linear"].float()
    nearest_power = powers.gather(1, nearest_index.unsqueeze(-1)).squeeze(-1)
    nearest_rank = 1 + ((powers > nearest_power.unsqueeze(-1)) & path_mask).sum(-1)
    result = {
        "valid": measured.valid,
        "error_deg": error,
        "hit_at_5": measured.valid & (error <= 5.0),
        "hit_at_10": measured.valid & (error <= 10.0),
        "strongest_index": measured.path_index,
        "top2_gap_db": measured.top2_power_gap_db,
        "near_tie": measured.near_tie,
        "target_nearest_index": nearest_index,
        "target_nearest_error_deg": nearest_error,
        "target_nearest_power_rank": nearest_rank,
        "target_nearest_is_top1": measured.valid
        & (nearest_index == measured.path_index),
    }
    actual = payload.get("actual_conditions")
    requested = payload.get("conditions")
    validity = payload.get("condition_validity")
    if actual is not None and requested is not None and validity is not None:
        result["physical_absolute_error"] = (actual.float() - requested.float()).abs()
        result["physical_validity"] = validity.bool()
    return result


def _mean(values: torch.Tensor) -> float | None:
    finite = values.float().flatten()
    finite = finite[torch.isfinite(finite)]
    return float(finite.mean()) if finite.numel() else None


def _summary(
    measured: Mapping[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, Any]:
    valid = measured["valid"][indices].bool()
    error = measured["error_deg"][indices][valid]
    gap = measured["top2_gap_db"][indices][valid]
    finite_gap = gap[torch.isfinite(gap)]
    result: dict[str, Any] = {
        "samples": int(indices.numel()),
        "valid_samples": int(valid.sum()),
        "valid_rate": float(valid.float().mean()),
        "mean_error_deg": _mean(error),
        "p95_error_deg": (
            float(torch.quantile(error.float(), 0.95)) if error.numel() else None
        ),
        "hit_at_5_deg": float(measured["hit_at_5"][indices].float().mean()),
        "hit_at_10_deg": float(measured["hit_at_10"][indices].float().mean()),
        "target_nearest_path_top1_rate": float(
            measured["target_nearest_is_top1"][indices].float().mean()
        ),
        "target_nearest_path_power_rank_mean": _mean(
            measured["target_nearest_power_rank"][indices][valid]
        ),
        "target_nearest_path_error_deg_mean": _mean(
            measured["target_nearest_error_deg"][indices][valid]
        ),
        "near_tie_rate": float(measured["near_tie"][indices].float().mean()),
        "top2_power_gap_db_mean": _mean(finite_gap),
    }
    if "physical_absolute_error" in measured:
        absolute = measured["physical_absolute_error"][indices]
        physical_validity = measured["physical_validity"][indices].bool()
        result["physical_condition_mae"] = {
            name: _mean(absolute[:, number][physical_validity[:, number]])
            for number, name in enumerate(PHYSICAL_NAMES)
        }
    return result


def _transition(
    raw: Mapping[str, torch.Tensor],
    final: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, Any]:
    valid = raw["valid"][indices] & final["valid"][indices]
    raw_hit = raw["hit_at_10"][indices]
    final_hit = final["hit_at_10"][indices]
    raw_top1 = raw["target_nearest_is_top1"][indices]
    final_top1 = final["target_nearest_is_top1"][indices]
    identity_changed = (
        raw["strongest_index"][indices] != final["strongest_index"][indices]
    ) & valid
    beneficial_hit = valid & ~raw_hit & final_hit
    harmful_hit = valid & raw_hit & ~final_hit
    beneficial_top1 = valid & ~raw_top1 & final_top1
    harmful_top1 = valid & raw_top1 & ~final_top1
    denominator = valid.float().sum().clamp_min(1.0)

    def rate(mask: torch.Tensor) -> float:
        return float((mask & valid).float().sum() / denominator)

    return {
        "valid_pairs": int(valid.sum()),
        "strongest_identity_changed_rate": rate(identity_changed),
        "hit_at_10_beneficial_change_rate": rate(beneficial_hit),
        "hit_at_10_harmful_change_rate": rate(harmful_hit),
        "hit_at_10_net_change": rate(beneficial_hit) - rate(harmful_hit),
        "target_nearest_top1_beneficial_change_rate": rate(beneficial_top1),
        "target_nearest_top1_harmful_change_rate": rate(harmful_top1),
        "target_nearest_top1_net_change": rate(beneficial_top1)
        - rate(harmful_top1),
        "mean_error_deg_change": _mean(
            final["error_deg"][indices][valid] - raw["error_deg"][indices][valid]
        ),
    }


def _bootstrap_delta(
    values: list[float], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    delta = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    selections = torch.randint(
        delta.numel(),
        (bootstrap_samples, delta.numel()),
        generator=generator,
    )
    distribution = delta[selections].mean(-1)
    interval = torch.quantile(
        distribution, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return {
        "groups": int(delta.numel()),
        "macro_mean": float(delta.mean()),
        "bootstrap_95_percent_ci": [float(interval[0]), float(interval[1])],
    }


def _paired_map_changes(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    definitions = {
        "hit_at_10_deg": ("higher_is_better", "hit_at_10_deg"),
        "target_nearest_path_top1_rate": (
            "higher_is_better",
            "target_nearest_path_top1_rate",
        ),
        "mean_error_deg": ("lower_is_better", "mean_error_deg"),
        "target_nearest_path_power_rank_mean": (
            "lower_is_better",
            "target_nearest_path_power_rank_mean",
        ),
    }
    result: dict[str, Any] = {}
    for number, (name, (direction, field)) in enumerate(definitions.items()):
        deltas = [
            float(record["after_dp"][field]) - float(record["before_dp"][field])
            for record in records
            if record["after_dp"].get(field) is not None
            and record["before_dp"].get(field) is not None
        ]
        if not deltas:
            continue
        section = _bootstrap_delta(
            deltas,
            bootstrap_samples=bootstrap_samples,
            seed=seed + number,
        )
        section["direction"] = direction
        section["delta_definition"] = "after_dp_minus_before_dp"
        result[name] = section
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Strongest-path ranking changes introduced by frozen DP."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum-samples-per-map", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.minimum_samples_per_map < 2:
        raise ValueError("minimum-samples-per-map must be at least two")
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")

    device = resolve_device(args.device)
    source = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if source.get("condition_type") != "strongest":
        raise ValueError("artifact is not Strongest-AoD conditioned")
    if source.get("split") != "test":
        raise ValueError("DP transition diagnosis requires the frozen Test artifact")
    target = source.get("target_aod_deg")
    if not isinstance(target, torch.Tensor):
        raise ValueError("artifact has no target_aod_deg")
    target = target.to(device).float()
    payloads = source.get("payloads", {})
    required = [
        f"aod_cfg{scale}_{stage}"
        for scale in CFG_SCALES
        for stage in ("raw", "dp")
    ]
    missing = [label for label in required if label not in payloads]
    if missing:
        raise ValueError(
            "artifact predates raw-stage persistence; regenerate it. Missing: "
            + ", ".join(missing)
        )
    loaded = {
        label: _sample_payload(payloads[label], device) for label in required
    }
    sample_count = int(target.numel())
    if any(value["path_mask"].shape[0] != sample_count for value in loaded.values()):
        raise ValueError("all CFG raw/DP payloads must align with target_aod_deg")
    map_ids = loaded["aod_cfg1_raw"]["map_id"].long()
    los = loaded["aod_cfg1_raw"]["conditions"][:, 3] >= 0.5
    measured = {
        label: _measure(payload, target) for label, payload in loaded.items()
    }
    all_indices = torch.arange(sample_count, device=device)

    cfg: dict[str, Any] = {}
    for scale in CFG_SCALES:
        raw_label = f"aod_cfg{scale}_raw"
        dp_label = f"aod_cfg{scale}_dp"
        per_map: list[dict[str, Any]] = []
        per_map_los: list[dict[str, Any]] = []
        for map_tensor in torch.unique(map_ids, sorted=True):
            indices = torch.nonzero(map_ids == map_tensor, as_tuple=False).flatten()
            if indices.numel() < args.minimum_samples_per_map:
                continue
            per_map.append(
                {
                    "map_id": int(map_tensor.item()),
                    "samples": int(indices.numel()),
                    "before_dp": _summary(measured[raw_label], indices),
                    "after_dp": _summary(measured[dp_label], indices),
                    "transition": _transition(
                        measured[raw_label], measured[dp_label], indices
                    ),
                }
            )
            for los_value in (0, 1):
                local = indices[los[indices] == bool(los_value)]
                if local.numel() < max(4, args.minimum_samples_per_map // 2):
                    continue
                per_map_los.append(
                    {
                        "map_id": int(map_tensor.item()),
                        "los_status": los_value,
                        "samples": int(local.numel()),
                        "before_dp": _summary(measured[raw_label], local),
                        "after_dp": _summary(measured[dp_label], local),
                        "transition": _transition(
                            measured[raw_label], measured[dp_label], local
                        ),
                    }
                )
        if not per_map:
            raise RuntimeError(f"CFG={scale} has no Map with enough Test samples")
        cfg[str(scale)] = {
            "pooled": {
                "before_dp": _summary(measured[raw_label], all_indices),
                "after_dp": _summary(measured[dp_label], all_indices),
                "transition": _transition(
                    measured[raw_label], measured[dp_label], all_indices
                ),
            },
            "pooled_by_los": {
                str(value): {
                    "before_dp": _summary(
                        measured[raw_label],
                        torch.nonzero(los == bool(value), as_tuple=False).flatten(),
                    ),
                    "after_dp": _summary(
                        measured[dp_label],
                        torch.nonzero(los == bool(value), as_tuple=False).flatten(),
                    ),
                    "transition": _transition(
                        measured[raw_label],
                        measured[dp_label],
                        torch.nonzero(los == bool(value), as_tuple=False).flatten(),
                    ),
                }
                for value in (0, 1)
                if bool((los == bool(value)).any())
            },
            "paired_per_map_after_minus_before": _paired_map_changes(
                per_map,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + scale * 100,
            ),
            "per_map": per_map,
            "per_map_los": per_map_los,
        }

    report = {
        "format_version": 1,
        "diagnostic": "strongest_aod_dp_rank_transitions",
        "artifact": str(Path(args.artifact).resolve()),
        "samples": sample_count,
        "maps": int(torch.unique(map_ids).numel()),
        "cfg": cfg,
        "protocol": {
            "paired_stage_samples": True,
            "raw_and_dp_share_angles_phases_aoa": True,
            "strongest_identity": "final-power deterministic first argmax",
            "target_nearest_path": "minimum circular AoD distance to target",
            "macro_unit": "Map",
        },
        "decision_rule": {
            "no_new_dp_ranking_damage": (
                "At the frozen production CFG, after-DP Hit@10 and target-nearest "
                "Top1 should not fall by more than the declared tolerance; harmful "
                "transition rates must be reported beside beneficial rates."
            )
        },
    }
    if not all(
        math.isfinite(float(cfg[str(scale)]["pooled"]["transition"]["hit_at_10_net_change"]))
        for scale in CFG_SCALES
    ):
        raise FloatingPointError("DP transition report contains a non-finite net change")
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "strongest_dp_rank_transition_complete",
                "output": str(Path(args.output).resolve()),
                "samples": sample_count,
                "cfg": {
                    scale: values["pooled"]["transition"]
                    for scale, values in cfg.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
