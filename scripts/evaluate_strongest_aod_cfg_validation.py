#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.evaluation.aod_support import (
    circular_distance_degrees,
    enforce_final_strongest_anchor,
    strongest_aod,
)
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.models.scene_delay_power_residual import (
    SceneConditionedDelayPowerResidual,
)
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


def _subset(
    payload: Mapping[str, Any], indices: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    sample_count = int(payload["path_mask"].shape[0])
    return {
        key: value[indices].to(device)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,)
    }


def _select_validation_indices(
    reference: Mapping[str, torch.Tensor],
    *,
    path_count: int,
    maximum_per_map: int,
    seed: int,
) -> torch.Tensor:
    count = reference["path_mask"].bool().sum(-1)
    map_ids = reference["map_id"].long()
    generator = torch.Generator().manual_seed(seed)
    chunks: list[torch.Tensor] = []
    for map_id in torch.unique(map_ids, sorted=True):
        indices = torch.nonzero(
            (map_ids == map_id) & (count == path_count), as_tuple=False
        ).flatten()
        if indices.numel() > maximum_per_map:
            indices = indices[
                torch.randperm(indices.numel(), generator=generator)[:maximum_per_map]
            ]
        if indices.numel():
            chunks.append(indices)
    if not chunks:
        raise ValueError(f"Validation cache contains no L={path_count} samples")
    return torch.cat(chunks).sort().values


def _metrics(error: torch.Tensor) -> dict[str, float | int | None]:
    values = error.float().flatten()
    values = values[torch.isfinite(values)]
    if not values.numel():
        return {
            "samples": 0,
            "mean_error_deg": None,
            "p95_error_deg": None,
            "hit_at_5_deg": None,
            "hit_at_10_deg": None,
        }
    return {
        "samples": int(values.numel()),
        "mean_error_deg": float(values.mean()),
        "p95_error_deg": float(torch.quantile(values, 0.95)),
        "hit_at_5_deg": float((values <= 5.0).float().mean()),
        "hit_at_10_deg": float((values <= 10.0).float().mean()),
    }


def _stratified_metrics(
    error: torch.Tensor,
    success: torch.Tensor,
    output_group: torch.Tensor,
    query_group: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    groups = torch.unique(torch.cat((output_group, query_group)), sorted=True)
    for group in groups.tolist():
        output_mask = output_group == group
        query_mask = query_group == group
        report[str(group)] = {
            "control": _metrics(error[output_mask]),
            "queries": int(query_mask.sum()),
            "natural_filter_query_success_rate": (
                float(success[query_mask].float().mean())
                if bool(query_mask.any())
                else None
            ),
        }
    return report


def _load_dp(
    checkpoint: Mapping[str, Any], device: torch.device
) -> SceneConditionedDelayPowerResidual:
    if checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("--dp-checkpoint is not a Delay-Power residual")
    model = SceneConditionedDelayPowerResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(
        checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
    )
    return model.requires_grad_(False).eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Strongest-AoD CFG on independent Validation queries after the "
            "frozen Delay-Power residual."
        )
    )
    parser.add_argument("--aod-checkpoint", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--guidance-scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0]
    )
    parser.add_argument("--maximum-samples-per-map", type=int, default=64)
    parser.add_argument("--candidates-per-query", type=int, default=2)
    parser.add_argument("--outputs-per-query", type=int, default=1)
    parser.add_argument("--natural-angle-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument(
        "--require-strongest-anchor",
        action="store_true",
        help="Reject a legacy/Ranking checkpoint before expensive generation.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.batch_size < 1 or args.maximum_samples_per_map < 1:
        raise ValueError("batch and per-Map sample limits must be positive")
    if not 1 <= args.outputs_per_query <= args.candidates_per_query:
        raise ValueError("outputs-per-query must lie in [1, candidates-per-query]")
    if not 0.0 <= args.natural_angle_tolerance_deg <= 180.0:
        raise ValueError("natural angle tolerance must lie in [0, 180]")
    if any(scale <= 0.0 for scale in args.guidance_scales):
        raise ValueError("every guidance scale must be positive")

    device = resolve_device(args.device)
    aod_checkpoint = torch.load(
        args.aod_checkpoint, map_location="cpu", weights_only=False
    )
    if aod_checkpoint.get("model_kind") not in {
        "multipath_diffusion_scene_id_strongest_aod_conditioned",
        "multipath_diffusion_scene_id_strongest_aod_anchor",
    }:
        raise ValueError("--aod-checkpoint is not a Strongest-AoD B3 checkpoint")
    config = aod_checkpoint["config"]
    if (
        config["multipath_diffusion"].get("aod_conditioning", {}).get(
            "condition_type", "mean"
        )
        != "strongest"
    ):
        raise ValueError("Strongest-AoD checkpoint condition_type is inconsistent")
    anchor_enabled = bool(
        config["multipath_diffusion"]
        .get("aod_conditioning", {})
        .get("anchor_parameterization", {})
        .get("enabled", False)
    )
    if args.require_strongest_anchor and not anchor_enabled:
        raise ValueError("checkpoint does not enable Strongest Anchor parameterization")
    trained_l = int(config.get("aod_conditioned_data", {}).get("path_count", -1))
    if trained_l != 5:
        raise ValueError(f"first-stage CFG validation requires L=5, found L={trained_l}")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("kind") != "b3_dp_residual_cache" or "validation" not in cache.get(
        "splits", {}
    ):
        raise ValueError("--cache must be a B3 DP cache with a Validation split")
    if (
        config["scene_id_experiment"]["fingerprint"]
        != cache.get("scene_plan_fingerprint")
    ):
        raise ValueError("Strongest-AoD checkpoint and cache use different scene splits")
    reference = cache["splits"]["validation"]["reference"]
    indices = _select_validation_indices(
        reference,
        path_count=trained_l,
        maximum_per_map=args.maximum_samples_per_map,
        seed=args.seed,
    )

    system = SystemConfig.from_mapping(config["data"]["system"])
    model = MultipathDiffusionModel(
        config["multipath_diffusion"], system, aod_checkpoint["statistics"]
    ).to(device)
    model.load_state_dict(
        aod_checkpoint.get("ema_state_dict", aod_checkpoint["model_state_dict"])
    )
    model.requires_grad_(False).eval()
    dp_checkpoint = torch.load(
        args.dp_checkpoint, map_location="cpu", weights_only=False
    )
    dp_contract = dp_checkpoint.get("cache_contract", {})
    if (
        dp_contract.get("scene_plan_fingerprint")
        != cache.get("scene_plan_fingerprint")
        or dp_contract.get("b3_checkpoint_sha256")
        != cache.get("b3_checkpoint_sha256")
    ):
        raise ValueError("DP checkpoint was not trained from the supplied B3 cache")
    if dp_checkpoint.get("system") != cache.get("system"):
        raise ValueError("DP checkpoint and cache use different system configurations")
    if system.as_dict() != cache.get("system"):
        raise ValueError("Strongest-AoD checkpoint and cache use different systems")
    dp = _load_dp(dp_checkpoint, device)
    if model.map_id_to_index != dp.map_id_to_index or model.max_paths != dp.max_paths:
        raise ValueError("Strongest-AoD B3 and DP residual contracts do not match")

    sampling = config["sampling"]
    sampling_steps = args.sampling_steps or int(sampling["sampling_steps"])
    errors: dict[float, list[torch.Tensor]] = {
        float(scale): [] for scale in args.guidance_scales
    }
    gaps: dict[float, list[torch.Tensor]] = {
        float(scale): [] for scale in args.guidance_scales
    }
    successes: dict[float, list[torch.Tensor]] = {
        float(scale): [] for scale in args.guidance_scales
    }
    los_flags: dict[float, list[torch.Tensor]] = {
        float(scale): [] for scale in args.guidance_scales
    }
    output_maps: dict[float, list[torch.Tensor]] = {
        float(scale): [] for scale in args.guidance_scales
    }
    query_maps: dict[float, list[torch.Tensor]] = {
        float(scale): [] for scale in args.guidance_scales
    }
    output_angle_bins: dict[float, list[torch.Tensor]] = {
        float(scale): [] for scale in args.guidance_scales
    }
    query_angle_bins: dict[float, list[torch.Tensor]] = {
        float(scale): [] for scale in args.guidance_scales
    }
    query_los_flags: dict[float, list[torch.Tensor]] = {
        float(scale): [] for scale in args.guidance_scales
    }

    with torch.inference_mode():
        for start in range(0, indices.numel(), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            real = _subset(reference, batch_indices, device)
            conditions = real["conditions"].float()
            validity = real["condition_validity"].bool()
            map_ids = real["map_id"].long()
            target = strongest_aod(real).angle_deg
            repeated_conditions = conditions.repeat_interleave(
                args.candidates_per_query, dim=0
            )
            repeated_validity = validity.repeat_interleave(
                args.candidates_per_query, dim=0
            )
            repeated_maps = map_ids.repeat_interleave(args.candidates_per_query)
            repeated_target = target.repeat_interleave(args.candidates_per_query)
            map_index = dp.map_indices(repeated_maps)

            for scale in args.guidance_scales:
                scale = float(scale)
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + start
                )
                generated = model.generate(
                    repeated_conditions,
                    repeated_conditions.shape[0],
                    condition_validity=repeated_validity,
                    sampler=str(sampling["sampler"]),
                    sampling_steps=sampling_steps,
                    eta=float(sampling.get("eta", 0.0)),
                    guidance_scale=scale,
                    forced_path_count=trained_l,
                    strongest_aod_az_deg=repeated_target,
                    map_id=repeated_maps,
                    generator=generator,
                )
                dp_output = dp(
                    generated,
                    repeated_conditions,
                    repeated_validity,
                    map_index=map_index,
                )
                final_payload = dp.apply_to_payload(generated, dp_output)
                if model.strongest_anchor_enabled:
                    final_payload = enforce_final_strongest_anchor(
                        final_payload, repeated_target
                    )
                measured = strongest_aod(final_payload)
                error = circular_distance_degrees(
                    measured.angle_deg, repeated_target
                ).reshape(-1, args.candidates_per_query)
                valid = measured.valid.reshape(-1, args.candidates_per_query)
                repeated_los = (conditions[:, 3] >= 0.5).unsqueeze(-1).expand_as(valid)
                eligible = valid & (
                    error <= args.natural_angle_tolerance_deg
                )
                errors[scale].append(error[valid].cpu())
                los_flags[scale].append(repeated_los[valid].cpu())
                output_maps[scale].append(
                    repeated_maps.reshape(-1, args.candidates_per_query)[valid].cpu()
                )
                repeated_target_matrix = repeated_target.reshape(
                    -1, args.candidates_per_query
                )
                output_angle_bins[scale].append(
                    torch.floor((repeated_target_matrix[valid] + 180.0) / 30.0)
                    .long()
                    .clamp(0, 11)
                    .cpu()
                )
                gaps[scale].append(
                    measured.top2_power_gap_db.reshape(
                        -1, args.candidates_per_query
                    )[valid].cpu()
                )
                successes[scale].append(
                    (eligible.sum(-1) >= args.outputs_per_query).cpu()
                )
                query_maps[scale].append(map_ids.cpu())
                query_los_flags[scale].append((conditions[:, 3] >= 0.5).cpu())
                query_angle_bins[scale].append(
                    torch.floor((target + 180.0) / 30.0)
                    .long()
                    .clamp(0, 11)
                    .cpu()
                )

            print(
                json.dumps(
                    {
                        "event": "strongest_aod_cfg_validation_progress",
                        "complete": min(start + args.batch_size, indices.numel()),
                        "total": int(indices.numel()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    scale_reports: dict[str, Any] = {}
    for scale in args.guidance_scales:
        scale = float(scale)
        error = torch.cat(errors[scale])
        gap = torch.cat(gaps[scale])
        success = torch.cat(successes[scale]).float()
        los = torch.cat(los_flags[scale]).bool()
        output_map = torch.cat(output_maps[scale]).long()
        query_map = torch.cat(query_maps[scale]).long()
        output_bin = torch.cat(output_angle_bins[scale]).long()
        query_bin = torch.cat(query_angle_bins[scale]).long()
        query_los = torch.cat(query_los_flags[scale]).bool()
        strata = {
            "gap_le_0_5_db": gap <= 0.5,
            "gap_0_5_to_3_db": (gap > 0.5) & (gap < 3.0),
            "gap_ge_3_db": gap >= 3.0,
        }
        angle_bin_report = _stratified_metrics(
            error, success, output_bin, query_bin
        )
        named_angle_bins = {
            f"[{lower},{lower + 30})": angle_bin_report.get(str(number), {
                "control": _metrics(torch.empty(0)),
                "queries": 0,
                "natural_filter_query_success_rate": None,
            })
            for number, lower in enumerate(range(-180, 180, 30))
        }
        scale_reports[f"{scale:g}"] = {
            "final_after_dp": _metrics(error),
            "final_after_dp_los": _metrics(error[los]),
            "final_after_dp_nlos": _metrics(error[~los]),
            "natural_filter_query_success_rate": float(success.mean()),
            "natural_filter_query_success_rate_los": (
                float(success[query_los].mean()) if bool(query_los.any()) else None
            ),
            "natural_filter_query_success_rate_nlos": (
                float(success[~query_los].mean()) if bool((~query_los).any()) else None
            ),
            "top2_gap_near_tie_rate": float((gap <= 0.5).float().mean()),
            "top2_power_gap_strata": {
                name: _metrics(error[mask]) for name, mask in strata.items()
            },
            "per_map": _stratified_metrics(
                error, success, output_map, query_map
            ),
            "target_angle_bins_30deg": named_angle_bins,
        }

    result = {
        "format_version": 1,
        "analysis": "strongest_aod_cfg_validation",
        "split": "validation",
        "queries": int(indices.numel()),
        "maps": int(torch.unique(reference["map_id"][indices]).numel()),
        "path_count": trained_l,
        "strongest_anchor_enabled": model.strongest_anchor_enabled,
        "anchor_reasserted_after_dp": model.strongest_anchor_enabled,
        "guidance_scales": [float(value) for value in args.guidance_scales],
        "candidates_per_query": args.candidates_per_query,
        "outputs_per_query": args.outputs_per_query,
        "natural_angle_tolerance_deg": args.natural_angle_tolerance_deg,
        "paired_initial_noise_across_cfg": True,
        "pipeline": ["Strongest-AoD B3", "frozen DP", "final-power Strongest audit"],
        "cfg": scale_reports,
        "selection_note": (
            "Freeze CFG using Validation only; spatial realism and Multipath Recall "
            "remain required gates before production selection."
        ),
        "sources": {
            "aod_checkpoint": str(Path(args.aod_checkpoint).resolve()),
            "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()),
            "cache": str(Path(args.cache).resolve()),
        },
    }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
