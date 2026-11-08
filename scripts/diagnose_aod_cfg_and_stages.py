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

from physics_csi.data.aod_conditioned import (
    AoDConditionedDataset,
    make_aod_conditioned_datasets,
    power_weighted_mean_aod,
)
from physics_csi.data.scene_id import make_scene_id_datasets
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.models.scene_delay_power_residual import (
    SceneConditionedDelayPowerResidual,
)
from physics_csi.models.scene_link_global_aod_residual import (
    SceneConditionedLinkGlobalAoDResidual,
)
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import move_to_device, resolve_device
from physics_csi.utils.io import atomic_json_dump
from physics_csi.utils.seed import seed_everything


def _csv_floats(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or not all(math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("expected a comma-separated finite number list")
    return result


def _csv_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from error
    if not result:
        raise argparse.ArgumentTypeError("integer list cannot be empty")
    return result


def _load_dp(
    checkpoint: Mapping[str, Any], device: torch.device
) -> SceneConditionedDelayPowerResidual:
    if checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("--dp-checkpoint is not a Delay-Power residual")
    model = SceneConditionedDelayPowerResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    return model.requires_grad_(False).eval()


def _load_link(
    checkpoint: Mapping[str, Any], device: torch.device
) -> SceneConditionedLinkGlobalAoDResidual:
    if checkpoint.get("model_kind") != "scene_link_global_aod_residual":
        raise ValueError("--link-checkpoint is not a Link-global AoD residual")
    model = SceneConditionedLinkGlobalAoDResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    return model.requires_grad_(False).eval()


def _synthesize(
    payload: Mapping[str, torch.Tensor], synthesizer: MIMOOFDMSynthesizer
) -> torch.Tensor:
    return synthesizer(
        payload["path_delays_ns"],
        payload["path_powers_linear"],
        payload["path_phases_rad"],
        payload["aoa_az_deg"],
        payload["aoa_el_deg"],
        payload["aod_az_deg"],
        payload["aod_el_deg"],
        payload["path_mask"],
    )


def _circular_error(actual_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    difference = torch.deg2rad(actual_deg.float() - target_deg.float())
    return torch.rad2deg(torch.atan2(difference.sin(), difference.cos())).abs()


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten().cpu()
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def _stage_metrics(
    payload: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    target_conditions: torch.Tensor,
    target_validity: torch.Tensor,
    synthesizer: MIMOOFDMSynthesizer,
) -> dict[str, Any]:
    actual, concentration = power_weighted_mean_aod(
        payload["path_powers_linear"], payload["aod_az_deg"], payload["path_mask"]
    )
    error = _circular_error(actual, target)
    mask = payload["path_mask"].bool()
    power = payload["path_powers_linear"].float().clamp_min(0.0) * mask
    weights = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
    relative = torch.deg2rad(payload["aod_az_deg"].float() - actual.unsqueeze(-1))
    relative = torch.atan2(relative.sin(), relative.cos())
    within_link_spread = torch.rad2deg(
        (weights * relative.square()).sum(-1).clamp_min(0.0).sqrt()
    )
    # Residual ``apply_to_payload`` helpers preserve tensor metadata from the
    # previous stage, including a cached CSI tensor.  That tensor no longer
    # corresponds to the corrected Delay/Power/AoD parameters.  Always
    # synthesize again here so every stage is evaluated from its own paths.
    csi = _synthesize(payload, synthesizer)
    feature = torch.cat(
        [csi.real.reshape(csi.shape[0], -1), csi.imag.reshape(csi.shape[0], -1)],
        dim=-1,
    ).float()
    feature = feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    centered = feature - feature.mean(0, keepdim=True)
    csi_dispersion = centered.square().sum(-1).mean().sqrt()
    feasible = payload.get("constraint_feasible")
    actual_conditions = payload.get("actual_conditions")
    condition_error: dict[str, float] = {}
    if actual_conditions is not None:
        names = ("total_power_db", "delay_spread_ns", "k_factor_db", "los_status")
        difference = (actual_conditions.float() - target_conditions.float()).abs()
        for index, name in enumerate(names):
            selected = target_validity[:, index].bool()
            if bool(selected.any()):
                condition_error[name] = float(difference[selected, index].mean())
    return {
        "samples": int(error.numel()),
        "mean_aod_error_deg": _quantiles(error),
        "hit_at_5deg": float((error <= 5.0).float().mean()),
        "hit_at_10deg": float((error <= 10.0).float().mean()),
        "mean_aod_concentration": _quantiles(concentration),
        "within_link_relative_aod_spread_deg": _quantiles(within_link_spread),
        "normalized_csi_rms_dispersion": float(csi_dispersion),
        "constraint_feasible_rate": (
            float(feasible.float().mean()) if feasible is not None else None
        ),
        "mean_absolute_physical_condition_error": condition_error,
    }


def _combine_stage_metrics(cells: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(cell["samples"]) for cell in cells)
    if total == 0:
        return {}

    def weighted(path: tuple[str, ...]) -> float:
        value = 0.0
        for cell in cells:
            current: Any = cell
            for key in path:
                current = current[key]
            value += int(cell["samples"]) * float(current)
        return value / total

    physical_names = sorted(
        {
            name
            for cell in cells
            for name in cell["mean_absolute_physical_condition_error"]
        }
    )
    return {
        "samples": total,
        "mean_aod_error_deg": {
            "mean": weighted(("mean_aod_error_deg", "mean")),
            "p50_cell_weighted": weighted(("mean_aod_error_deg", "p50")),
            "p95_cell_weighted": weighted(("mean_aod_error_deg", "p95")),
        },
        "hit_at_5deg": weighted(("hit_at_5deg",)),
        "hit_at_10deg": weighted(("hit_at_10deg",)),
        "mean_aod_concentration": weighted(("mean_aod_concentration", "mean")),
        "within_link_relative_aod_spread_deg": weighted(
            ("within_link_relative_aod_spread_deg", "mean")
        ),
        "normalized_csi_rms_dispersion": weighted(
            ("normalized_csi_rms_dispersion",)
        ),
        "constraint_feasible_rate": (
            weighted(("constraint_feasible_rate",))
            if all(cell["constraint_feasible_rate"] is not None for cell in cells)
            else None
        ),
        "mean_absolute_physical_condition_error": {
            name: sum(
                int(cell["samples"])
                * float(cell["mean_absolute_physical_condition_error"].get(name, 0.0))
                for cell in cells
                if name in cell["mean_absolute_physical_condition_error"]
            )
            / max(
                sum(
                    int(cell["samples"])
                    for cell in cells
                    if name in cell["mean_absolute_physical_condition_error"]
                ),
                1,
            )
            for name in physical_names
        },
    }


def _condition_batch(
    dataset: AoDConditionedDataset,
    candidates: list[int],
    sample_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not candidates:
        raise ValueError("condition pool is empty")
    generator = torch.Generator().manual_seed(seed)
    if len(candidates) >= sample_count:
        selected = torch.randperm(len(candidates), generator=generator)[:sample_count]
    else:
        selected = torch.randint(
            0, len(candidates), (sample_count,), generator=generator
        )
    records = [dataset[candidates[int(index)]] for index in selected]
    return (
        torch.stack([record["conditions"].float() for record in records]),
        torch.stack([record["condition_validity"].bool() for record in records]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan CFG and trace Mean-AoD control across B3, DP and Link-global."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dp-checkpoint", default=None)
    parser.add_argument("--link-checkpoint", default=None)
    parser.add_argument("--maps", type=_csv_ints, default=_csv_ints("10,25,50,75,96"))
    parser.add_argument(
        "--target-angles",
        type=_csv_floats,
        default=_csv_floats("-179,-150,-90,-30,30,38.5,90,150,179"),
    )
    parser.add_argument(
        "--guidance-scales",
        type=_csv_floats,
        default=_csv_floats("1.0,1.5,2.0,3.0"),
    )
    parser.add_argument("--los-statuses", type=_csv_ints, default=[0, 1])
    parser.add_argument("--samples-per-cell", type=int, default=128)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.samples_per_cell <= 0:
        raise ValueError("samples-per-cell must be positive")
    if any(value <= 0 for value in args.guidance_scales):
        raise ValueError("guidance scales must be positive")
    if any(value not in (0, 1) for value in args.los_statuses):
        raise ValueError("los-statuses may contain only 0 and 1")
    if args.link_checkpoint and not args.dp_checkpoint:
        raise ValueError("--link-checkpoint requires --dp-checkpoint")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "multipath_diffusion_scene_id_aod_conditioned":
        raise ValueError("checkpoint is not an AoD-conditioned scene-ID diffusion model")
    config = checkpoint["config"]
    plan = config.get("scene_id_experiment")
    if plan is None:
        raise ValueError("checkpoint has no embedded scene split")
    model = MultipathDiffusionModel(
        config["multipath_diffusion"],
        SystemConfig.from_mapping(config["data"]["system"]),
        checkpoint["statistics"],
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.requires_grad_(False).eval()
    unknown_maps = sorted(set(args.maps) - set(model.map_id_to_index))
    if unknown_maps:
        raise ValueError(f"checkpoint has no Map IDs: {unknown_maps}")

    data_config = dict(config.get("aod_conditioned_data", {}))
    datasets = make_aod_conditioned_datasets(
        make_scene_id_datasets(config["data"]["prepared_root"], plan),
        path_count=(
            int(data_config["path_count"])
            if data_config.get("path_count") is not None
            else None
        ),
        minimum_concentration=float(data_config.get("minimum_concentration", 0.10)),
        reject_low_concentration=bool(data_config.get("reject_low_concentration", True)),
    )
    test_dataset = datasets["test"]
    pool: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (map_id, los_status) in enumerate(
        zip(test_dataset.map_ids, test_dataset.los_status)
    ):
        pool[(int(map_id), int(los_status))].append(index)

    dp_model: SceneConditionedDelayPowerResidual | None = None
    link_model: SceneConditionedLinkGlobalAoDResidual | None = None
    if args.dp_checkpoint:
        dp_checkpoint = torch.load(
            args.dp_checkpoint, map_location="cpu", weights_only=False
        )
        dp_model = _load_dp(dp_checkpoint, device)
        if dp_model.map_id_to_index != model.map_id_to_index:
            raise ValueError("AoD-conditioned B3 and DP residual use different Map mappings")
    if args.link_checkpoint:
        link_checkpoint = torch.load(
            args.link_checkpoint, map_location="cpu", weights_only=False
        )
        link_model = _load_link(link_checkpoint, device)
        if link_model.map_id_to_index != model.map_id_to_index:
            raise ValueError("AoD-conditioned B3 and Link-global use different Map mappings")

    sampling = config["sampling"]
    sampling_steps = args.sampling_steps or int(sampling["sampling_steps"])
    trained_path_count = int(data_config.get("path_count", 5))
    synthesizer = MIMOOFDMSynthesizer(
        SystemConfig.from_mapping(config["data"]["system"])
    ).to(device).eval()
    cells: list[dict[str, Any]] = []
    stage_names = ["aod_conditioned_b3"]
    if dp_model is not None:
        stage_names.append("b3_plus_dp")
    if link_model is not None:
        stage_names.append("b3_plus_dp_plus_link_global")

    with torch.inference_mode():
        for map_number, map_id in enumerate(args.maps):
            for los_status in args.los_statuses:
                candidates = pool.get((map_id, los_status), [])
                if not candidates:
                    cells.append(
                        {
                            "map_id": map_id,
                            "los_status": los_status,
                            "status": "no_matching_L5_test_conditions",
                            "real_condition_support": 0,
                        }
                    )
                    continue
                condition_seed = args.seed + map_number * 1009 + los_status * 101
                conditions_cpu, validity_cpu = _condition_batch(
                    test_dataset,
                    candidates,
                    args.samples_per_cell,
                    condition_seed,
                )
                conditions = conditions_cpu.to(device)
                validity = validity_cpu.to(device)
                map_ids = torch.full(
                    (args.samples_per_cell,), map_id, device=device, dtype=torch.long
                )
                map_indices = model.map_indices(map_ids, args.samples_per_cell, device)
                for angle_number, target_angle in enumerate(args.target_angles):
                    target = torch.full(
                        (args.samples_per_cell,),
                        target_angle,
                        device=device,
                        dtype=torch.float32,
                    )
                    generation_seed = (
                        args.seed
                        + map_number * 100_003
                        + los_status * 10_007
                        + angle_number * 503
                    )
                    for guidance_scale in args.guidance_scales:
                        generator = torch.Generator(device=device).manual_seed(
                            generation_seed
                        )
                        base = model.generate(
                            conditions,
                            args.samples_per_cell,
                            condition_validity=validity,
                            map_id=map_ids,
                            sampler=str(sampling["sampler"]),
                            sampling_steps=sampling_steps,
                            eta=float(sampling.get("eta", 0.0)),
                            guidance_scale=guidance_scale,
                            path_count_temperature=float(
                                sampling.get("path_count_temperature", 1.0)
                            ),
                            forced_path_count=trained_path_count,
                            mean_aod_az_deg=target,
                            generator=generator,
                        )
                        stages: dict[str, Mapping[str, torch.Tensor]] = {
                            "aod_conditioned_b3": base
                        }
                        if dp_model is not None:
                            dp_output = dp_model(
                                base,
                                conditions,
                                validity,
                                map_index=map_indices,
                            )
                            dp_payload = dp_model.apply_to_payload(base, dp_output)
                            stages["b3_plus_dp"] = dp_payload
                            if link_model is not None:
                                link_output = link_model(
                                    dp_payload,
                                    conditions,
                                    validity,
                                    map_index=map_indices,
                                )
                                stages["b3_plus_dp_plus_link_global"] = (
                                    link_model.apply_to_payload(dp_payload, link_output)
                                )
                        cell = {
                            "map_id": map_id,
                            "los_status": los_status,
                            "target_mean_aod_deg": target_angle,
                            "guidance_scale": guidance_scale,
                            "status": "ok",
                            "samples": args.samples_per_cell,
                            "real_condition_support": len(candidates),
                            "generation_seed": generation_seed,
                            "stages": {
                                name: _stage_metrics(
                                    payload,
                                    target,
                                    conditions,
                                    validity,
                                    synthesizer,
                                )
                                for name, payload in stages.items()
                            },
                        }
                        if "b3_plus_dp" in cell["stages"]:
                            cell["dp_mean_aod_error_change_deg"] = (
                                cell["stages"]["b3_plus_dp"]["mean_aod_error_deg"][
                                    "mean"
                                ]
                                - cell["stages"]["aod_conditioned_b3"][
                                    "mean_aod_error_deg"
                                ]["mean"]
                            )
                        if "b3_plus_dp_plus_link_global" in cell["stages"]:
                            cell["link_mean_aod_error_change_deg"] = (
                                cell["stages"]["b3_plus_dp_plus_link_global"][
                                    "mean_aod_error_deg"
                                ]["mean"]
                                - cell["stages"]["b3_plus_dp"][
                                    "mean_aod_error_deg"
                                ]["mean"]
                            )
                        cells.append(cell)
                        print(
                            json.dumps(
                                {
                                    "event": "aod_cfg_cell_complete",
                                    "map_id": map_id,
                                    "los_status": los_status,
                                    "target_angle": target_angle,
                                    "guidance_scale": guidance_scale,
                                    "base_mean_error_deg": cell["stages"][
                                        "aod_conditioned_b3"
                                    ]["mean_aod_error_deg"]["mean"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

    valid_cells = [cell for cell in cells if cell.get("status") == "ok"]
    summaries: dict[str, Any] = {}
    for guidance_scale in args.guidance_scales:
        selected = [
            cell
            for cell in valid_cells
            if float(cell["guidance_scale"]) == guidance_scale
        ]
        summaries[str(guidance_scale)] = {
            "all": {
                stage: _combine_stage_metrics(
                    [cell["stages"][stage] for cell in selected]
                )
                for stage in stage_names
            },
            "by_los": {
                str(los_status): {
                    stage: _combine_stage_metrics(
                        [
                            cell["stages"][stage]
                            for cell in selected
                            if int(cell["los_status"]) == los_status
                        ]
                    )
                    for stage in stage_names
                }
                for los_status in args.los_statuses
            },
        }

    report = {
        "format_version": 1,
        "diagnostic": "aod_conditioned_cfg_and_residual_stage_audit",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dp_checkpoint": (
            str(Path(args.dp_checkpoint).resolve()) if args.dp_checkpoint else None
        ),
        "link_checkpoint": (
            str(Path(args.link_checkpoint).resolve()) if args.link_checkpoint else None
        ),
        "protocol": {
            "maps": args.maps,
            "target_angles_deg": args.target_angles,
            "los_statuses": args.los_statuses,
            "guidance_scales": args.guidance_scales,
            "samples_per_cell": args.samples_per_cell,
            "sampling_steps": sampling_steps,
            "forced_path_count": trained_path_count,
            "physical_conditions": (
                "Sampled from independent L=5 test records for the same Map and LoS; "
                "the exact same conditions and initial noise seed are reused across CFG scales."
            ),
        },
        "cfg_implementation_audit": {
            "formula": "epsilon_uncond + s * (epsilon_cond - epsilon_uncond)",
            "scale_one_equals_conditional_prediction": True,
            "current_unconditional_scope": "all_conditions",
            "strengthened_together": [
                "Power",
                "Delay Spread",
                "K-factor/validity",
                "LoS",
                "Map ID",
                "Mean AoD",
            ],
            "pure_aod_cfg_supported_by_current_training": False,
            "interpretation_warning": (
                "This checkpoint used one joint condition-drop mask. The scan measures "
                "joint CFG sensitivity and must not be described as isolated AoD guidance."
            ),
        },
        "stage_interpretation": {
            "aod_conditioned_b3": "Direct output of the newly trained conditional diffusion.",
            "b3_plus_dp": "The same sample after the frozen DP residual changes Delay and Power.",
            "b3_plus_dp_plus_link_global": (
                "The DP sample after the frozen Link-global residual rotates AoD."
            ),
            "important": (
                "DP can change Mean AoD without changing angle values because Mean AoD is "
                "power weighted. Link-global changes AoD explicitly."
            ),
        },
        "summaries_by_guidance_scale": summaries,
        "cells": cells,
    }
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "aod_cfg_and_stage_audit_complete",
                "output": str(Path(args.output).resolve()),
                "valid_cells": len(valid_cells),
                "stage_names": stage_names,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
