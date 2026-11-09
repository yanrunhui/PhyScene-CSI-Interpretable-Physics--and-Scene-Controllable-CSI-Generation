#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.scene_id import make_scene_id_datasets
from physics_csi.data.delay_power_residual import DelayPowerResidualCacheDataset
from physics_csi.evaluation.scene_support import SceneConditionSupportChecker
from physics_csi.evaluation.strongest_anchor_support import (
    classify_strongest_anchor_requests,
)
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


def _map_id(value: str) -> int:
    normalized = value.strip().lower()
    if normalized.startswith("map_"):
        normalized = normalized[4:]
    try:
        return int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Map ID must look like 10 or map_10") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSI with B3-S-ID.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-id", type=_map_id, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--total-power-db", type=float, required=True)
    parser.add_argument("--delay-spread-ns", type=float, required=True)
    parser.add_argument("--k-factor-db", type=float, default=0.0)
    parser.add_argument("--los-status", type=int, choices=[0, 1], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--path-count", type=int, default=None)
    parser.add_argument(
        "--mean-aod-az-deg",
        type=float,
        default=None,
        help="Continuous power-weighted Mean AoD azimuth target in degrees.",
    )
    parser.add_argument(
        "--strongest-aod-az-deg",
        type=float,
        default=None,
        help="Continuous strongest-power-path AoD azimuth target in degrees.",
    )
    parser.add_argument("--path-count-temperature", type=float, default=None)
    parser.add_argument(
        "--skip-support-check",
        action="store_true",
        help="Skip the empirical warning only; unknown Map IDs still raise an error.",
    )
    parser.add_argument(
        "--aod-support-cache",
        default=None,
        help=(
            "B3 DP cache containing the training reference used for Strongest "
            "Anchor A/B/C support classification."
        ),
    )
    parser.add_argument("--minimum-aod-map-support", type=int, default=8)
    parser.add_argument("--maximum-condition-distance", type=float, default=1.5)
    parser.add_argument("--maximum-angle-distance-deg", type=float, default=10.0)
    parser.add_argument(
        "--known-los-direct-aod-deg",
        type=float,
        default=None,
        help="Optional geometry-derived direct-path AoD used only for class-C conflicts.",
    )
    parser.add_argument("--los-direct-tolerance-deg", type=float, default=10.0)
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scene_kinds = {
        "multipath_diffusion_scene_id",
        "multipath_diffusion_scene_id_spatial",
        "multipath_diffusion_scene_residual",
        "multipath_diffusion_scene_residual_spatial",
        "multipath_diffusion_scene_id_aod_conditioned",
        "multipath_diffusion_scene_id_strongest_aod_conditioned",
        "multipath_diffusion_scene_id_strongest_aod_anchor",
    }
    if checkpoint.get("model_kind") not in scene_kinds:
        raise ValueError("checkpoint is not a supported scene-aware model")
    config = checkpoint["config"]
    plan = config.get("scene_id_experiment")
    if plan is None:
        raise ValueError("checkpoint does not contain the scene split/mapping metadata")
    model = MultipathDiffusionModel(
        config["multipath_diffusion"],
        SystemConfig.from_mapping(config["data"]["system"]),
        checkpoint["statistics"],
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.eval()
    conditions = {
        "total_power_db": args.total_power_db,
        "delay_spread_ns": args.delay_spread_ns,
        "k_factor_db": args.k_factor_db,
        "los_status": args.los_status,
    }
    aod_conditioned = bool(
        config["multipath_diffusion"].get("aod_conditioning", {}).get("enabled", False)
    )
    aod_condition_type = str(
        config["multipath_diffusion"]
        .get("aod_conditioning", {})
        .get("condition_type", "mean")
    )
    if aod_condition_type not in {"mean", "strongest"}:
        raise ValueError(
            f"unsupported AoD condition_type in checkpoint: {aod_condition_type!r}"
        )
    strongest_model_kind = checkpoint.get("model_kind") in {
        "multipath_diffusion_scene_id_strongest_aod_conditioned",
        "multipath_diffusion_scene_id_strongest_aod_anchor",
    }
    if aod_conditioned and (aod_condition_type == "strongest") != strongest_model_kind:
        raise ValueError("checkpoint model_kind and AoD condition_type disagree")
    if args.mean_aod_az_deg is not None and args.strongest_aod_az_deg is not None:
        raise ValueError("Mean-AoD and Strongest-AoD targets are mutually exclusive")
    if aod_conditioned and aod_condition_type == "mean":
        if args.mean_aod_az_deg is None:
            raise ValueError("--mean-aod-az-deg is required by this checkpoint")
        if args.strongest_aod_az_deg is not None:
            raise ValueError("a Mean-AoD checkpoint cannot accept a Strongest-AoD target")
    elif aod_conditioned and aod_condition_type == "strongest":
        if args.strongest_aod_az_deg is None:
            raise ValueError("--strongest-aod-az-deg is required by this checkpoint")
        if args.mean_aod_az_deg is not None:
            raise ValueError("a Strongest-AoD checkpoint cannot accept a Mean-AoD target")
    elif args.mean_aod_az_deg is not None or args.strongest_aod_az_deg is not None:
        raise ValueError("this checkpoint was not trained with an AoD condition")
    trained_path_count = config.get("aod_conditioned_data", {}).get("path_count")
    forced_path_count = args.path_count
    if aod_conditioned and trained_path_count is not None:
        if forced_path_count is None:
            forced_path_count = int(trained_path_count)
        elif forced_path_count != int(trained_path_count):
            raise ValueError(
                f"first-stage AoD model was trained only for L={trained_path_count}"
            )
    support: dict[str, Any] | None = None
    if not args.skip_support_check and not model.strongest_anchor_enabled:
        datasets = make_scene_id_datasets(config["data"]["prepared_root"], plan)
        checker = SceneConditionSupportChecker.from_dataset(
            datasets["train"], checkpoint["statistics"]
        )
        support = checker.check(args.map_id, conditions)
        print(json.dumps({"scene_condition_support": support}, indent=2), flush=True)
    anchor_support: dict[str, Any] | None = None
    if model.strongest_anchor_enabled and not args.skip_support_check:
        if forced_path_count is None:
            raise ValueError(
                "Strongest Anchor support classification requires a fixed "
                "--path-count or aod_conditioned_data.path_count in the checkpoint"
            )
        if args.aod_support_cache is None:
            raise ValueError(
                "Strongest Anchor support classification requires --aod-support-cache "
                "or an explicit --skip-support-check"
            )
        support_cache = torch.load(
            args.aod_support_cache, map_location="cpu", weights_only=False
        )
        support_dataset = DelayPowerResidualCacheDataset(support_cache, "train")
        condition_row = torch.tensor(
            [[
                args.total_power_db,
                args.delay_spread_ns,
                args.k_factor_db,
                float(args.los_status),
            ]],
            dtype=torch.float32,
        )
        condition_validity = torch.ones_like(condition_row, dtype=torch.bool)
        condition_validity[:, 2] = args.los_status == 1
        classification = classify_strongest_anchor_requests(
            support_dataset.reference,
            target_conditions=condition_row,
            target_validity=condition_validity,
            target_map_ids=torch.tensor([args.map_id]),
            target_aod_deg=torch.tensor([args.strongest_aod_az_deg]),
            target_path_count=int(forced_path_count),
            condition_std=torch.as_tensor(
                support_cache["b3_statistics"]["condition_std"]
            ),
            maximum_condition_distance=args.maximum_condition_distance,
            maximum_angle_distance_deg=args.maximum_angle_distance_deg,
            minimum_map_support=args.minimum_aod_map_support,
            known_los_direct_aod_deg=(
                torch.tensor([args.known_los_direct_aod_deg])
                if args.known_los_direct_aod_deg is not None
                else None
            ),
            los_direct_tolerance_deg=args.los_direct_tolerance_deg,
        )
        anchor_support = {
            "class": classification.labels[0],
            "map_support_count": int(classification.map_support_count[0]),
            "cross_scene_support_count": int(
                classification.cross_scene_support_count[0]
            ),
            "physically_feasible": bool(classification.physically_feasible[0]),
            "conflict_reasons": list(classification.conflict_reasons[0]),
            "policy": (
                "A: strict Map-conditioned fidelity; B: counterfactual physical and "
                "cross-scene checks; C: reject before generation."
            ),
        }
        print(json.dumps({"strongest_anchor_support": anchor_support}, indent=2))
        if classification.labels[0] == "C":
            conflict_path = Path(args.output).with_suffix(".conflict.json")
            atomic_json_dump(
                {
                    "format_version": 1,
                    "status": "rejected_condition_conflict",
                    "map_id": args.map_id,
                    "conditions": conditions,
                    "target_strongest_aod_az_deg": args.strongest_aod_az_deg,
                    "strongest_anchor_support": anchor_support,
                    "message": "Class-C requests are reported and are not sampled.",
                },
                conflict_path,
            )
            raise ValueError(
                "Strongest Anchor request is class C: "
                + ", ".join(classification.conflict_reasons[0])
                + f"; report={conflict_path}"
            )
        datasets = make_scene_id_datasets(config["data"]["prepared_root"], plan)
        checker = SceneConditionSupportChecker.from_dataset(
            datasets["train"], checkpoint["statistics"]
        )
        support = checker.check(args.map_id, conditions)
        print(json.dumps({"scene_condition_support": support}, indent=2), flush=True)
    # Resolve once before expensive sampling. For Anchor requests this happens
    # after classification so an unseen Map is reported as a class-C conflict.
    model.map_indices(args.map_id, 1, device)
    sampling = config["sampling"]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    started = time.perf_counter()
    generated = model.generate(
        conditions,
        args.num_samples,
        map_id=args.map_id,
        sampler=args.sampler or sampling["sampler"],
        sampling_steps=(
            args.sampling_steps
            if args.sampling_steps is not None
            else int(sampling["sampling_steps"])
        ),
        eta=float(sampling.get("eta", 0.0)),
        guidance_scale=(
            args.guidance_scale
            if args.guidance_scale is not None
            else float(sampling["guidance_scale"])
        ),
        path_count_temperature=(
            args.path_count_temperature
            if args.path_count_temperature is not None
            else float(sampling.get("path_count_temperature", 1.0))
        ),
        forced_path_count=forced_path_count,
        mean_aod_az_deg=args.mean_aod_az_deg,
        strongest_aod_az_deg=args.strongest_aod_az_deg,
        generator=generator,
    )
    elapsed = time.perf_counter() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_kind": checkpoint["model_kind"],
            "generated": _cpu(generated),
            "conditions": conditions,
            "aod_condition_type": aod_condition_type if aod_conditioned else None,
            "mean_aod_az_deg": args.mean_aod_az_deg,
            "strongest_aod_az_deg": args.strongest_aod_az_deg,
            "forced_path_count": forced_path_count,
            "map_id": args.map_id,
            "map_name": f"map_{args.map_id}",
            "scene_condition_support": support,
            "strongest_anchor_support": anchor_support,
            "seed": args.seed,
            "generation_metadata": {
                "seconds": elapsed,
                "samples_per_second": args.num_samples / max(elapsed, 1e-12),
                "sampler": args.sampler or sampling["sampler"],
                "sampling_steps": args.sampling_steps or int(sampling["sampling_steps"]),
                "guidance_scale": (
                    args.guidance_scale
                    if args.guidance_scale is not None
                    else float(sampling["guidance_scale"])
                ),
                "forced_path_count": forced_path_count,
            },
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "scene_plan_fingerprint": plan["fingerprint"],
        },
        output,
    )


if __name__ == "__main__":
    main()
