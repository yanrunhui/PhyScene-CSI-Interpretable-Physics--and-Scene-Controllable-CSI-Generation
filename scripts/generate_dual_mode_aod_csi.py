#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.generation_mode import AoDControlMode, resolve_generation_route
from physics_csi.evaluation.aod_support import (
    circular_distance_degrees,
    conditioned_aod_measurement,
    enforce_final_strongest_anchor,
    normalized_condition_features,
    strongest_aod,
)
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.physics.aod_control import (
    AoDControlProjector,
    select_payload_samples,
    select_random_aod_candidates_within_tolerance,
    wrap_degrees,
)
from physics_csi.physics.constraints import check_condition_feasibility
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device
from physics_csi.utils.io import atomic_json_dump


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


def _append_payload(
    chunks: dict[str, list[torch.Tensor]], payload: Mapping[str, torch.Tensor]
) -> None:
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            chunks.setdefault(key, []).append(value.detach().cpu())


def _finish_payload(
    chunks: Mapping[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def _to_device(
    payload: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in payload.items()}


def _map_id(value: str) -> int:
    normalized = value.strip().lower()
    if normalized.startswith("map_"):
        normalized = normalized[4:]
    try:
        return int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Map ID must look like 10 or map_10") from error


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


def _refresh_strongest_aod_fields(
    payload: Mapping[str, torch.Tensor],
    target_az_deg: float | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Recompute every Strongest-AoD derivative after a power-changing stage."""
    result = {
        key: value for key, value in payload.items() if isinstance(value, torch.Tensor)
    }
    measured = strongest_aod(result)
    target = torch.as_tensor(
        target_az_deg,
        device=measured.angle_deg.device,
        dtype=measured.angle_deg.dtype,
    ).reshape(-1)
    if target.numel() == 1:
        target = target.expand(measured.angle_deg.shape[0])
    if target.shape != measured.angle_deg.shape:
        raise ValueError("Strongest-AoD target must be scalar or match payload size")
    target = wrap_degrees(target)
    result.update(
        target_strongest_aod_az_deg=target,
        target_strongest_aod_validity=torch.ones_like(target, dtype=torch.bool),
        actual_strongest_aod_az_deg=measured.angle_deg,
        actual_strongest_path_index=measured.path_index,
        actual_strongest_aod_valid=measured.valid,
        strongest_top2_power_gap_db=measured.top2_power_gap_db,
        strongest_near_tie=measured.near_tie,
        strongest_aod_circular_error_deg=circular_distance_degrees(
            measured.angle_deg, target
        ),
    )
    return result


def _audit(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def _load_dp_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[SceneConditionedDelayPowerResidual, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("--dp-checkpoint is not a DP residual")
    model = SceneConditionedDelayPowerResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(
        checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
    )
    return model.requires_grad_(False).eval(), checkpoint


def _load_aod_conditioned_model(
    checkpoint_path: Path,
    dp: SceneConditionedDelayPowerResidual,
    device: torch.device,
    expected_condition_type: str,
) -> tuple[MultipathDiffusionModel, Mapping[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_kinds = {
        "mean": {"multipath_diffusion_scene_id_aod_conditioned"},
        "strongest": {"multipath_diffusion_scene_id_strongest_aod_anchor"},
    }[expected_condition_type]
    if checkpoint.get("model_kind") not in expected_kinds:
        raise ValueError("--aod-checkpoint is not an AoD-conditioned B3 checkpoint")
    config = checkpoint["config"]
    aod_config = config["multipath_diffusion"].get("aod_conditioning", {})
    if not aod_config.get("enabled", False):
        raise ValueError("--aod-checkpoint does not enable AoD conditioning")
    condition_type = str(aod_config.get("condition_type", "mean"))
    if condition_type != expected_condition_type:
        raise ValueError(
            "--aod-checkpoint condition type does not match the requested AoD target"
        )
    anchor_enabled = bool(
        aod_config.get("anchor_parameterization", {}).get("enabled", False)
    )
    if expected_condition_type == "strongest" and not anchor_enabled:
        raise ValueError("Strongest production mode requires a Strongest Anchor checkpoint")
    trained_path_count = config.get("aod_conditioned_data", {}).get("path_count")
    if trained_path_count is None:
        raise ValueError("AoD checkpoint does not record its trained path count")
    system = SystemConfig.from_mapping(config["data"]["system"])
    model = MultipathDiffusionModel(
        config["multipath_diffusion"], system, checkpoint["statistics"]
    ).to(device)
    model.load_state_dict(
        checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
    )
    model.requires_grad_(False).eval()
    if model.map_id_to_index != dp.map_id_to_index:
        raise ValueError("AoD-conditioned B3 and DP residual use different Map mappings")
    if model.max_paths != dp.max_paths:
        raise ValueError("AoD-conditioned B3 and DP residual use different max_paths")
    return model, checkpoint, int(trained_path_count)


def _support_audit(
    cache_path: str | None,
    *,
    map_id: int,
    conditions: torch.Tensor,
    validity: torch.Tensor,
    path_count: int,
    target_aod_deg: float,
    condition_type: str,
) -> dict[str, Any] | None:
    if cache_path is None:
        return None
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if "train" not in cache["splits"]:
        return {"status": "unavailable", "reason": "cache has no train split"}
    reference = cache["splits"]["train"]["reference"]
    values = reference["conditions"].float()
    valid = reference["condition_validity"].bool()
    count = reference["path_mask"].bool().sum(-1)
    maps = reference["map_id"].long()
    exact = (
        (maps == map_id)
        & (count == path_count)
        & ((values[:, 3] >= 0.5) == (conditions[0, 3] >= 0.5))
        & (valid[:, 2] == validity[0, 2])
    )
    condition_std = torch.as_tensor(
        cache.get("b3_statistics", {}).get("condition_std", []),
        dtype=torch.float32,
    )
    if condition_std.numel() != 4:
        condition_std = values.std(0, unbiased=False).clamp_min(1e-6)
    real_features = normalized_condition_features(values, valid, condition_std)
    request_features = normalized_condition_features(
        conditions[:1].float(), validity[:1].bool(), condition_std
    )
    condition_distance = torch.linalg.vector_norm(
        real_features - request_features[0], dim=-1
    )
    real_angle, real_angle_valid, _ = conditioned_aod_measurement(
        reference, condition_type=condition_type
    )
    angle_distance = circular_distance_degrees(
        real_angle, torch.tensor(float(target_aod_deg))
    )
    joint = (
        exact
        & real_angle_valid
        & (condition_distance <= 1.5)
        & (angle_distance <= 10.0)
    )
    cross_scene = (
        (maps != map_id)
        & (count == path_count)
        & ((values[:, 3] >= 0.5) == (conditions[0, 3] >= 0.5))
        & (valid[:, 2] == validity[0, 2])
        & real_angle_valid
        & (condition_distance <= 1.5)
        & (angle_distance <= 10.0)
    )
    support_count = int(joint.sum())
    structural_count = int(exact.sum())
    return {
        "support_class": "A" if support_count >= 8 else "B",
        "interpretation": (
            "A: Map-conditioned real support; B: low Map support but the declared "
            "aggregate physical conditions are feasible. Known link geometry is not "
            "available here, so this audit does not declare class C conflicts."
        ),
        "aod_semantics": condition_type,
        "target_aod_deg": float(target_aod_deg),
        "map_joint_support_count": support_count,
        "cross_scene_joint_support_count": int(cross_scene.sum()),
        "minimum_support_for_class_a": 8,
        "maximum_standardized_condition_distance": 1.5,
        "maximum_circular_angle_distance_deg": 10.0,
        "exact_structural_candidates": structural_count,
        "map_id": map_id,
        "path_count": path_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CSI through one of two frozen AoD semantics: power-weighted "
            "Mean AoD or Strongest-path AoD."
        )
    )
    parser.add_argument("--mean-aod-checkpoint")
    parser.add_argument("--strongest-aod-checkpoint")
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--support-cache")
    parser.add_argument("--map-id", type=_map_id, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--total-power-db", type=float, required=True)
    parser.add_argument("--delay-spread-ns", type=float, required=True)
    parser.add_argument("--k-factor-db", type=float, default=0.0)
    parser.add_argument("--los-status", type=int, choices=[0, 1], required=True)
    parser.add_argument("--path-count", type=int, required=True)
    parser.add_argument(
        "--aod-mode",
        required=True,
        choices=[mode.value for mode in AoDControlMode],
    )
    parser.add_argument("--target-aod-deg", type=float, required=True)
    parser.add_argument("--natural-angle-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--minimum-mean-concentration", type=float, default=0.10)
    parser.add_argument("--candidate-generation-batch-size", type=int, default=512)
    parser.add_argument(
        "--selection-seed",
        type=int,
        help="Independent seed for uniform sampling from all eligible candidates.",
    )
    parser.add_argument(
        "--k-factor-validity",
        choices=["auto", "valid", "invalid"],
        default="auto",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"])
    parser.add_argument("--sampling-steps", type=int)
    args = parser.parse_args()

    if args.num_samples != 8:
        parser.error("the frozen production contract outputs exactly 8 CSI samples")
    if args.candidate_generation_batch_size < 1:
        parser.error("candidate-generation-batch-size must be positive")
    if not 0.0 <= args.natural_angle_tolerance_deg <= 180.0:
        parser.error("natural-angle-tolerance-deg must lie in [0, 180]")
    if args.natural_angle_tolerance_deg != 10.0:
        parser.error("the frozen production tolerance is exactly 10 degrees")

    device = resolve_device(args.device)
    dp, _ = _load_dp_model(Path(args.dp_checkpoint), device)
    requested_mode = AoDControlMode(args.aod_mode)
    requested_condition_type = (
        "mean"
        if requested_mode is AoDControlMode.MEAN_POWER_WEIGHTED
        else "strongest"
    )
    checkpoint_argument = (
        args.mean_aod_checkpoint
        if requested_mode is AoDControlMode.MEAN_POWER_WEIGHTED
        else args.strongest_aod_checkpoint
    )
    if not checkpoint_argument:
        parser.error(f"{args.aod_mode} requires its corresponding checkpoint")
    aod_model, checkpoint, trained_aod_path_count = _load_aod_conditioned_model(
        Path(checkpoint_argument), dp, device, requested_condition_type
    )

    try:
        route = resolve_generation_route(
            aod_mode=requested_mode,
            target_aod_deg=args.target_aod_deg,
            path_count=args.path_count,
            aod_trained_path_count=trained_aod_path_count,
        )
    except ValueError as error:
        parser.error(str(error))

    active_model = aod_model
    if not 1 <= args.path_count <= active_model.max_paths:
        parser.error(f"path-count must lie in [1, {active_model.max_paths}]")

    k_valid = (
        bool(args.los_status and args.path_count >= 2)
        if args.k_factor_validity == "auto"
        else args.k_factor_validity == "valid"
    )
    condition_dict = {
        "total_power_db": args.total_power_db,
        "delay_spread_ns": args.delay_spread_ns,
        "k_factor_db": args.k_factor_db,
        "los_status": args.los_status,
    }
    check_condition_feasibility(
        condition_dict, path_count=args.path_count, k_factor_valid=k_valid
    )

    apply_filter = route.apply_natural_angle_filter
    maximum_candidate_samples = args.num_samples * route.candidate_multiplier
    condition_row = torch.tensor(
        [args.total_power_db, args.delay_spread_ns, args.k_factor_db, args.los_status],
        dtype=torch.float32,
    )
    sampling = checkpoint["config"]["sampling"]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    base_chunks: dict[str, list[torch.Tensor]] = {}
    dp_chunks: dict[str, list[torch.Tensor]] = {}
    final_chunks: dict[str, list[torch.Tensor]] = {}
    candidate_samples = 0
    eligible_candidate_samples = 0
    generation_round_sizes: list[int] = []
    projector = AoDControlProjector(
        minimum_mean_concentration=args.minimum_mean_concentration
    )
    started = time.perf_counter()

    with torch.inference_mode():
        while candidate_samples < maximum_candidate_samples:
            round_size = maximum_candidate_samples - candidate_samples
            generation_round_sizes.append(round_size)
            round_complete = 0
            while round_complete < round_size:
                batch_size = min(
                    args.candidate_generation_batch_size,
                    round_size - round_complete,
                )
                conditions = condition_row.to(device).expand(batch_size, -1)
                validity = torch.ones_like(conditions, dtype=torch.bool)
                validity[:, 2] = k_valid
                map_ids = torch.full(
                    (batch_size,), args.map_id, device=device, dtype=torch.long
                )
                map_index = dp.map_indices(map_ids)

                base_batch = aod_model.generate(
                    conditions,
                    batch_size,
                    condition_validity=validity,
                    sampler=args.sampler or str(sampling["sampler"]),
                    sampling_steps=args.sampling_steps
                    or int(sampling["sampling_steps"]),
                    eta=float(sampling.get("eta", 0.0)),
                    guidance_scale=float(route.guidance_scale),
                    forced_path_count=args.path_count,
                    mean_aod_az_deg=route.target_mean_aod_az_deg,
                    strongest_aod_az_deg=route.target_strongest_aod_az_deg,
                    map_id=map_ids,
                    generator=generator,
                )

                dp_output = dp(base_batch, conditions, validity, map_index=map_index)
                dp_batch = dp.apply_to_payload(base_batch, dp_output)
                if route.aod_condition_type == "strongest":
                    assert route.target_strongest_aod_az_deg is not None
                    target = torch.full(
                        (batch_size,),
                        route.target_strongest_aod_az_deg,
                        device=device,
                    )
                    dp_batch = enforce_final_strongest_anchor(
                        dp_batch,
                        target,
                        strategy=route.strongest_rebind_strategy or "swap",
                    )
                    dp_batch = _refresh_strongest_aod_fields(
                        dp_batch, route.target_strongest_aod_az_deg
                    )
                final_batch = dp_batch

                if apply_filter:
                    measurement = projector(
                        final_batch,
                        route.target_aod_az_deg,
                        mode=route.aod_condition_type,
                    )
                    eligible_candidate_samples += int(
                        (
                            measurement.control_valid
                            & (
                                measurement.rotation_deg.abs()
                                <= args.natural_angle_tolerance_deg
                            )
                        ).sum()
                    )

                _append_payload(base_chunks, base_batch)
                _append_payload(dp_chunks, dp_batch)
                _append_payload(final_chunks, final_batch)
                round_complete += batch_size
                candidate_samples += batch_size
                print(
                    json.dumps(
                        {
                            "event": "dual_route_generation_progress",
                            "generation_mode": route.mode.value,
                            "generated_candidates": candidate_samples,
                            "eligible_candidates": eligible_candidate_samples,
                            "requested_outputs": args.num_samples,
                            "maximum_candidate_budget": maximum_candidate_samples,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if apply_filter and eligible_candidate_samples < args.num_samples:
            report_path = (
                Path(args.report)
                if args.report
                else Path(args.output).with_suffix(".json")
            )
            failure_report = {
                "format_version": 4,
                "experiment": "dual_aod_semantics_csi_generation",
                "status": "insufficient_eligible_candidates",
                "generation_mode": route.mode.value,
                "guidance_scale": route.guidance_scale,
                "conditions": condition_dict,
                "condition_validity": [True, True, k_valid, True],
                "requested_samples": args.num_samples,
                "candidate_samples_generated": candidate_samples,
                "eligible_samples": eligible_candidate_samples,
                "candidate_generation_round_sizes": generation_round_sizes,
                "maximum_candidate_budget": maximum_candidate_samples,
                "natural_angle_tolerance_deg": args.natural_angle_tolerance_deg,
                "tolerance_was_relaxed": False,
                "selection_policy": "uniform_random_without_replacement",
                "map_id": args.map_id,
                "path_count": args.path_count,
                "target_mean_aod_az_deg": route.target_mean_aod_az_deg,
                "target_strongest_aod_az_deg": route.target_strongest_aod_az_deg,
                "message": (
                    "The fixed 16-candidate Mean-AoD budget did not contain eight "
                    "naturally eligible CSI; no output artifact was written."
                ),
            }
            atomic_json_dump(failure_report, report_path)
            print(json.dumps(failure_report, indent=2, sort_keys=True), flush=True)
            raise RuntimeError(
                "fixed Mean-AoD 16-to-8 policy has insufficient eligible candidates"
            )

        base = _finish_payload(base_chunks)
        dp_payload = _finish_payload(dp_chunks)
        final_payload = _finish_payload(final_chunks)
        selection_report: dict[str, Any] = {
            "policy": "none",
            "candidate_samples": candidate_samples,
            "selected_samples": args.num_samples,
        }
        if apply_filter:
            selection_seed = (
                args.selection_seed
                if args.selection_seed is not None
                else args.seed + 1_000_003
            )
            selection = select_random_aod_candidates_within_tolerance(
                final_payload,
                route.target_aod_az_deg,
                samples=args.num_samples,
                tolerance_deg=args.natural_angle_tolerance_deg,
                mode=route.aod_condition_type,
                projector=projector,
                generator=torch.Generator().manual_seed(selection_seed),
            )
            base = select_payload_samples(base, selection.indices)
            dp_payload = select_payload_samples(dp_payload, selection.indices)
            final_payload = select_payload_samples(final_payload, selection.indices)
            selection_report = {
                "policy": (
                    f"uniform_random_among_natural_{route.aod_condition_type}_aod_eligible"
                ),
                "candidate_samples": selection.candidate_samples,
                "eligible_samples": selection.eligible_samples,
                "acceptance_rate": selection.eligible_samples
                / selection.candidate_samples,
                "selected_samples": selection.requested_samples,
                "unique_selected_samples": int(torch.unique(selection.indices).numel()),
                "candidate_multiplier": route.candidate_multiplier,
                "maximum_candidate_budget": maximum_candidate_samples,
                "candidate_generation_round_sizes": generation_round_sizes,
                "natural_angle_tolerance_deg": args.natural_angle_tolerance_deg,
                "tolerance_was_relaxed": False,
                "selection_seed": selection_seed,
                "selected_angle_error_deg": _audit(selection.angular_distance_deg),
                "selected_candidate_indices": selection.indices.cpu().tolist(),
            }
        else:
            selection_report = {
                "policy": "direct_8_after_dp_swap_and_top1_verification",
                "candidate_samples": candidate_samples,
                "selected_samples": args.num_samples,
                "candidate_multiplier": route.candidate_multiplier,
                "strongest_rebind_strategy": route.strongest_rebind_strategy,
                "natural_angle_tolerance_deg": args.natural_angle_tolerance_deg,
            }

        synthesizer = MIMOOFDMSynthesizer(active_model.synthesizer.config).to(device).eval()
        dp_device = _to_device(dp_payload, device)
        dp_device["csi"] = _synthesize(dp_device, synthesizer)
        dp_payload = _cpu(dp_device)
        final_device = _to_device(final_payload, device)
        final_device["csi"] = _synthesize(final_device, synthesizer)
        final_payload = _cpu(final_device)

        control_report: dict[str, Any] | None = None
        control_tensors: dict[str, torch.Tensor] | None = None
        measurement = AoDControlProjector(
            minimum_mean_concentration=args.minimum_mean_concentration
        )(
            final_device,
            route.target_aod_az_deg,
            mode=route.aod_condition_type,
        )
        error = measurement.rotation_deg.abs()
        control_report = {
                "target_mean_aod_az_deg": route.target_mean_aod_az_deg,
                "target_strongest_aod_az_deg": route.target_strongest_aod_az_deg,
                "aod_mode": route.aod_mode.value,
                "condition_type": route.aod_condition_type,
                "projection_applied": False,
                "aod_error_deg": _audit(error),
                "hit_at_5_deg": float((error <= 5.0).float().mean()),
                "hit_at_10_deg": float((error <= 10.0).float().mean()),
                "valid_rate": float(measurement.control_valid.float().mean()),
        }
        if route.aod_condition_type == "mean":
            control_report["mean_aod_error_deg"] = control_report["aod_error_deg"]
            control_report["mean_resultant_concentration"] = _audit(
                measurement.mean_resultant_concentration
            )
            control_tensors = {
                "target_mean_aod_az_deg": measurement.target_az_deg,
                "actual_mean_aod_az_deg": measurement.reference_before_deg,
                "mean_aod_circular_error_deg": error,
                "mean_aod_valid": measurement.control_valid,
                "mean_aod_concentration": measurement.mean_resultant_concentration,
            }
        else:
            strongest = strongest_aod(final_device)
            top1_pass = measurement.control_valid & (
                error <= args.natural_angle_tolerance_deg
            )
            control_report.update(
                strongest_aod_error_deg=control_report["aod_error_deg"],
                strongest_top2_power_gap_db=_audit(strongest.top2_power_gap_db),
                strongest_near_tie_rate=float(strongest.near_tie.float().mean()),
                top1_verification_pass=bool(top1_pass.all()),
            )
            selection_report["top1_verified_samples"] = int(top1_pass.sum())
            selection_report["top1_verification_pass"] = bool(top1_pass.all())
            control_tensors = {
                "target_strongest_aod_az_deg": measurement.target_az_deg,
                "actual_strongest_aod_az_deg": strongest.angle_deg,
                "actual_strongest_path_index": strongest.path_index,
                "strongest_aod_circular_error_deg": error,
                "strongest_aod_valid": strongest.valid,
                "strongest_top2_power_gap_db": strongest.top2_power_gap_db,
                "strongest_near_tie": strongest.near_tie,
            }
            if not bool(top1_pass.all()):
                report_path = (
                    Path(args.report)
                    if args.report
                    else Path(args.output).with_suffix(".json")
                )
                failure_report = {
                    "format_version": 4,
                    "experiment": "dual_aod_semantics_csi_generation",
                    "status": "strongest_top1_verification_failed",
                    "aod_mode": route.aod_mode.value,
                    "control": control_report,
                    "selection": selection_report,
                    "message": "No output artifact was written.",
                }
                atomic_json_dump(failure_report, report_path)
                raise RuntimeError("Strongest-path Top1 verification failed")

    elapsed = time.perf_counter() - started
    output_conditions = condition_row.expand(args.num_samples, -1).clone()
    output_validity = torch.ones_like(output_conditions, dtype=torch.bool)
    output_validity[:, 2] = k_valid
    support = _support_audit(
        args.support_cache,
        map_id=args.map_id,
        conditions=output_conditions,
        validity=output_validity,
        path_count=args.path_count,
        target_aod_deg=route.target_aod_az_deg,
        condition_type=route.aod_condition_type,
    )
    pipeline = [
        f"{route.aod_mode.value} B3 (CFG={route.guidance_scale:g})",
        "DP Residual",
        (
            "fixed 16-to-8 natural Mean-AoD tolerance filter"
            if apply_filter
            else "AoD Swap on DP Top1 change + final Top1 verification"
        ),
        "physics check",
        "CSI synthesis",
    ]
    checkpoint_paths = {
        "dp": str(Path(args.dp_checkpoint).resolve()),
        "mean_aod_b3": (
            str(Path(args.mean_aod_checkpoint).resolve())
            if args.mean_aod_checkpoint
            else None
        ),
        "strongest_anchor_b3": (
            str(Path(args.strongest_aod_checkpoint).resolve())
            if args.strongest_aod_checkpoint
            else None
        ),
        "selected_aod_b3": str(Path(checkpoint_argument).resolve()),
        "link_global": None,
    }
    artifact = {
        "format_version": 4,
        "model_kind": "dual_route_physics_controllable_csi_generator",
        "generation_mode": route.mode.value,
        "aod_mode": route.aod_mode.value,
        "pipeline": pipeline,
        "conditions": condition_dict,
        "condition_validity": output_validity[0],
        "map_id": args.map_id,
        "path_count": args.path_count,
        "target_mean_aod_az_deg": route.target_mean_aod_az_deg,
        "target_strongest_aod_az_deg": route.target_strongest_aod_az_deg,
        "seed": args.seed,
        "base": _cpu(base),
        "dp_only": _cpu(dp_payload),
        "generated": _cpu(final_payload),
        "control_audit": _cpu(control_tensors),
        "selection_audit": selection_report,
        "aod_condition_support": support,
        "checkpoint_paths": checkpoint_paths,
        "system_config": active_model.synthesizer.config.as_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(output)

    report = {
        "format_version": 4,
        "experiment": "dual_aod_semantics_csi_generation",
        "status": "success",
        "artifact": str(output.resolve()),
        "generation_mode": route.mode.value,
        "aod_mode": route.aod_mode.value,
        "pipeline": pipeline,
        "samples": args.num_samples,
        "candidate_samples": candidate_samples,
        "seconds": elapsed,
        "samples_per_second": args.num_samples / max(elapsed, 1e-20),
        "conditions": condition_dict,
        "condition_validity": output_validity[0].tolist(),
        "map_id": args.map_id,
        "path_count": args.path_count,
        "selection": selection_report,
        "control": control_report,
        "aod_condition_support": support,
        "checkpoint_paths": checkpoint_paths,
        "limitations": [
            "Both AoD modes are currently trained and validated only for L=5.",
            "Mean mode uses a fixed 16-candidate budget and never relaxes ±10 degrees.",
            "Strongest mode requires the Anchor checkpoint and AoD Swap after DP.",
            "Class-B requests are counterfactual and do not guarantee the Map's typical distribution.",
        ],
    }
    report_path = Path(args.report) if args.report else output.with_suffix(".json")
    atomic_json_dump(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
