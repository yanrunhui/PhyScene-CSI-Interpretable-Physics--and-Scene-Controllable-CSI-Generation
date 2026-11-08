#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.generation_mode import CandidateBudgetPolicy, resolve_generation_route
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.models.scene_link_global_aod_residual import (
    SceneConditionedLinkGlobalAoDResidual,
)
from physics_csi.physics.aod_control import (
    AoDControlProjector,
    select_payload_samples,
    select_random_aod_candidates_within_tolerance,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _audit(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def _load_models(
    b3_path: Path,
    dp_path: Path,
    link_path: Path,
    device: torch.device,
) -> tuple[
    MultipathDiffusionModel,
    SceneConditionedDelayPowerResidual,
    SceneConditionedLinkGlobalAoDResidual,
    Mapping[str, Any],
]:
    b3_checkpoint = torch.load(b3_path, map_location="cpu", weights_only=False)
    dp_checkpoint = torch.load(dp_path, map_location="cpu", weights_only=False)
    link_checkpoint = torch.load(link_path, map_location="cpu", weights_only=False)
    if dp_checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("--dp-checkpoint is not a DP residual")
    if link_checkpoint.get("model_kind") != "scene_link_global_aod_residual":
        raise ValueError("--link-checkpoint is not a Link-global AoD residual")
    fingerprint = _sha256(b3_path)
    for checkpoint in (dp_checkpoint, link_checkpoint):
        if checkpoint["cache_contract"]["b3_checkpoint_sha256"] != fingerprint:
            raise ValueError("residual checkpoint was trained with a different B3 backbone")
    config = b3_checkpoint["config"]
    system = SystemConfig.from_mapping(config["data"]["system"])
    b3 = MultipathDiffusionModel(
        config["multipath_diffusion"], system, b3_checkpoint["statistics"]
    ).to(device)
    if b3.scene_aware_enabled:
        raise ValueError("dual-mode pipeline requires the Map-independent B3 backbone")
    b3.load_state_dict(
        b3_checkpoint.get("ema_state_dict", b3_checkpoint["model_state_dict"])
    )
    b3.requires_grad_(False).eval()
    dp = SceneConditionedDelayPowerResidual(**dp_checkpoint["model_init"]).to(device)
    dp.load_state_dict(dp_checkpoint.get("ema_state_dict", dp_checkpoint["model_state_dict"]))
    dp.requires_grad_(False).eval()
    link = SceneConditionedLinkGlobalAoDResidual(**link_checkpoint["model_init"]).to(device)
    link.load_state_dict(
        link_checkpoint.get("ema_state_dict", link_checkpoint["model_state_dict"])
    )
    link.requires_grad_(False).eval()
    if dp.map_id_to_index != link.map_id_to_index:
        raise ValueError("DP and Link-global checkpoints use different Map mappings")
    return b3, dp, link, b3_checkpoint


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
) -> tuple[MultipathDiffusionModel, Mapping[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "multipath_diffusion_scene_id_aod_conditioned":
        raise ValueError("--aod-checkpoint is not an AoD-conditioned B3 checkpoint")
    config = checkpoint["config"]
    if not config["multipath_diffusion"].get("aod_conditioning", {}).get(
        "enabled", False
    ):
        raise ValueError("--aod-checkpoint does not enable Mean-AoD conditioning")
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
    candidates = torch.nonzero(exact, as_tuple=False).flatten()
    if not candidates.numel():
        return {
            "status": "out_of_support",
            "exact_structural_candidates": 0,
            "map_id": map_id,
            "path_count": path_count,
        }
    scale = values.std(0, unbiased=False).clamp_min(1e-6)
    dimensions = torch.tensor([0, 1, 2]) if bool(validity[0, 2]) else torch.tensor([0, 1])
    distance = (
        (values[candidates][:, dimensions] - conditions[0, dimensions])
        / scale[dimensions]
    ).square().sum(-1).sqrt()
    nearest = float(distance.min())
    within_one = int((distance <= 1.0).sum())
    within_two = int((distance <= 2.0).sum())
    status = "supported" if within_one >= 12 else "low_support" if within_two >= 12 else "out_of_support"
    return {
        "status": status,
        "exact_structural_candidates": int(candidates.numel()),
        "standardized_condition_distance_minimum": nearest,
        "neighbors_within_1": within_one,
        "neighbors_within_2": within_two,
        "map_id": map_id,
        "path_count": path_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CSI through one of two frozen routes: B3+DP+Link-global "
            "without a Mean-AoD request, or AoD-B3(CFG=2)+DP with bounded "
            "natural-angle candidate filtering when Mean AoD is set."
        )
    )
    parser.add_argument("--b3-checkpoint")
    parser.add_argument("--aod-checkpoint")
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--link-checkpoint")
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
        "--target-aod-az-deg",
        type=float,
        help=(
            "Power-weighted Mean AoD target. Omitting it selects five-condition mode; "
            "supplying it selects the AoD-conditioned six-condition route."
        ),
    )
    parser.add_argument("--aod-guidance-scale", type=float, default=2.0)
    parser.add_argument(
        "--disable-aod-filter",
        action="store_true",
        help="Return direct AoD-B3 samples without the default natural ±tolerance filter.",
    )
    parser.add_argument("--natural-angle-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--minimum-mean-concentration", type=float, default=0.10)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument(
        "--candidate-top-up-size",
        type=int,
        default=8,
        help="Additional candidates generated per retry when fewer than requested pass.",
    )
    parser.add_argument(
        "--maximum-candidate-multiplier",
        type=int,
        default=8,
        help="Hard candidate budget relative to the requested output count.",
    )
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
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="Optional guidance override for five-condition B3 only.",
    )
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("num-samples must be positive")
    if args.candidate_multiplier < 1:
        parser.error("candidate-multiplier must be positive")
    if args.candidate_top_up_size < 1:
        parser.error("candidate-top-up-size must be positive")
    if args.maximum_candidate_multiplier < args.candidate_multiplier:
        parser.error(
            "maximum-candidate-multiplier cannot be smaller than candidate-multiplier"
        )
    if args.candidate_generation_batch_size < 1:
        parser.error("candidate-generation-batch-size must be positive")
    if not 0.0 <= args.natural_angle_tolerance_deg <= 180.0:
        parser.error("natural-angle-tolerance-deg must lie in [0, 180]")

    device = resolve_device(args.device)
    dp, _ = _load_dp_model(Path(args.dp_checkpoint), device)
    aod_model: MultipathDiffusionModel | None = None
    natural_b3: MultipathDiffusionModel | None = None
    link: SceneConditionedLinkGlobalAoDResidual | None = None
    checkpoint: Mapping[str, Any]
    trained_aod_path_count = 5

    if args.target_aod_az_deg is None:
        if not args.b3_checkpoint or not args.link_checkpoint:
            parser.error(
                "five-condition mode requires --b3-checkpoint and --link-checkpoint"
            )
        natural_b3, dp, link, checkpoint = _load_models(
            Path(args.b3_checkpoint),
            Path(args.dp_checkpoint),
            Path(args.link_checkpoint),
            device,
        )
    else:
        if not args.aod_checkpoint:
            parser.error("six-condition Mean-AoD mode requires --aod-checkpoint")
        aod_model, checkpoint, trained_aod_path_count = _load_aod_conditioned_model(
            Path(args.aod_checkpoint), dp, device
        )

    try:
        route = resolve_generation_route(
            target_mean_aod_az_deg=args.target_aod_az_deg,
            path_count=args.path_count,
            aod_trained_path_count=trained_aod_path_count,
            aod_guidance_scale=args.aod_guidance_scale,
        )
    except ValueError as error:
        parser.error(str(error))

    active_model = natural_b3 if route.use_natural_b3 else aod_model
    if active_model is None:
        raise RuntimeError("generation route did not load its required backbone")
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

    apply_filter = route.use_aod_conditioned_b3 and not args.disable_aod_filter
    candidate_policy = CandidateBudgetPolicy(
        requested_outputs=args.num_samples,
        initial_multiplier=args.candidate_multiplier,
        top_up_candidates=args.candidate_top_up_size,
        maximum_multiplier=args.maximum_candidate_multiplier,
    )
    maximum_candidate_samples = (
        candidate_policy.maximum_candidates if apply_filter else args.num_samples
    )
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
            if apply_filter and eligible_candidate_samples >= args.num_samples:
                break
            round_size = (
                candidate_policy.next_round_size(candidate_samples)
                if apply_filter
                else args.num_samples
            )
            if round_size == 0:
                break
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

                if route.use_natural_b3:
                    assert natural_b3 is not None and link is not None
                    if not torch.equal(map_index, link.map_indices(map_ids)):
                        raise RuntimeError(
                            "DP and Link-global resolved the Map ID differently"
                        )
                    base_batch = natural_b3.generate(
                        conditions,
                        batch_size,
                        condition_validity=validity,
                        sampler=args.sampler or str(sampling["sampler"]),
                        sampling_steps=args.sampling_steps
                        or int(sampling["sampling_steps"]),
                        eta=float(sampling.get("eta", 0.0)),
                        guidance_scale=(
                            args.guidance_scale
                            if args.guidance_scale is not None
                            else float(sampling["guidance_scale"])
                        ),
                        forced_path_count=args.path_count,
                        generator=generator,
                    )
                else:
                    assert (
                        aod_model is not None
                        and route.target_mean_aod_az_deg is not None
                    )
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
                        map_id=map_ids,
                        generator=generator,
                    )

                dp_output = dp(base_batch, conditions, validity, map_index=map_index)
                dp_batch = dp.apply_to_payload(base_batch, dp_output)
                if route.use_link_global_aod:
                    assert link is not None
                    link_output = link(
                        dp_batch, conditions, validity, map_index=map_index
                    )
                    final_batch = link.apply_to_payload(dp_batch, link_output)
                else:
                    final_batch = dp_batch

                if apply_filter:
                    assert route.target_mean_aod_az_deg is not None
                    measurement = projector(
                        final_batch,
                        route.target_mean_aod_az_deg,
                        mode="mean",
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
            if not apply_filter:
                break

        if apply_filter and eligible_candidate_samples < args.num_samples:
            report_path = (
                Path(args.report)
                if args.report
                else Path(args.output).with_suffix(".json")
            )
            failure_report = {
                "format_version": 3,
                "experiment": "dual_route_physics_controllable_csi_generation",
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
                "message": (
                    "Maximum candidate budget was exhausted without enough "
                    "naturally eligible CSI; no output artifact was written."
                ),
            }
            atomic_json_dump(failure_report, report_path)
            print(json.dumps(failure_report, indent=2, sort_keys=True), flush=True)
            raise RuntimeError(
                "insufficient naturally eligible Mean-AoD candidates at the "
                "configured maximum budget"
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
                route.target_mean_aod_az_deg,
                samples=args.num_samples,
                tolerance_deg=args.natural_angle_tolerance_deg,
                mode="mean",
                projector=projector,
                generator=torch.Generator().manual_seed(selection_seed),
            )
            base = select_payload_samples(base, selection.indices)
            dp_payload = select_payload_samples(dp_payload, selection.indices)
            final_payload = select_payload_samples(final_payload, selection.indices)
            selection_report = {
                "policy": "uniform_random_among_natural_mean_aod_eligible",
                "candidate_samples": selection.candidate_samples,
                "eligible_samples": selection.eligible_samples,
                "acceptance_rate": selection.eligible_samples
                / selection.candidate_samples,
                "selected_samples": selection.requested_samples,
                "unique_selected_samples": int(torch.unique(selection.indices).numel()),
                "candidate_multiplier": args.candidate_multiplier,
                "candidate_top_up_size": args.candidate_top_up_size,
                "maximum_candidate_multiplier": args.maximum_candidate_multiplier,
                "maximum_candidate_budget": maximum_candidate_samples,
                "candidate_generation_round_sizes": generation_round_sizes,
                "natural_angle_tolerance_deg": args.natural_angle_tolerance_deg,
                "tolerance_was_relaxed": False,
                "selection_seed": selection_seed,
                "selected_angle_error_deg": _audit(selection.angular_distance_deg),
                "selected_candidate_indices": selection.indices.cpu().tolist(),
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
        if route.target_mean_aod_az_deg is not None:
            measurement = AoDControlProjector(
                minimum_mean_concentration=args.minimum_mean_concentration
            )(final_device, route.target_mean_aod_az_deg, mode="mean")
            error = measurement.rotation_deg.abs()
            control_report = {
                "target_mean_aod_az_deg": route.target_mean_aod_az_deg,
                "projection_applied": False,
                "mean_aod_error_deg": _audit(error),
                "hit_at_5_deg": float((error <= 5.0).float().mean()),
                "hit_at_10_deg": float((error <= 10.0).float().mean()),
                "valid_rate": float(measurement.control_valid.float().mean()),
                "mean_resultant_concentration": _audit(
                    measurement.mean_resultant_concentration
                ),
            }
            control_tensors = {
                "target_mean_aod_az_deg": measurement.target_az_deg,
                "actual_mean_aod_az_deg": measurement.reference_before_deg,
                "mean_aod_circular_error_deg": error,
                "mean_aod_valid": measurement.control_valid,
                "mean_aod_concentration": measurement.mean_resultant_concentration,
            }

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
    )
    pipeline = (
        ["B3", "DP Residual", "Link-global AoD", "physics check", "CSI synthesis"]
        if route.use_natural_b3
        else [
            f"AoD-conditioned B3 (CFG={route.guidance_scale:g})",
            "DP Residual",
            (
                "bounded candidate top-up + natural Mean-AoD tolerance filter "
                "+ uniform eligible sampling"
                if apply_filter
                else "no AoD filter"
            ),
            "physics check",
            "CSI synthesis",
        ]
    )
    checkpoint_paths = {
        "dp": str(Path(args.dp_checkpoint).resolve()),
        "b3": str(Path(args.b3_checkpoint).resolve()) if args.b3_checkpoint else None,
        "aod_conditioned_b3": (
            str(Path(args.aod_checkpoint).resolve()) if args.aod_checkpoint else None
        ),
        "link_global": (
            str(Path(args.link_checkpoint).resolve()) if args.link_checkpoint else None
        ),
    }
    artifact = {
        "format_version": 3,
        "model_kind": "dual_route_physics_controllable_csi_generator",
        "generation_mode": route.mode.value,
        "pipeline": pipeline,
        "conditions": condition_dict,
        "condition_validity": output_validity[0],
        "map_id": args.map_id,
        "path_count": args.path_count,
        "target_mean_aod_az_deg": route.target_mean_aod_az_deg,
        "seed": args.seed,
        "base": _cpu(base),
        "dp_only": _cpu(dp_payload),
        "generated": _cpu(final_payload),
        "control_audit": _cpu(control_tensors),
        "selection_audit": selection_report,
        "checkpoint_paths": checkpoint_paths,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(output)

    report = {
        "format_version": 3,
        "experiment": "dual_route_physics_controllable_csi_generation",
        "status": "success",
        "artifact": str(output.resolve()),
        "generation_mode": route.mode.value,
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
        "scene_condition_support": support,
        "checkpoint_paths": checkpoint_paths,
        "limitations": (
            [
                "Mean-AoD mode is currently trained and validated only for L=5.",
                "Natural tolerance filtering enforces output tolerance without angle rotation.",
                "Eligible candidates are sampled uniformly without replacement, not ranked by angular error.",
                "Candidate top-up is bounded and never relaxes the requested angle tolerance.",
                "Low-support Map-condition-angle requests remain extrapolations.",
            ]
            if route.use_aod_conditioned_b3
            else []
        ),
    }
    report_path = Path(args.report) if args.report else output.with_suffix(".json")
    atomic_json_dump(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
