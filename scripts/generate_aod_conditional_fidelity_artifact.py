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

from physics_csi.data.delay_power_residual import DelayPowerResidualCacheDataset
from physics_csi.evaluation.aod_support import generated_support_counts, mean_aod
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.models.scene_link_global_aod_residual import (
    SceneConditionedLinkGlobalAoDResidual,
)
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import move_to_device, resolve_device


def _subset(payload: Mapping[str, Any], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    sample_count = int(
        next(value for value in payload.values() if isinstance(value, torch.Tensor)).shape[0]
    )
    return {
        key: value[indices]
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (sample_count,)
    }


def _append(chunks: dict[str, list[torch.Tensor]], payload: Mapping[str, torch.Tensor]) -> None:
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            chunks.setdefault(key, []).append(value.detach().cpu())


def _finish(chunks: Mapping[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(values) for key, values in chunks.items()}


def _load_dp(checkpoint: Mapping[str, Any], device: torch.device) -> SceneConditionedDelayPowerResidual:
    if checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("invalid DP checkpoint")
    model = SceneConditionedDelayPowerResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    return model.requires_grad_(False).eval()


def _load_link(
    checkpoint: Mapping[str, Any], device: torch.device
) -> SceneConditionedLinkGlobalAoDResidual:
    if checkpoint.get("model_kind") != "scene_link_global_aod_residual":
        raise ValueError("invalid Link-global checkpoint")
    model = SceneConditionedLinkGlobalAoDResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    return model.requires_grad_(False).eval()


def _synthesize(payload: dict[str, torch.Tensor], model: MIMOOFDMSynthesizer) -> None:
    payload["csi"] = model(
        payload["path_delays_ns"],
        payload["path_powers_linear"],
        payload["path_phases_rad"],
        payload["aoa_az_deg"],
        payload["aoa_el_deg"],
        payload["aod_az_deg"],
        payload["aod_el_deg"],
        payload["path_mask"],
    )


def _decorate(
    payload: dict[str, torch.Tensor],
    conditions: torch.Tensor,
    validity: torch.Tensor,
    map_ids: torch.Tensor,
    synthesizer: MIMOOFDMSynthesizer,
) -> dict[str, torch.Tensor]:
    payload["conditions"] = conditions
    payload["condition_validity"] = validity
    payload["map_id"] = map_ids
    _synthesize(payload, synthesizer)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matched CFG/DP/Link variants on high-support AoD conditions."
    )
    parser.add_argument("--aod-checkpoint", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--link-checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--pair-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    device = resolve_device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    plan = torch.load(args.pair_plan, map_location="cpu", weights_only=False)
    if plan.get("kind") != "map_condition_aod_supported_pair_plan":
        raise ValueError("pair plan has an unsupported kind")
    if plan.get("cache_scene_plan_fingerprint") != cache.get("scene_plan_fingerprint"):
        raise ValueError("pair plan and cache use different scene splits")
    dataset = DelayPowerResidualCacheDataset(cache, plan["split"])
    query_indices = plan["query_indices"].long()
    donor_indices = plan["donor_indices"].long()
    if query_indices.shape != donor_indices.shape:
        raise ValueError("query and donor indices must have equal shape")

    aod_checkpoint = torch.load(
        args.aod_checkpoint, map_location="cpu", weights_only=False
    )
    if aod_checkpoint.get("model_kind") != "multipath_diffusion_scene_id_aod_conditioned":
        raise ValueError("invalid AoD-conditioned B3 checkpoint")
    config = aod_checkpoint["config"]
    if (
        config["scene_id_experiment"]["fingerprint"]
        != cache["scene_plan_fingerprint"]
    ):
        raise ValueError("AoD checkpoint and cache use different scene plans")
    system = SystemConfig.from_mapping(config["data"]["system"])
    aod_model = MultipathDiffusionModel(
        config["multipath_diffusion"], system, aod_checkpoint["statistics"]
    ).to(device)
    aod_model.load_state_dict(
        aod_checkpoint.get("ema_state_dict", aod_checkpoint["model_state_dict"])
    )
    aod_model.requires_grad_(False).eval()
    dp_checkpoint = torch.load(args.dp_checkpoint, map_location="cpu", weights_only=False)
    link_checkpoint = torch.load(
        args.link_checkpoint, map_location="cpu", weights_only=False
    )
    dp_model = _load_dp(dp_checkpoint, device)
    link_model = _load_link(link_checkpoint, device)
    if not (
        aod_model.map_id_to_index == dp_model.map_id_to_index == link_model.map_id_to_index
    ):
        raise ValueError("AoD, DP and Link models use different Map mappings")
    synthesizer = MIMOOFDMSynthesizer(system).to(device).eval()
    sampling = config["sampling"]
    sampling_steps = args.sampling_steps or int(sampling["sampling_steps"])
    forced_l = int(config.get("aod_conditioned_data", {}).get("path_count", 5))
    labels = (
        "reference",
        "real_query",
        "baseline_b3_dp_link",
        "aod_cfg1_dp",
        "aod_cfg2_dp",
        "aod_cfg3_dp",
        "aod_cfg2_dp_link",
        "aod_cfg3_dp_link",
    )
    chunks: dict[str, dict[str, list[torch.Tensor]]] = {
        label: {} for label in labels
    }
    targets: list[torch.Tensor] = []
    query_support: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, query_indices.numel(), args.batch_size):
            query = query_indices[start : start + args.batch_size]
            donor = donor_indices[start : start + args.batch_size]
            real_query = move_to_device(_subset(dataset.reference, query), device)
            reference = move_to_device(_subset(dataset.reference, donor), device)
            old_base = move_to_device(_subset(dataset.base, query), device)
            conditions = real_query["conditions"].float()
            validity = real_query["condition_validity"].bool()
            map_ids = real_query["map_id"].long()
            map_index = aod_model.map_indices(map_ids, map_ids.numel(), device)
            target_angle, _ = mean_aod(real_query)

            # Original five-condition reference baseline.
            baseline_dp_output = dp_model(
                old_base, conditions, validity, map_index=map_index
            )
            baseline_dp = dp_model.apply_to_payload(old_base, baseline_dp_output)
            baseline_link_output = link_model(
                baseline_dp, conditions, validity, map_index=map_index
            )
            baseline = link_model.apply_to_payload(
                baseline_dp, baseline_link_output
            )
            _decorate(baseline, conditions, validity, map_ids, synthesizer)

            generated_dp: dict[int, dict[str, torch.Tensor]] = {}
            generated_link: dict[int, dict[str, torch.Tensor]] = {}
            for scale in (1, 2, 3):
                # Reset the generator for a paired CFG comparison: only s changes.
                generator = torch.Generator(device=device).manual_seed(args.seed + start)
                generated = aod_model.generate(
                    conditions,
                    conditions.shape[0],
                    condition_validity=validity,
                    sampler=str(sampling["sampler"]),
                    sampling_steps=sampling_steps,
                    eta=float(sampling.get("eta", 0.0)),
                    guidance_scale=float(scale),
                    path_count_temperature=float(
                        sampling.get("path_count_temperature", 1.0)
                    ),
                    forced_path_count=forced_l,
                    mean_aod_az_deg=target_angle,
                    map_id=map_ids,
                    generator=generator,
                )
                dp_output = dp_model(
                    generated, conditions, validity, map_index=map_index
                )
                dp_payload = dp_model.apply_to_payload(generated, dp_output)
                _decorate(dp_payload, conditions, validity, map_ids, synthesizer)
                generated_dp[scale] = dp_payload
                if scale in (2, 3):
                    link_output = link_model(
                        dp_payload, conditions, validity, map_index=map_index
                    )
                    link_payload = link_model.apply_to_payload(dp_payload, link_output)
                    _decorate(link_payload, conditions, validity, map_ids, synthesizer)
                    generated_link[scale] = link_payload

            for payload in (real_query, reference):
                if "csi" not in payload:
                    _synthesize(payload, synthesizer)
            batch_payloads = {
                "reference": reference,
                "real_query": real_query,
                "baseline_b3_dp_link": baseline,
                "aod_cfg1_dp": generated_dp[1],
                "aod_cfg2_dp": generated_dp[2],
                "aod_cfg3_dp": generated_dp[3],
                "aod_cfg2_dp_link": generated_link[2],
                "aod_cfg3_dp_link": generated_link[3],
            }
            for label, payload in batch_payloads.items():
                _append(chunks[label], payload)
            targets.append(target_angle.cpu())
            query_support.append(plan["query_support_count"][start : start + query.numel()])
            print(
                json.dumps(
                    {
                        "event": "conditional_fidelity_generation_progress",
                        "complete": min(start + args.batch_size, query_indices.numel()),
                        "total": int(query_indices.numel()),
                    }
                ),
                flush=True,
            )

    payloads = {label: _finish(chunks[label]) for label in labels}
    target_angle = torch.cat(targets)
    target_conditions = payloads["real_query"]["conditions"]
    target_validity = payloads["real_query"]["condition_validity"]
    target_map_ids = payloads["real_query"]["map_id"]
    support_protocol = plan["protocol"]
    output_support: dict[str, torch.Tensor] = {}
    for label in labels[2:]:
        output_support[label] = generated_support_counts(
            payloads[label],
            target_conditions,
            target_validity,
            target_map_ids,
            dataset.reference,
            condition_std=plan["condition_std"],
            maximum_condition_distance=float(
                support_protocol["maximum_standardized_condition_distance"]
            ),
            maximum_angle_distance_deg=float(
                support_protocol["maximum_circular_angle_distance_deg"]
            ),
        )
    artifact = {
        "format_version": 1,
        "model_kind": "aod_conditioned_cfg_link_fidelity_comparison",
        "split": plan["split"],
        "payloads": payloads,
        "target_mean_aod_deg": target_angle,
        "query_real_support_count": torch.cat(query_support),
        "generated_real_support_count": output_support,
        "support_protocol": support_protocol,
        "query_indices": query_indices,
        "donor_indices": donor_indices,
        "system": system.as_dict(),
        "scene_plan_fingerprint": cache["scene_plan_fingerprint"],
        "sampling": {
            "steps": sampling_steps,
            "sampler": sampling["sampler"],
            "eta": float(sampling.get("eta", 0.0)),
            "seed": args.seed,
            "paired_noise_across_cfg": True,
        },
        "sources": {
            "aod_checkpoint": str(Path(args.aod_checkpoint).resolve()),
            "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()),
            "link_checkpoint": str(Path(args.link_checkpoint).resolve()),
            "cache": str(Path(args.cache).resolve()),
            "pair_plan": str(Path(args.pair_plan).resolve()),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "event": "aod_conditional_fidelity_artifact_complete",
                "output": str(output.resolve()),
                "samples": int(target_angle.numel()),
                "split": plan["split"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
