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

from physics_csi.evaluation.aod_support import enforce_final_strongest_anchor
from physics_csi.evaluation.diversity_selection import (
    candidate_diversity_features,
    select_mild_diversity_candidates,
    select_random_candidates,
    strongest_candidate_eligibility,
)
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.models.scene_delay_power_residual import SceneConditionedDelayPowerResidual
from physics_csi.physics.aod_control import select_payload_samples
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig
from physics_csi.training.trainer import resolve_device


VARIANTS = {
    "direct_8": (8, "random"),
    "random_16_to_8": (16, "random"),
    "diversity_16_to_8": (16, "diversity"),
    "random_24_to_8": (24, "random"),
    "diversity_24_to_8": (24, "diversity"),
}


def _load_anchor(
    checkpoint: Mapping[str, Any], device: torch.device
) -> MultipathDiffusionModel:
    if checkpoint.get("model_kind") != "multipath_diffusion_scene_id_strongest_aod_anchor":
        raise ValueError("--aod-checkpoint is not a Strongest Anchor checkpoint")
    config = checkpoint["config"]
    system = SystemConfig.from_mapping(config["data"]["system"])
    model = MultipathDiffusionModel(
        config["multipath_diffusion"], system, checkpoint["statistics"]
    ).to(device)
    model.load_state_dict(
        checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
    )
    return model.requires_grad_(False).eval()


def _load_dp(
    checkpoint: Mapping[str, Any], device: torch.device
) -> SceneConditionedDelayPowerResidual:
    if checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("--dp-checkpoint is not a scene Delay-Power residual")
    model = SceneConditionedDelayPowerResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(
        checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
    )
    return model.requires_grad_(False).eval()


def _query_subset(
    payload: Mapping[str, Any], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    samples = int(payload["path_mask"].shape[0])
    return {
        key: value[indices]
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and value.shape[:1] == (samples,)
    }


def _append(
    chunks: dict[str, list[torch.Tensor]], payload: Mapping[str, torch.Tensor]
) -> None:
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            chunks.setdefault(key, []).append(value.detach().cpu())


def _finish(chunks: Mapping[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(values) for key, values in chunks.items()}


def _choose_queries(
    map_ids: torch.Tensor, maximum_per_map: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    selected: list[torch.Tensor] = []
    for map_tensor in torch.unique(map_ids, sorted=True):
        indices = torch.nonzero(map_ids == map_tensor, as_tuple=False).flatten()
        order = torch.randperm(indices.numel(), generator=generator)
        selected.append(indices[order[: min(maximum_per_map, indices.numel())]])
    return torch.cat(selected).sort().values


def _synthesize(
    payload: dict[str, torch.Tensor], synthesizer: MIMOOFDMSynthesizer
) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one shared CFG1 Anchor+DP+Swap pool and derive random and "
            "mild-diversity 8-output selections at budgets 8, 16 and 24."
        )
    )
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--aod-checkpoint", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-queries-per-map", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--candidates-per-query", type=int, default=24)
    parser.add_argument("--outputs-per-query", type=int, default=8)
    parser.add_argument("--diversity-strength", type=float, default=0.35)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--angle-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--minimum-real-support", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.candidates_per_query != 24 or args.outputs_per_query != 8:
        raise ValueError("the locked experiment contract requires a 24-candidate pool and 8 outputs")
    if args.maximum_queries_per_map < 1 or args.query_batch_size < 1:
        raise ValueError("query limits must be positive")
    if args.minimum_real_support < 1:
        raise ValueError("minimum real support must be positive")

    device = resolve_device(args.device)
    source = torch.load(args.source_artifact, map_location="cpu", weights_only=False)
    if source.get("model_kind") != "strongest_anchor_rebinding_ablation":
        raise ValueError("--source-artifact must be the paired Anchor rebinding artifact")
    if int(source.get("cfg", -1)) != 1:
        raise ValueError("the diversity experiment is locked to CFG1")
    real_query_source = source["payloads"]["real_query"]
    reference_source = source["payloads"]["reference"]
    target_source = source["target_aod_deg"].float()
    test_artifact_path = Path(source.get("sources", {}).get("test_artifact", ""))
    if not test_artifact_path.is_file():
        raise FileNotFoundError(
            "the linked supported Test artifact is required to verify class-A support: "
            f"{test_artifact_path}"
        )
    test_artifact = torch.load(
        test_artifact_path, map_location="cpu", weights_only=False
    )
    real_support_source = test_artifact.get("query_real_support_count")
    if not isinstance(real_support_source, torch.Tensor) or real_support_source.shape != target_source.shape:
        raise ValueError("linked Test artifact has no aligned query_real_support_count")
    if bool((real_support_source < args.minimum_real_support).any()):
        raise ValueError(
            "source contains non-A requests below --minimum-real-support; "
            "use the supported-pair Test artifact"
        )
    query_indices = _choose_queries(
        real_query_source["map_id"].long(),
        args.maximum_queries_per_map,
        args.seed,
    )
    real_query = _query_subset(real_query_source, query_indices)
    reference = _query_subset(reference_source, query_indices)
    targets = target_source[query_indices]
    real_support = real_support_source[query_indices]

    aod_checkpoint = torch.load(
        args.aod_checkpoint, map_location="cpu", weights_only=False
    )
    dp_checkpoint = torch.load(
        args.dp_checkpoint, map_location="cpu", weights_only=False
    )
    anchor = _load_anchor(aod_checkpoint, device)
    dp = _load_dp(dp_checkpoint, device)
    if anchor.map_id_to_index != dp.map_id_to_index:
        raise ValueError("Anchor and DP checkpoints use different Map mappings")
    system = SystemConfig.from_mapping(aod_checkpoint["config"]["data"]["system"])
    synthesizer = MIMOOFDMSynthesizer(system).to(device).eval()
    sampling = aod_checkpoint["config"]["sampling"]
    steps = args.sampling_steps or int(sampling["sampling_steps"])
    forced_paths = int(
        aod_checkpoint["config"].get("aod_conditioned_data", {}).get("path_count", 5)
    )

    chunks: dict[str, dict[str, list[torch.Tensor]]] = {
        variant: {} for variant in VARIANTS
    }
    query_id_chunks: dict[str, list[torch.Tensor]] = {variant: [] for variant in VARIANTS}
    audit: dict[str, list[torch.Tensor]] = {
        variant: [] for variant in VARIANTS
    }
    success_audit: dict[str, list[torch.Tensor]] = {
        variant: [] for variant in VARIANTS
    }
    with torch.inference_mode():
        total_queries = int(query_indices.numel())
        for start in range(0, total_queries, args.query_batch_size):
            stop = min(start + args.query_batch_size, total_queries)
            batch_queries = stop - start
            conditions = real_query["conditions"][start:stop].to(device).float()
            validity = real_query["condition_validity"][start:stop].to(device).bool()
            maps = real_query["map_id"][start:stop].to(device).long()
            target = targets[start:stop].to(device)
            repeated_conditions = conditions.repeat_interleave(args.candidates_per_query, 0)
            repeated_validity = validity.repeat_interleave(args.candidates_per_query, 0)
            repeated_maps = maps.repeat_interleave(args.candidates_per_query)
            repeated_target = target.repeat_interleave(args.candidates_per_query)
            generated = anchor.generate(
                repeated_conditions,
                repeated_conditions.shape[0],
                condition_validity=repeated_validity,
                sampler=str(sampling["sampler"]),
                sampling_steps=steps,
                eta=float(sampling.get("eta", 0.0)),
                guidance_scale=args.guidance_scale,
                forced_path_count=forced_paths,
                strongest_aod_az_deg=repeated_target,
                map_id=repeated_maps,
                generator=torch.Generator(device=device).manual_seed(args.seed + start),
            )
            generated["conditions"] = repeated_conditions
            generated["condition_validity"] = repeated_validity
            generated["map_id"] = repeated_maps
            map_index = dp.map_indices(repeated_maps)
            dp_output = dp(
                generated,
                repeated_conditions,
                repeated_validity,
                map_index=map_index,
            )
            pool = dp.apply_to_payload(generated, dp_output)
            pool = enforce_final_strongest_anchor(
                pool, repeated_target, strategy="swap"
            )
            _synthesize(pool, synthesizer)
            eligibility = strongest_candidate_eligibility(
                pool,
                target,
                queries=batch_queries,
                candidates_per_query=args.candidates_per_query,
                tolerance_deg=args.angle_tolerance_deg,
            )
            features = candidate_diversity_features(pool, repeated_target).reshape(
                batch_queries, args.candidates_per_query, -1
            )
            for variant_number, (variant, (budget, method)) in enumerate(VARIANTS.items()):
                current_eligible = eligibility[:, :budget]
                selection_generator = torch.Generator(device=device).manual_seed(
                    args.seed + 100_000 + start * 10 + variant_number
                )
                if method == "diversity":
                    selection = select_mild_diversity_candidates(
                        features[:, :budget],
                        current_eligible,
                        outputs_per_query=args.outputs_per_query,
                        diversity_strength=args.diversity_strength,
                        generator=selection_generator,
                    )
                else:
                    selection = select_random_candidates(
                        current_eligible,
                        outputs_per_query=args.outputs_per_query,
                        generator=selection_generator,
                    )
                flat = selection.local_indices + (
                    torch.arange(batch_queries, device=device).unsqueeze(-1)
                    * args.candidates_per_query
                )
                selected = select_payload_samples(pool, flat.flatten())
                _append(chunks[variant], selected)
                global_query = torch.arange(start, stop).repeat_interleave(
                    args.outputs_per_query
                )
                query_id_chunks[variant].append(global_query)
                audit[variant].append(selection.eligible_count.detach().cpu())
                success_audit[variant].append(selection.success.detach().cpu())
            print(
                json.dumps(
                    {
                        "event": "strongest_anchor_diversity_generation_progress",
                        "complete_queries": stop,
                        "total_queries": total_queries,
                    }
                ),
                flush=True,
            )

    payloads = {variant: _finish(values) for variant, values in chunks.items()}
    for variant in VARIANTS:
        payloads[variant]["query_index"] = torch.cat(query_id_chunks[variant])
    output = {
        "format_version": 1,
        "model_kind": "strongest_anchor_diversity_selection",
        "cfg": 1,
        "query_indices_in_source": query_indices,
        "target_aod_deg": targets,
        "query_real_support_count": real_support,
        "payloads": {
            "reference": reference,
            "real_query": real_query,
            **payloads,
        },
        "selection_audit": {
            variant: {
                "candidate_budget": VARIANTS[variant][0],
                "method": VARIANTS[variant][1],
                "eligible_count_min": int(torch.cat(audit[variant]).min()),
                "eligible_count_mean": float(torch.cat(audit[variant]).float().mean()),
                "query_success_rate": float(
                    torch.cat(success_audit[variant]).float().mean()
                ),
                "successful_queries": int(torch.cat(success_audit[variant]).sum()),
                "queries": int(torch.cat(success_audit[variant]).numel()),
            }
            for variant in VARIANTS
        },
        "protocol": {
            "shared_maximum_candidate_pool": 24,
            "outputs_per_query": 8,
            "same_checkpoint_conditions_and_noise_pool": True,
            "diversity_selector": "tempered facility-location representativeness",
            "diversity_strength": args.diversity_strength,
            "selection_features": (
                "relative delay, target-centered AoD sin/cos, normalized path power, "
                "path mask, Tx energy and adjacent-Tx complex correlation"
            ),
            "extreme_farthest_point_selection": False,
            "support_class": "A: Map-conditioned real support is at or above the locked threshold",
            "minimum_real_support": args.minimum_real_support,
        },
        "system": system.as_dict(),
        "sources": {
            "source_artifact": str(Path(args.source_artifact).resolve()),
            "aod_checkpoint": str(Path(args.aod_checkpoint).resolve()),
            "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()),
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(output, temporary)
    temporary.replace(destination)
    print(
        json.dumps(
            {
                "event": "strongest_anchor_diversity_artifact_complete",
                "output": str(destination.resolve()),
                "queries": int(query_indices.numel()),
                "selected_samples_per_variant": int(query_indices.numel()) * 8,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
