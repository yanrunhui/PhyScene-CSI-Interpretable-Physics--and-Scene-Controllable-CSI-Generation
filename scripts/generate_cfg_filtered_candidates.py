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

from physics_csi.evaluation.filtered_candidates import (
    select_first_eligible_candidates,
)
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.models.scene_delay_power_residual import (
    SceneConditionedDelayPowerResidual,
)
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import resolve_device


OUTPUT_KEYS = (
    "path_mask",
    "path_delays_ns",
    "path_powers_linear",
    "path_phases_rad",
    "aoa_az_deg",
    "aoa_el_deg",
    "aod_az_deg",
    "aod_el_deg",
    "direct_path_mask",
    "actual_conditions",
    "actual_condition_validity",
    "constraint_feasible",
)


def _load_dp(
    checkpoint: Mapping[str, Any], device: torch.device
) -> SceneConditionedDelayPowerResidual:
    if checkpoint.get("model_kind") != "scene_delay_power_residual":
        raise ValueError("invalid DP residual checkpoint")
    model = SceneConditionedDelayPowerResidual(**checkpoint["model_init"]).to(device)
    model.load_state_dict(
        checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
    )
    return model.requires_grad_(False).eval()


def _append_selected(
    chunks: dict[str, list[torch.Tensor]],
    payload: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> None:
    queries, outputs = indices.shape
    flat = indices.reshape(-1)
    sample_count = int(payload["path_mask"].shape[0])
    for key in OUTPUT_KEYS:
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or value.shape[:1] != (sample_count,):
            continue
        selected = value[flat]
        chunks.setdefault(key, []).append(
            selected.reshape(queries, outputs, *selected.shape[1:]).detach().cpu()
        )


def _finish(chunks: Mapping[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired CFG2/CFG3 candidate pools and retain the first eight "
            "naturally accepted Mean-AoD samples per query."
        )
    )
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--aod-checkpoint", required=True)
    parser.add_argument("--dp-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates-per-query", type=int, default=16)
    parser.add_argument("--outputs-per-query", type=int, default=8)
    parser.add_argument("--tolerance-deg", type=float, default=10.0)
    parser.add_argument("--minimum-concentration", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.outputs_per_query > args.candidates_per_query:
        raise ValueError("outputs-per-query cannot exceed candidates-per-query")

    device = resolve_device(args.device)
    source_path = Path(args.source_artifact)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("model_kind") != "aod_conditioned_cfg_link_fidelity_comparison":
        raise ValueError("source is not an AoD conditional-fidelity artifact")
    if source.get("split") != "test":
        raise ValueError("filtered-candidate comparison requires the independent Test artifact")
    real_query = source["payloads"]["real_query"]
    required = ("conditions", "condition_validity", "map_id", "path_mask")
    if any(key not in real_query for key in required):
        raise ValueError("source real_query is missing conditions, Map IDs or path masks")
    queries = int(real_query["conditions"].shape[0])
    target = source["target_mean_aod_deg"].float()
    if target.shape != (queries,):
        raise ValueError("source target Mean AoD does not align with queries")

    aod_checkpoint = torch.load(
        args.aod_checkpoint, map_location="cpu", weights_only=False
    )
    if aod_checkpoint.get("model_kind") != "multipath_diffusion_scene_id_aod_conditioned":
        raise ValueError("invalid AoD-conditioned B3 checkpoint")
    config = aod_checkpoint["config"]
    trained_l = int(config.get("aod_conditioned_data", {}).get("path_count", -1))
    if trained_l != 5:
        raise ValueError(f"expected the validated L=5 checkpoint, found L={trained_l}")
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
    dp = _load_dp(dp_checkpoint, device)
    if model.map_id_to_index != dp.map_id_to_index:
        raise ValueError("AoD-conditioned B3 and DP residual use different Map mappings")
    if not bool((real_query["path_mask"].sum(-1) == trained_l).all()):
        raise ValueError("source Test queries are not all L=5")

    sampling = config["sampling"]
    sampling_steps = args.sampling_steps or int(sampling["sampling_steps"])
    chunks: dict[str, dict[str, list[torch.Tensor]]] = {
        "cfg2": {},
        "cfg3": {},
    }
    success_chunks: dict[str, list[torch.Tensor]] = {"cfg2": [], "cfg3": []}
    eligible_chunks: dict[str, list[torch.Tensor]] = {"cfg2": [], "cfg3": []}
    error_chunks: dict[str, list[torch.Tensor]] = {"cfg2": [], "cfg3": []}

    with torch.inference_mode():
        for start in range(0, queries, args.batch_size):
            stop = min(start + args.batch_size, queries)
            batch_queries = stop - start
            conditions = real_query["conditions"][start:stop].to(device).float()
            validity = real_query["condition_validity"][start:stop].to(device).bool()
            map_ids = real_query["map_id"][start:stop].to(device).long()
            targets = target[start:stop].to(device)
            repeated_conditions = conditions.repeat_interleave(
                args.candidates_per_query, dim=0
            )
            repeated_validity = validity.repeat_interleave(
                args.candidates_per_query, dim=0
            )
            repeated_maps = map_ids.repeat_interleave(args.candidates_per_query)
            repeated_targets = targets.repeat_interleave(args.candidates_per_query)
            map_index = dp.map_indices(repeated_maps)

            for scale in (2, 3):
                # Identical initial noise across CFG2 and CFG3 within each batch.
                generator = torch.Generator(device=device).manual_seed(args.seed + start)
                generated = model.generate(
                    repeated_conditions,
                    repeated_conditions.shape[0],
                    condition_validity=repeated_validity,
                    sampler=str(sampling["sampler"]),
                    sampling_steps=sampling_steps,
                    eta=float(sampling.get("eta", 0.0)),
                    guidance_scale=float(scale),
                    forced_path_count=trained_l,
                    mean_aod_az_deg=repeated_targets,
                    map_id=repeated_maps,
                    generator=generator,
                )
                dp_output = dp(
                    generated,
                    repeated_conditions,
                    repeated_validity,
                    map_index=map_index,
                )
                filtered = dp.apply_to_payload(generated, dp_output)
                selection = select_first_eligible_candidates(
                    filtered,
                    targets,
                    queries=batch_queries,
                    candidates_per_query=args.candidates_per_query,
                    outputs_per_query=args.outputs_per_query,
                    tolerance_deg=args.tolerance_deg,
                    minimum_concentration=args.minimum_concentration,
                )
                label = f"cfg{scale}"
                _append_selected(
                    chunks[label], filtered, selection.selected_flat_indices
                )
                success_chunks[label].append(selection.success.cpu())
                eligible_chunks[label].append(selection.eligible_count.cpu())
                error_chunks[label].append(selection.selected_error_deg.cpu())
                del generated, dp_output, filtered

            print(
                json.dumps(
                    {
                        "event": "cfg_filtered_candidate_progress",
                        "complete": stop,
                        "total": queries,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    variants = {label: _finish(chunks[label]) for label in chunks}
    audit = {
        label: {
            "success": torch.cat(success_chunks[label]),
            "eligible_count": torch.cat(eligible_chunks[label]),
            "selected_error_deg": torch.cat(error_chunks[label]),
        }
        for label in chunks
    }
    output_payload = {
        "format_version": 1,
        "model_kind": "cfg2_cfg3_filtered_candidate_comparison",
        "source_artifact": str(source_path.resolve()),
        "variants": variants,
        "selection_audit": audit,
        "protocol": {
            "cfg_scales": [2.0, 3.0],
            "candidates_per_query": args.candidates_per_query,
            "outputs_per_query": args.outputs_per_query,
            "natural_mean_aod_tolerance_deg": args.tolerance_deg,
            "minimum_mean_aod_concentration": args.minimum_concentration,
            "selection": "first eligible candidates in independent generation order",
            "angle_projection_applied": False,
            "paired_initial_noise_across_cfg": True,
            "path_count": trained_l,
        },
        "sampling": {
            "seed": args.seed,
            "sampler": sampling["sampler"],
            "steps": sampling_steps,
            "eta": float(sampling.get("eta", 0.0)),
        },
        "system": system.as_dict(),
        "sources": {
            "aod_checkpoint": str(Path(args.aod_checkpoint).resolve()),
            "dp_checkpoint": str(Path(args.dp_checkpoint).resolve()),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(output_payload, temporary)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "event": "cfg_filtered_candidate_artifact_complete",
                "output": str(output.resolve()),
                "queries": queries,
                "cfg2_success_rate": float(audit["cfg2"]["success"].float().mean()),
                "cfg3_success_rate": float(audit["cfg3"]["success"].float().mean()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
