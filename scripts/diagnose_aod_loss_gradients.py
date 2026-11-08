#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.aod_conditioned import make_aod_conditioned_datasets
from physics_csi.data.scene_id import make_scene_id_datasets
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.trainer import move_to_device, resolve_device
from physics_csi.utils.io import atomic_json_dump
from physics_csi.utils.seed import seed_everything


def _gradient_pair(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> list[torch.Tensor | None]:
    return list(
        torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )
    )


def _group_name(parameter_name: str) -> str:
    for prefix in (
        "denoiser",
        "aod_conditioner",
        "scene_conditioner",
        "condition_encoder",
        "path_count_model",
    ):
        if parameter_name.startswith(prefix + "."):
            return prefix
    return "other"


def _gradient_statistics(
    names: list[str],
    diffusion: list[torch.Tensor | None],
    aod: list[torch.Tensor | None],
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    groups["all"] = list(range(len(names)))
    for index, name in enumerate(names):
        groups[_group_name(name)].append(index)
    result: dict[str, dict[str, float | int]] = {}
    for group, indices in groups.items():
        diffusion_sq = torch.zeros((), dtype=torch.float64)
        aod_sq = torch.zeros((), dtype=torch.float64)
        dot = torch.zeros((), dtype=torch.float64)
        diffusion_connected = 0
        aod_connected = 0
        for index in indices:
            left = diffusion[index]
            right = aod[index]
            if left is not None:
                diffusion_connected += 1
                diffusion_sq += left.detach().double().square().sum().cpu()
            if right is not None:
                aod_connected += 1
                aod_sq += right.detach().double().square().sum().cpu()
            if left is not None and right is not None:
                dot += (left.detach().double() * right.detach().double()).sum().cpu()
        diffusion_norm = float(diffusion_sq.sqrt())
        aod_norm = float(aod_sq.sqrt())
        denominator = max(diffusion_norm * aod_norm, 1e-30)
        result[group] = {
            "parameter_tensors": len(indices),
            "diffusion_connected_tensors": diffusion_connected,
            "weighted_aod_connected_tensors": aod_connected,
            "diffusion_gradient_l2": diffusion_norm,
            "weighted_aod_gradient_l2": aod_norm,
            "weighted_aod_to_diffusion_ratio": aod_norm / max(diffusion_norm, 1e-30),
            "gradient_cosine": float(dot) / denominator,
        }
    return result


def _mean_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    scalar_keys = sorted(
        key
        for key in records[0]
        if isinstance(records[0][key], (int, float)) and key != "batch"
    )
    result: dict[str, Any] = {
        key: sum(float(record[key]) for record in records) / len(records)
        for key in scalar_keys
    }
    group_names = sorted(records[0]["gradient_groups"])
    result["gradient_groups"] = {}
    for group in group_names:
        keys = sorted(records[0]["gradient_groups"][group])
        result["gradient_groups"][group] = {
            key: sum(
                float(record["gradient_groups"][group][key]) for record in records
            )
            / len(records)
            for key in keys
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure independent diffusion and weighted Mean-AoD gradients."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--weights", choices=["online", "ema"], default="online")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batches <= 0:
        raise ValueError("batch-size and batches must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_kind") != "multipath_diffusion_scene_id_aod_conditioned":
        raise ValueError("checkpoint is not an AoD-conditioned scene-ID diffusion model")
    config = checkpoint["config"]
    plan = config.get("scene_id_experiment")
    if plan is None:
        raise ValueError("checkpoint has no embedded scene split")
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
    loader = DataLoader(
        datasets["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = MultipathDiffusionModel(
        config["multipath_diffusion"],
        SystemConfig.from_mapping(config["data"]["system"]),
        checkpoint["statistics"],
    ).to(device)
    state = (
        checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
        if args.weights == "ema"
        else checkpoint["model_state_dict"]
    )
    model.load_state_dict(state)
    model.train()
    names_and_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [item[0] for item in names_and_parameters]
    parameters = [item[1] for item in names_and_parameters]
    angle_weight = float(
        config["multipath_diffusion"]["aod_conditioning"].get("loss_weight", 0.0)
    )
    records: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(loader):
        if batch_number >= args.batches:
            break
        torch.manual_seed(args.seed + batch_number)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + batch_number)
        batch = move_to_device(batch, device)
        metrics = model(batch)
        weighted_aod = metrics["aod_condition_auxiliary_weighted"]
        if not weighted_aod.requires_grad or float(
            metrics["aod_condition_supervision_fraction"].detach()
        ) == 0.0:
            continue
        diffusion_gradients = _gradient_pair(
            metrics["diffusion_loss"], parameters, retain_graph=True
        )
        aod_gradients = _gradient_pair(weighted_aod, parameters, retain_graph=False)
        gradient_groups = _gradient_statistics(
            names, diffusion_gradients, aod_gradients
        )
        records.append(
            {
                "batch": batch_number,
                "samples": int(batch["conditions"].shape[0]),
                "diffusion_loss": float(metrics["diffusion_loss"].detach()),
                "aod_loss_unweighted": float(metrics["aod_condition_loss"].detach()),
                "aod_loss_weighted": float(weighted_aod.detach()),
                "aod_error_deg_low_noise": float(
                    metrics["aod_condition_error_deg"].detach()
                ),
                "aod_supervision_fraction": float(
                    metrics["aod_condition_supervision_fraction"].detach()
                ),
                "gradient_groups": gradient_groups,
            }
        )
    if not records:
        raise RuntimeError("no batch contained an active low-noise AoD supervision sample")

    aggregate = _mean_records(records)
    all_gradients = aggregate["gradient_groups"]["all"]
    denoiser = aggregate["gradient_groups"].get("denoiser", {})
    report = {
        "format_version": 1,
        "diagnostic": "aod_conditioned_loss_gradient_audit",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights": args.weights,
        "seed": args.seed,
        "requested_batches": args.batches,
        "evaluated_batches": len(records),
        "aod_loss_weight": angle_weight,
        "implementation_audit": {
            "aod_loss_input": "predicted_clean_x0",
            "aod_loss_uses_noisy_xt_directly": False,
            "aod_loss_is_differentiable": bool(
                float(all_gradients["weighted_aod_gradient_l2"]) > 0.0
            ),
            "aod_loss_reaches_denoiser": bool(
                float(denoiser.get("weighted_aod_gradient_l2", 0.0)) > 0.0
            ),
            "reported_training_error_scope": (
                "Only non-dropped samples whose diffusion timestep lies in the configured "
                "low-noise fraction; it is not final DDIM generation error."
            ),
        },
        "aggregate_mean": aggregate,
        "per_batch": records,
    }
    if not math.isfinite(float(all_gradients["weighted_aod_gradient_l2"])):
        raise FloatingPointError("AoD gradient audit produced a non-finite norm")
    atomic_json_dump(report, args.output)
    print(
        json.dumps(
            {
                "event": "aod_loss_gradient_audit_complete",
                "output": str(Path(args.output).resolve()),
                "evaluated_batches": len(records),
                "all_gradient_ratio": all_gradients[
                    "weighted_aod_to_diffusion_ratio"
                ],
                "denoiser_aod_gradient_l2": denoiser.get(
                    "weighted_aod_gradient_l2", 0.0
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
