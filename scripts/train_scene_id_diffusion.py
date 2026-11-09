#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.config import deep_merge, load_yaml
from physics_csi.data.aod_conditioned import make_aod_conditioned_datasets
from physics_csi.data.diffusion_stats import load_or_compute_diffusion_statistics
from physics_csi.data.scene_id import (
    load_seen_map_spatial_plan,
    make_scene_id_dataloaders,
    make_scene_id_datasets,
)
from physics_csi.models.diffusion.multipath_diffusion import MultipathDiffusionModel
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.training.diffusion_trainer import DiffusionTrainer
from physics_csi.training.trainer import move_to_device, resolve_device
from physics_csi.utils.seed import seed_everything


def _load_compatible_weights(
    model: MultipathDiffusionModel, checkpoint_path: str
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    five_condition_supported = {
        "multipath_diffusion",
        "multipath_diffusion_seen_map_baseline",
        "multipath_diffusion_scene_id",
        "multipath_diffusion_scene_id_spatial",
    }
    supported = five_condition_supported | {
        "multipath_diffusion_scene_residual",
        "multipath_diffusion_scene_residual_spatial",
        "multipath_diffusion_scene_id_aod_conditioned",
        "multipath_diffusion_aod_conditioned",
        "multipath_diffusion_scene_id_strongest_aod_conditioned",
        "multipath_diffusion_strongest_aod_conditioned",
    }
    condition_type = str(model.aod_conditioning.get("condition_type", "mean"))
    if condition_type == "strongest":
        if model.strongest_anchor_enabled:
            # The first run starts from the five-condition B3.  A later Anchor
            # continuation may reuse an Anchor checkpoint, but an absolute-AoD
            # Strongest checkpoint is intentionally excluded because its
            # denoiser has a different output-coordinate contract.
            supported = (
                {
                    "multipath_diffusion_scene_id",
                    "multipath_diffusion_scene_id_spatial",
                    "multipath_diffusion_scene_id_strongest_aod_anchor",
                }
                if model.scene_conditioning_enabled
                else {
                    "multipath_diffusion",
                    "multipath_diffusion_seen_map_baseline",
                    "multipath_diffusion_strongest_aod_anchor",
                }
            )
        else:
            # The first Strongest-AoD experiment starts from the original
            # five-condition B3 rather than inherit a Mean-AoD fine-tune.
            supported = (
                {
                    "multipath_diffusion_scene_id",
                    "multipath_diffusion_scene_id_spatial",
                }
                if model.scene_conditioning_enabled
                else {
                    "multipath_diffusion",
                    "multipath_diffusion_seen_map_baseline",
                }
            )
    if checkpoint.get("model_kind") not in supported:
        if condition_type == "strongest":
            expected = (
                "an original five-condition B3/B3-S-ID or Strongest Anchor checkpoint"
                if model.strongest_anchor_enabled
                else "an original five-condition B3/B3-S-ID checkpoint"
            )
            raise ValueError(
                f"Strongest-AoD fine-tuning must initialize from {expected}"
            )
        raise ValueError("--initialize-from points to an unsupported checkpoint")
    source = checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "loaded_tensors": len(compatible),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "source_model_kind": checkpoint.get("model_kind"),
        "note": "optimizer, scheduler and source history are intentionally not restored",
    }


def _model_kind(model: MultipathDiffusionModel) -> str:
    if model.aod_conditioning_enabled:
        condition_type = str(model.aod_conditioning.get("condition_type", "mean"))
        if condition_type == "strongest":
            if model.strongest_anchor_enabled:
                return (
                    "multipath_diffusion_scene_id_strongest_aod_anchor"
                    if model.scene_conditioning_enabled
                    else "multipath_diffusion_strongest_aod_anchor"
                )
            return (
                "multipath_diffusion_scene_id_strongest_aod_conditioned"
                if model.scene_conditioning_enabled
                else "multipath_diffusion_strongest_aod_conditioned"
            )
        return (
            "multipath_diffusion_scene_id_aod_conditioned"
            if model.scene_conditioning_enabled
            else "multipath_diffusion_aod_conditioned"
        )
    if model.scene_adapter_enabled:
        return (
            "multipath_diffusion_scene_residual_spatial"
            if model.spatial_supervision_enabled
            else "multipath_diffusion_scene_residual"
        )
    if model.scene_conditioning_enabled:
        return (
            "multipath_diffusion_scene_id_spatial"
            if model.spatial_supervision_enabled
            else "multipath_diffusion_scene_id"
        )
    return "multipath_diffusion_seen_map_baseline"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train B3-S-ID or its matched physics-only seen-map baseline."
    )
    parser.add_argument(
        "--config", default="configs/multipath_diffusion_scene_id.yaml"
    )
    parser.add_argument(
        "--scene-plan",
        default=None,
        help="Plan produced by create_scene_id_split.py; required for a new run.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the training seed for a new run; ignored seeds are never changed on resume.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one forward/backward batch and exit without writing a checkpoint.",
    )
    parser.add_argument(
        "--dry-run-batch-size",
        type=int,
        default=32,
        help="Small sequential validation batch used only by --dry-run.",
    )
    parser.add_argument(
        "--initialize-from-b3",
        default=None,
        help="Deprecated alias for --initialize-from.",
    )
    parser.add_argument(
        "--initialize-from",
        default=None,
        help=(
            "Load shape-compatible EMA weights from B3, B3-S-ID, E1 or S1, "
            "but start a new optimizer and history."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs for a new run."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate for a new run.",
    )
    args = parser.parse_args()
    initialize_from = args.initialize_from or args.initialize_from_b3
    if args.initialize_from and args.initialize_from_b3:
        raise ValueError("use only one initialization argument")
    if args.resume and initialize_from:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    if args.resume and args.seed is not None:
        raise ValueError("--seed cannot be changed while resuming a checkpoint")
    if args.resume and (args.epochs is not None or args.learning_rate is not None):
        raise ValueError("epochs and learning rate are restored when resuming")
    resume_checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    initialization_checkpoint = (
        torch.load(initialize_from, map_location="cpu", weights_only=False)
        if initialize_from
        else None
    )
    config = copy.deepcopy(
        resume_checkpoint["config"]
        if resume_checkpoint is not None
        else deep_merge(load_yaml(ROOT / "configs/data.yaml"), load_yaml(args.config))
    )
    if resume_checkpoint is None:
        if not args.scene_plan:
            raise ValueError("a new run requires --scene-plan")
        plan = load_seen_map_spatial_plan(args.scene_plan)
        config["scene_id_experiment"] = plan
        model_config = config["multipath_diffusion"]
        for key in ("scene_conditioning", "scene_residual_adapter"):
            scene_config = model_config.setdefault(key, {})
            if bool(scene_config.get("enabled", False)):
                scene_config["map_id_to_index"] = dict(plan["map_id_to_index"])
        config["statistics_path"] = str(
            Path("artifacts/statistics")
            / f"diffusion_seen_map_{plan['fingerprint'][:12]}.json"
        )
    else:
        if args.scene_plan:
            external = load_seen_map_spatial_plan(args.scene_plan)
            embedded = config.get("scene_id_experiment")
            if embedded is None or external["fingerprint"] != embedded["fingerprint"]:
                raise ValueError("external scene plan does not match resumed checkpoint")
        plan = config.get("scene_id_experiment")
        if plan is None:
            raise ValueError("checkpoint has no embedded scene-ID experiment plan")
    if args.checkpoint_dir:
        config["train"]["checkpoint_dir"] = args.checkpoint_dir
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("epochs must be positive")
        config["train"]["epochs"] = args.epochs
    if args.learning_rate is not None:
        if args.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        config["train"]["learning_rate"] = args.learning_rate
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    train = config["train"]
    seed_everything(int(train["seed"]))
    prepared_root = config["data"]["prepared_root"]
    datasets = make_scene_id_datasets(prepared_root, plan)
    aod_data_config = dict(config.get("aod_conditioned_data", {}))
    aod_model_config = dict(
        config["multipath_diffusion"].get("aod_conditioning", {})
    )
    if bool(aod_model_config.get("enabled", False)):
        model_condition_type = str(aod_model_config.get("condition_type", "mean"))
        data_condition_type = str(
            aod_data_config.get("condition_type", model_condition_type)
        )
        if data_condition_type != model_condition_type:
            raise ValueError(
                "aod_conditioned_data.condition_type must match "
                "multipath_diffusion.aod_conditioning.condition_type"
            )
        aod_data_config["condition_type"] = data_condition_type
        config["aod_conditioned_data"] = aod_data_config
        datasets = make_aod_conditioned_datasets(
            datasets,
            condition_type=data_condition_type,
            path_count=(
                int(aod_data_config["path_count"])
                if aod_data_config.get("path_count") is not None
                else None
            ),
            minimum_concentration=float(
                aod_data_config.get("minimum_concentration", 0.10)
            ),
            reject_low_concentration=bool(
                aod_data_config.get("reject_low_concentration", True)
            ),
        )
        config["aod_dataset_audit"] = {
            split: dataset.audit() for split, dataset in datasets.items()
        }
        print(
            json.dumps(
                {"event": "aod_dataset_audit", **config["aod_dataset_audit"]},
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    system = SystemConfig.from_mapping(config["data"]["system"])
    reuse_initialization_statistics = bool(
        config["multipath_diffusion"]
        .get("aod_conditioning", {})
        .get("reuse_initialization_statistics", True)
    )
    if resume_checkpoint is not None:
        statistics = resume_checkpoint["statistics"]
    elif reuse_initialization_statistics and initialization_checkpoint is not None:
        statistics = initialization_checkpoint["statistics"]
        config["statistics_source"] = {
            "kind": "initialization_checkpoint",
            "checkpoint": str(Path(initialize_from).resolve()),
            "reason": "preserve B3 normalization during AoD-conditioned fine-tuning",
        }
    else:
        statistics = load_or_compute_diffusion_statistics(
            config["statistics_path"],
            prepared_root,
            system,
            batch_size=int(config.get("statistics_batch_size", 512)),
            num_workers=int(train["num_workers"]),
            dataset=datasets["train"],
            source_identity=f"seen_map_spatial:{plan['fingerprint']}",
        )
    if args.dry_run_batch_size <= 0:
        raise ValueError("--dry-run-batch-size must be positive")
    loader_batch_size = (
        min(int(train["batch_size"]), args.dry_run_batch_size)
        if args.dry_run
        else int(train["batch_size"])
    )
    loader_workers = 0 if args.dry_run else int(train["num_workers"])
    loaders = make_scene_id_dataloaders(
        datasets,
        batch_size=loader_batch_size,
        num_workers=loader_workers,
        training_block_shuffle_size=(
            int(aod_data_config.get("training_shuffle_block_size", 2048))
            if bool(
                config["multipath_diffusion"]
                .get("aod_conditioning", {})
                .get("enabled", False)
            )
            else None
        ),
        training_seed=int(train["seed"]),
    )
    model = MultipathDiffusionModel(
        config["multipath_diffusion"], system, statistics
    )
    if initialize_from:
        assert initialization_checkpoint is not None
        source_checkpoint = initialization_checkpoint
        source_plan = source_checkpoint.get("config", {}).get("scene_id_experiment")
        if source_plan is not None and source_plan.get("fingerprint") != plan["fingerprint"]:
            raise ValueError("initialization checkpoint uses a different scene split")
        config["initialization"] = _load_compatible_weights(model, initialize_from)
        print(json.dumps(config["initialization"], indent=2), flush=True)
    model_kind = _model_kind(model)
    if args.dry_run:
        device = resolve_device(str(train.get("device", "auto")))
        model = model.to(device).train()
        # The validation view is ordered by persisted index, avoiding the
        # random-shard I/O amplification of the shuffled training loader.
        batch = move_to_device(next(iter(loaders["validation"])), device)
        metrics = model(batch)
        if not all(torch.isfinite(value).all() for value in metrics.values()):
            raise FloatingPointError("dry run produced a non-finite metric")
        metrics["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(train.get("gradient_clip_norm", 1.0)),
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("dry run produced a non-finite gradient norm")
        print(
            json.dumps(
                {
                    "event": "dry_run_passed",
                    "model_kind": model_kind,
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    "total_parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "gradient_norm": float(gradient_norm),
                    "metrics": {
                        key: float(value.detach()) for key, value in metrics.items()
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    trainer = DiffusionTrainer(model, train, config, statistics, model_kind)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit(loaders)


if __name__ == "__main__":
    main()
