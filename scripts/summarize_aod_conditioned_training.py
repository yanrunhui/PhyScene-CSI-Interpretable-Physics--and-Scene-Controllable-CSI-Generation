#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AoD-conditioned B3 training.")
    parser.add_argument("checkpoint")
    parser.add_argument("--generation-artifact", default=None)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    history = checkpoint.get("history", [])
    best = min(history, key=lambda row: row["validation"]["loss"]) if history else None
    config = checkpoint["config"]
    generation = None
    if args.generation_artifact:
        artifact = torch.load(
            args.generation_artifact, map_location="cpu", weights_only=False
        )
        generated = artifact["generated"]
        angle_error = generated["mean_aod_circular_error_deg"].float()
        target = generated["target_conditions"].float()
        actual = generated["actual_conditions"].float()
        validity = (
            generated["target_condition_validity"].bool()
            & generated["actual_condition_validity"].bool()
        )
        condition_names = ("power_db", "delay_spread_ns", "k_factor_db", "los")
        generation = {
            "samples": int(angle_error.numel()),
            "mean_aod_error_deg": {
                "mean": float(angle_error.mean()),
                "median": float(torch.quantile(angle_error, 0.50)),
                "p95": float(torch.quantile(angle_error, 0.95)),
                "maximum": float(angle_error.max()),
            },
            "actual_mean_aod_concentration": {
                "mean": float(
                    generated["actual_mean_aod_concentration"].float().mean()
                ),
                "p05": float(
                    torch.quantile(
                        generated["actual_mean_aod_concentration"].float(), 0.05
                    )
                ),
            },
            "condition_absolute_error_mean": {
                name: (
                    float(
                        (actual[:, index] - target[:, index])
                        .abs()[validity[:, index]]
                        .mean()
                    )
                    if bool(validity[:, index].any())
                    else None
                )
                for index, name in enumerate(condition_names)
            },
            "path_count_exact_rate": float(
                (generated["path_count"] == artifact["forced_path_count"])
                .float()
                .mean()
            ),
            "constraint_feasible_rate": float(
                generated["constraint_feasible"].float().mean()
            ),
        }
    print(
        json.dumps(
            {
                "model_kind": checkpoint.get("model_kind"),
                "training_seed": config["train"]["seed"],
                "checkpoint_epoch": checkpoint.get("epoch"),
                "best_validation": checkpoint.get("best_validation"),
                "best_epoch_record": best,
                "aod_conditioning": config["multipath_diffusion"].get(
                    "aod_conditioning"
                ),
                "aod_conditioned_data": config.get("aod_conditioned_data"),
                "aod_dataset_audit": config.get("aod_dataset_audit"),
                "initialization": config.get("initialization"),
                "statistics_source": config.get("statistics_source"),
                "generation_smoke": generation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
