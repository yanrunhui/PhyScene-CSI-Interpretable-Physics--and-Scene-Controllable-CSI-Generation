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

from physics_csi.evaluation.output_compliance import (
    ComplianceTolerances,
    audit_generated_output,
    override_targets,
    targets_from_artifact,
)
from physics_csi.physics.synthesizer import SystemConfig
from physics_csi.utils.io import atomic_json_dump


def _load_system_config(
    artifact: Mapping[str, Any], checkpoint: str | None
) -> SystemConfig | Mapping[str, Any] | None:
    configured = artifact.get("system_config")
    if configured is not None:
        return configured
    checkpoint_path = checkpoint
    if checkpoint_path is None:
        paths = artifact.get("checkpoint_paths", {})
        if isinstance(paths, Mapping):
            for key in (
                "selected_aod_b3",
                "strongest_anchor_b3",
                "mean_aod_b3",
                "aod_conditioned_b3",
                "b3",
            ):
                candidate = paths.get(key)
                if candidate:
                    checkpoint_path = candidate
                    break
    if checkpoint_path is None:
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        return SystemConfig.from_mapping(payload["config"]["data"]["system"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"checkpoint does not contain a valid system config: {checkpoint_path}"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute every final physical condition and AoD from a generated CSI "
            "artifact, then require every output sample to satisfy the user request."
        )
    )
    parser.add_argument("--artifact", required=True, help="Generated .pt artifact")
    parser.add_argument("--output", help="Audit JSON; defaults beside the artifact")
    parser.add_argument(
        "--checkpoint",
        help="AoD B3 checkpoint for old artifacts without an embedded system config",
    )
    parser.add_argument("--expected-map-id", type=int)
    parser.add_argument("--expected-samples", type=int, default=8)
    parser.add_argument("--expected-total-power-db", type=float)
    parser.add_argument("--expected-delay-spread-ns", type=float)
    parser.add_argument("--expected-k-factor-db", type=float)
    parser.add_argument(
        "--expected-k-factor-validity",
        choices=["valid", "invalid"],
        help="Override whether K-factor is an applicable requested condition",
    )
    parser.add_argument("--expected-los-status", type=int, choices=[0, 1])
    parser.add_argument("--expected-path-count", type=int)
    parser.add_argument(
        "--expected-aod-mode",
        choices=["mean_power_weighted", "strongest_path"],
    )
    parser.add_argument("--expected-aod-deg", type=float)
    parser.add_argument("--power-tolerance-db", type=float, default=0.1)
    parser.add_argument("--ds-absolute-tolerance-ns", type=float, default=0.5)
    parser.add_argument("--ds-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--k-tolerance-db", type=float, default=0.2)
    parser.add_argument("--aod-tolerance-deg", type=float, default=10.0)
    parser.add_argument(
        "--csi-reconstruction-tolerance", type=float, default=1e-5
    )
    parser.add_argument(
        "--skip-csi-reconstruction",
        action="store_true",
        help="Check final path parameters only; not recommended for final acceptance",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit with code 2 when any requested output fails",
    )
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, Mapping):
        parser.error("artifact must contain a mapping")

    target_source = dict(artifact)
    if args.expected_aod_mode is not None:
        target_source["aod_mode"] = args.expected_aod_mode
    if args.expected_aod_deg is not None:
        if args.expected_aod_mode == "mean_power_weighted":
            target_source["target_mean_aod_az_deg"] = args.expected_aod_deg
        elif args.expected_aod_mode == "strongest_path":
            target_source["target_strongest_aod_az_deg"] = args.expected_aod_deg
        else:
            # The inferred mode will read the corresponding field.
            target_source["target_mean_aod_az_deg"] = args.expected_aod_deg
            target_source["target_strongest_aod_az_deg"] = args.expected_aod_deg
    targets = targets_from_artifact(
        target_source, expected_samples=args.expected_samples
    )
    k_override = args.expected_k_factor_db
    k_valid_override = (
        args.expected_k_factor_validity == "valid"
        if args.expected_k_factor_validity is not None
        else None
    )
    targets = override_targets(
        targets,
        map_id=args.expected_map_id,
        total_power_db=args.expected_total_power_db,
        delay_spread_ns=args.expected_delay_spread_ns,
        k_factor_db=k_override,
        k_factor_valid=k_valid_override,
        los_status=args.expected_los_status,
        path_count=args.expected_path_count,
        aod_mode=args.expected_aod_mode,
        target_aod_deg=args.expected_aod_deg,
    )
    tolerances = ComplianceTolerances(
        total_power_abs_db=args.power_tolerance_db,
        delay_spread_abs_ns=args.ds_absolute_tolerance_ns,
        delay_spread_relative=args.ds_relative_tolerance,
        k_factor_abs_db=args.k_tolerance_db,
        aod_abs_deg=args.aod_tolerance_deg,
        csi_reconstruction_relative=args.csi_reconstruction_tolerance,
    )
    system_config = None
    if not args.skip_csi_reconstruction:
        system_config = _load_system_config(artifact, args.checkpoint)
        if system_config is None:
            parser.error(
                "CSI reconstruction needs --checkpoint for this old artifact, or use "
                "--skip-csi-reconstruction for a path-only diagnostic"
            )

    report = audit_generated_output(
        artifact,
        targets=targets,
        tolerances=tolerances,
        system_config=system_config,
        check_csi_reconstruction=not args.skip_csi_reconstruction,
    )
    report["artifact"] = str(artifact_path.resolve())
    output = (
        Path(args.output)
        if args.output
        else artifact_path.with_name(f"{artifact_path.stem}_condition_audit.json")
    )
    atomic_json_dump(report, output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"audit_report={output.resolve()}", flush=True)
    if args.fail_on_violation and not report["request_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
