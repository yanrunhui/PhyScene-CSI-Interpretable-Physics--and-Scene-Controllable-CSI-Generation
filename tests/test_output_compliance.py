from __future__ import annotations

import torch

from physics_csi.evaluation.output_compliance import (
    ComplianceTargets,
    audit_generated_output,
    targets_from_artifact,
)
from physics_csi.generation_mode import AoDControlMode
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig


def _artifact(aod_mode: AoDControlMode) -> dict[str, object]:
    powers = torch.tensor([[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]])
    delays = torch.tensor([[0.0, 10.0, 30.0], [0.0, 10.0, 30.0]])
    mask = torch.ones_like(powers, dtype=torch.bool)
    direct = torch.tensor([[True, False, False], [True, False, False]])
    aod = torch.tensor([[30.0, -20.0, 90.0], [30.0, -20.0, 90.0]])
    payload = {
        "path_powers_linear": powers,
        "path_delays_ns": delays,
        "path_phases_rad": torch.zeros_like(powers),
        "path_mask": mask,
        "direct_path_mask": direct,
        "aoa_az_deg": torch.zeros_like(powers),
        "aoa_el_deg": torch.zeros_like(powers),
        "aod_az_deg": aod,
        "aod_el_deg": torch.zeros_like(powers),
    }
    system = SystemConfig(num_subcarriers=8, tx_array_shape=(2, 2))
    payload["csi"] = MIMOOFDMSynthesizer(system)(
        delays,
        powers,
        payload["path_phases_rad"],
        payload["aoa_az_deg"],
        payload["aoa_el_deg"],
        aod,
        payload["aod_el_deg"],
        mask,
    )
    return {
        "aod_mode": aod_mode.value,
        "map_id": 94,
        "path_count": 3,
        "generated": payload,
        "system_config": system.as_dict(),
    }


def test_strongest_compliance_passes_exact_final_payload() -> None:
    artifact = _artifact(AoDControlMode.STRONGEST_PATH)
    targets = ComplianceTargets(
        total_power_db=0.0,
        delay_spread_ns=85.0**0.5,
        k_factor_db=10.0 * torch.log10(torch.tensor(0.7 / 0.3)).item(),
        k_factor_valid=True,
        los_status=1,
        path_count=3,
        aod_mode=AoDControlMode.STRONGEST_PATH,
        target_aod_deg=30.0,
        map_id=94,
        expected_samples=2,
    )
    report = audit_generated_output(artifact, targets=targets)

    assert report["request_pass"]
    assert report["all_requested_conditions_pass"]
    assert report["csi_reconstruction"]["all_pass"]


def test_compliance_rejects_wrong_target_and_reports_each_sample() -> None:
    artifact = _artifact(AoDControlMode.STRONGEST_PATH)
    targets = ComplianceTargets(
        total_power_db=-3.0,
        delay_spread_ns=85.0**0.5,
        k_factor_db=10.0 * torch.log10(torch.tensor(0.7 / 0.3)).item(),
        k_factor_valid=True,
        los_status=1,
        path_count=3,
        aod_mode=AoDControlMode.STRONGEST_PATH,
        target_aod_deg=30.0,
        map_id=94,
        expected_samples=2,
    )
    report = audit_generated_output(artifact, targets=targets)

    assert not report["request_pass"]
    assert report["condition_pass_rate"]["total_power"] == 0.0
    assert len(report["per_sample"]) == 2
    assert not report["per_sample"][0]["all_requested_conditions_pass"]


def test_mean_mode_audits_power_weighted_circular_mean_not_top1() -> None:
    artifact = _artifact(AoDControlMode.MEAN_POWER_WEIGHTED)
    payload = artifact["generated"]
    radians = torch.deg2rad(payload["aod_az_deg"][0])
    weights = payload["path_powers_linear"][0]
    target = torch.rad2deg(
        torch.atan2((weights * radians.sin()).sum(), (weights * radians.cos()).sum())
    ).item()
    targets = ComplianceTargets(
        total_power_db=0.0,
        delay_spread_ns=85.0**0.5,
        k_factor_db=10.0 * torch.log10(torch.tensor(0.7 / 0.3)).item(),
        k_factor_valid=True,
        los_status=1,
        path_count=3,
        aod_mode=AoDControlMode.MEAN_POWER_WEIGHTED,
        target_aod_deg=target,
        map_id=94,
        expected_samples=2,
    )

    report = audit_generated_output(artifact, targets=targets)

    assert report["request_pass"]
    assert report["per_sample"][0]["actual"]["aod_deg"] != 30.0


def test_legacy_artifact_infers_strongest_mode_and_target() -> None:
    artifact = _artifact(AoDControlMode.STRONGEST_PATH)
    artifact.pop("aod_mode")
    artifact.update(
        generation_mode="six_condition_strongest_aod",
        conditions={
            "total_power_db": 0.0,
            "delay_spread_ns": 85.0**0.5,
            "k_factor_db": 10.0 * torch.log10(torch.tensor(0.7 / 0.3)).item(),
            "los_status": 1,
        },
        condition_validity=torch.ones(4, dtype=torch.bool),
        target_strongest_aod_az_deg=30.0,
    )

    targets = targets_from_artifact(artifact, expected_samples=2)

    assert targets.aod_mode is AoDControlMode.STRONGEST_PATH
    assert targets.target_aod_deg == 30.0
