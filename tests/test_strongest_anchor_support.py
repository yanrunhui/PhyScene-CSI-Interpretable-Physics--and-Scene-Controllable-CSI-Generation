from __future__ import annotations

import torch

from physics_csi.evaluation.aod_support import enforce_final_strongest_anchor
from physics_csi.evaluation.strongest_anchor_support import (
    classify_strongest_anchor_requests,
)


def _reference() -> dict[str, torch.Tensor]:
    samples = 10
    return {
        "conditions": torch.tensor([[-80.0, 20.0, 3.0, 1.0]]).expand(samples, -1),
        "condition_validity": torch.ones(samples, 4, dtype=torch.bool),
        "map_id": torch.tensor([94] * 8 + [12] * 2),
        "path_mask": torch.ones(samples, 2, dtype=torch.bool),
        "path_powers_linear": torch.tensor([[0.8, 0.2]]).expand(samples, -1),
        "aod_az_deg": torch.tensor([[30.0, -40.0]]).expand(samples, -1),
    }


def test_anchor_support_distinguishes_supported_counterfactual_and_conflict() -> None:
    result = classify_strongest_anchor_requests(
        _reference(),
        target_conditions=torch.tensor(
            [
                [-80.0, 20.0, 3.0, 1.0],
                [-80.0, 20.0, 3.0, 1.0],
                [-80.0, 20.0, 3.0, 1.0],
            ]
        ),
        target_validity=torch.ones(3, 4, dtype=torch.bool),
        target_map_ids=torch.tensor([94, 94, 94]),
        target_aod_deg=torch.tensor([30.0, 120.0, 120.0]),
        target_path_count=2,
        condition_std=torch.tensor([10.0, 10.0, 5.0, 1.0]),
        minimum_map_support=8,
        known_los_direct_aod_deg=torch.tensor([30.0, 120.0, 30.0]),
        los_direct_tolerance_deg=10.0,
    )
    assert result.labels == ("A", "B", "C")
    assert result.map_support_count.tolist() == [8, 0, 0]
    assert result.physically_feasible.tolist() == [True, True, False]
    assert "known_los_direct_geometry" in result.conflict_reasons[2][0]


def test_anchor_is_reasserted_after_power_stage_changes_top1() -> None:
    payload = {
        "path_mask": torch.ones(1, 2, dtype=torch.bool),
        "path_powers_linear": torch.tensor([[0.2, 0.8]]),
        "aod_az_deg": torch.tensor([[30.0, 80.0]]),
    }
    result = enforce_final_strongest_anchor(payload, torch.tensor([30.0]))
    assert result["strongest_anchor_path_index"].tolist() == [1]
    assert result["aod_az_deg"].tolist() == [[30.0, 30.0]]
    assert result["strongest_anchor_pre_repair_aod_deg"].tolist() == [80.0]


def test_anchor_swap_preserves_angle_multiset_without_requiring_unique_target() -> None:
    payload = {
        "path_mask": torch.ones(1, 3, dtype=torch.bool),
        "path_powers_linear": torch.tensor([[0.30, 0.45, 0.25]]),
        "aod_az_deg": torch.tensor([[30.0, 65.0, -20.0]]),
        "strongest_anchor_path_index": torch.tensor([0]),
        "direct_path_mask": torch.tensor([[True, False, False]]),
    }
    result = enforce_final_strongest_anchor(
        payload, torch.tensor([30.0]), strategy="swap"
    )
    assert result["strongest_anchor_previous_path_index"].tolist() == [0]
    assert result["strongest_anchor_path_index"].tolist() == [1]
    assert result["strongest_anchor_identity_changed"].tolist() == [True]
    assert result["aod_az_deg"].tolist() == [[65.0, 30.0, -20.0]]
    assert result["strongest_anchor_added_target_match_count"].tolist() == [0]
    assert result["strongest_anchor_rebind_involves_direct_path"].tolist() == [True]
    assert torch.allclose(
        result["strongest_anchor_direct_aod_change_deg"], torch.tensor([35.0])
    )
