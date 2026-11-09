from __future__ import annotations

import torch

from physics_csi.evaluation.aod_support import strongest_aod
from physics_csi.evaluation.filtered_candidates import (
    circular_angle_bin,
    select_first_eligible_candidates,
)
from scripts.generate_dual_mode_aod_csi import _refresh_strongest_aod_fields


def test_filter_selects_first_eligible_without_error_ranking() -> None:
    angles = torch.tensor([[89.0], [80.5], [90.0], [95.0], [70.0]])
    payload = {
        "path_mask": torch.ones(5, 1, dtype=torch.bool),
        "path_powers_linear": torch.ones(5, 1),
        "aod_az_deg": angles,
    }
    selection = select_first_eligible_candidates(
        payload,
        torch.tensor([90.0]),
        queries=1,
        candidates_per_query=5,
        outputs_per_query=3,
        tolerance_deg=10.0,
        minimum_concentration=0.0,
    )
    # Candidate 2 is closest, but generation-order acceptance is 0, 1, 2.
    assert selection.selected_flat_indices.tolist() == [[0, 1, 2]]
    assert bool(selection.success.item())


def test_filter_reports_failed_query_without_repeating_as_success() -> None:
    payload = {
        "path_mask": torch.ones(4, 1, dtype=torch.bool),
        "path_powers_linear": torch.ones(4, 1),
        "aod_az_deg": torch.tensor([[90.0], [120.0], [130.0], [140.0]]),
    }
    selection = select_first_eligible_candidates(
        payload,
        torch.tensor([90.0]),
        queries=1,
        candidates_per_query=4,
        outputs_per_query=2,
        tolerance_deg=10.0,
        minimum_concentration=0.0,
    )
    assert not bool(selection.success.item())
    assert int(selection.eligible_count.item()) == 1
    assert torch.isnan(selection.selected_error_deg).all()


def test_strongest_filter_recomputes_identity_from_final_power() -> None:
    payload = {
        "path_mask": torch.ones(2, 2, dtype=torch.bool),
        "path_powers_linear": torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
        "aod_az_deg": torch.tensor([[0.0, 90.0], [0.0, 90.0]]),
    }

    selection = select_first_eligible_candidates(
        payload,
        torch.tensor([90.0]),
        queries=1,
        candidates_per_query=2,
        outputs_per_query=1,
        tolerance_deg=1.0,
        mode="strongest",
    )

    # A DP-like final power update makes path 1 strongest only in candidate 1.
    assert selection.selected_flat_indices.tolist() == [[1]]
    assert selection.eligible.tolist() == [[False, True]]


def test_strongest_measurement_uses_first_argmax_and_reports_tie() -> None:
    payload = {
        "path_mask": torch.tensor([[True, True, False]]),
        "path_powers_linear": torch.tensor([[0.5, 0.5, 100.0]]),
        "aod_az_deg": torch.tensor([[30.0, 80.0, -120.0]]),
    }

    measured = strongest_aod(payload, near_tie_gap_db=0.5)

    assert measured.path_index.tolist() == [0]
    assert measured.angle_deg.tolist() == [30.0]
    assert measured.top2_power_gap_db.tolist() == [0.0]
    assert measured.near_tie.tolist() == [True]
    assert measured.valid.tolist() == [True]


def test_strongest_filter_can_exclude_near_ties() -> None:
    payload = {
        "path_mask": torch.ones(2, 2, dtype=torch.bool),
        "path_powers_linear": torch.tensor([[0.5, 0.5], [0.8, 0.2]]),
        "aod_az_deg": torch.tensor([[45.0, 90.0], [45.0, 90.0]]),
    }

    selection = select_first_eligible_candidates(
        payload,
        torch.tensor([45.0]),
        queries=1,
        candidates_per_query=2,
        outputs_per_query=1,
        tolerance_deg=1.0,
        mode="strongest",
        exclude_strongest_near_ties=True,
        strongest_near_tie_gap_db=0.5,
    )

    assert selection.selected_flat_indices.tolist() == [[1]]
    assert selection.eligible.tolist() == [[False, True]]


def test_dp_stage_refreshes_saved_strongest_derivatives() -> None:
    dp_payload = {
        "path_mask": torch.tensor([[True, True]]),
        "path_powers_linear": torch.tensor([[0.1, 0.9]]),
        "aod_az_deg": torch.tensor([[10.0, 70.0]]),
        # Simulate stale fields copied before DP changed the power ordering.
        "actual_strongest_path_index": torch.tensor([0]),
        "actual_strongest_aod_az_deg": torch.tensor([10.0]),
    }

    refreshed = _refresh_strongest_aod_fields(dp_payload, 75.0)

    assert refreshed["actual_strongest_path_index"].tolist() == [1]
    assert refreshed["actual_strongest_aod_az_deg"].tolist() == [70.0]
    assert refreshed["strongest_aod_circular_error_deg"].tolist() == [5.0]


def test_circular_angle_bins_wrap_boundary_consistently() -> None:
    index, center = circular_angle_bin(
        torch.tensor([-180.0, 179.0, -1.0, 1.0]), width_deg=30.0
    )
    assert index.tolist() == [0, 11, 5, 6]
    assert center.tolist() == [-165.0, 165.0, -15.0, 15.0]
