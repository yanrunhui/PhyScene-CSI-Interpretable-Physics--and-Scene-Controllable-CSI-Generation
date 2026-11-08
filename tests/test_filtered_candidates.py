from __future__ import annotations

import torch

from physics_csi.evaluation.filtered_candidates import (
    circular_angle_bin,
    select_first_eligible_candidates,
)


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


def test_circular_angle_bins_wrap_boundary_consistently() -> None:
    index, center = circular_angle_bin(
        torch.tensor([-180.0, 179.0, -1.0, 1.0]), width_deg=30.0
    )
    assert index.tolist() == [0, 11, 5, 6]
    assert center.tolist() == [-165.0, 165.0, -15.0, 15.0]
