from __future__ import annotations

import torch

from physics_csi.evaluation.aod_support import (
    circular_distance_degrees,
    conditioned_aod_measurement,
    local_support_counts,
    match_independent_supported_pairs,
)
from scripts.evaluate_aod_conditional_fidelity import _rank_selection


def _payload(samples: int = 12) -> dict[str, torch.Tensor]:
    angles = torch.linspace(-5.0, 5.0, samples)
    mask = torch.ones(samples, 2, dtype=torch.bool)
    power = torch.tensor([0.7, 0.3]).expand(samples, -1).clone()
    return {
        "conditions": torch.stack(
            [
                torch.zeros(samples),
                torch.linspace(9.5, 10.5, samples),
                torch.zeros(samples),
                torch.zeros(samples),
            ],
            dim=-1,
        ),
        "condition_validity": torch.tensor([1, 1, 0, 1], dtype=torch.bool)
        .expand(samples, -1)
        .clone(),
        "map_id": torch.full((samples,), 10, dtype=torch.long),
        "path_mask": mask,
        "path_powers_linear": power,
        "aod_az_deg": torch.stack([angles - 1.0, angles + 2.0], dim=-1),
    }


def test_circular_distance_wraps_at_180_degrees() -> None:
    distance = circular_distance_degrees(
        torch.tensor([179.0, -179.0]), torch.tensor([-179.0, 179.0])
    )
    assert torch.allclose(distance, torch.tensor([2.0, 2.0]), atol=1e-4)


def test_support_and_independent_pairing() -> None:
    payload = _payload()
    support = local_support_counts(
        payload,
        condition_std=torch.ones(4),
        maximum_condition_distance=2.0,
        maximum_angle_distance_deg=20.0,
    )
    assert int(support.min()) == 11
    query, donor, _, audit = match_independent_supported_pairs(
        payload,
        condition_std=torch.ones(4),
        maximum_condition_distance=2.0,
        maximum_angle_distance_deg=20.0,
        minimum_support=4,
        path_count=2,
        maximum_pairs_per_map_los=6,
        seed=7,
    )
    assert query.numel() == donor.numel() == 6
    assert not bool(torch.isin(query, donor).any())
    assert audit["pairs"] == 6


def test_strongest_support_uses_final_power_path_and_excludes_near_ties() -> None:
    payload = _payload(3)
    payload["path_powers_linear"] = torch.tensor(
        [[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]
    )
    payload["aod_az_deg"] = torch.tensor(
        [[-40.0, 20.0], [-10.0, 70.0], [5.0, 80.0]]
    )
    angle, valid, gap = conditioned_aod_measurement(
        payload, condition_type="strongest"
    )
    assert torch.allclose(angle[[0, 2]], torch.tensor([-40.0, 80.0]))
    assert valid.tolist() == [True, False, True]
    assert torch.isnan(angle[1])
    assert float(gap[0]) > 0.5


def test_empty_los_macro_returns_unavailable_selection() -> None:
    selection = _rank_selection({})
    assert selection["selected_label"] is None
    assert selection["gate_status"] == "unavailable_no_supported_map_groups"
