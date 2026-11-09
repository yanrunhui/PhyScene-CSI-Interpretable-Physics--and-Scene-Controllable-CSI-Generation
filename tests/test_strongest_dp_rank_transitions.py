from __future__ import annotations

import pytest
import torch

from scripts.diagnose_strongest_dp_rank_transitions import (
    _measure,
    _summary,
    _transition,
)


def _payload(powers: torch.Tensor) -> dict[str, torch.Tensor]:
    samples = powers.shape[0]
    conditions = torch.tensor(
        [[-80.0, 20.0, 0.0, 0.0]] * samples, dtype=torch.float32
    )
    return {
        "path_mask": torch.ones(samples, 2, dtype=torch.bool),
        "path_powers_linear": powers,
        "aod_az_deg": torch.tensor(
            [[70.0, 31.0], [30.0, 80.0], [27.0, -90.0]]
        ),
        "conditions": conditions,
        "actual_conditions": conditions.clone(),
        "condition_validity": torch.ones(samples, 4, dtype=torch.bool),
    }


def test_dp_transition_reports_beneficial_and_harmful_rank_changes() -> None:
    target = torch.tensor([30.0, 30.0, 30.0])
    raw = _measure(
        _payload(torch.tensor([[0.8, 0.2], [0.8, 0.2], [0.7, 0.3]])),
        target,
    )
    final = _measure(
        _payload(torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])),
        target,
    )
    indices = torch.arange(3)
    transition = _transition(raw, final, indices)
    assert transition["strongest_identity_changed_rate"] == pytest.approx(2 / 3)
    assert transition["hit_at_10_beneficial_change_rate"] == pytest.approx(1 / 3)
    assert transition["hit_at_10_harmful_change_rate"] == pytest.approx(1 / 3)
    assert transition["hit_at_10_net_change"] == pytest.approx(0.0)
    assert transition["target_nearest_top1_net_change"] == pytest.approx(0.0)

    raw_summary = _summary(raw, indices)
    final_summary = _summary(final, indices)
    assert raw_summary["target_nearest_path_top1_rate"] == pytest.approx(2 / 3)
    assert final_summary["target_nearest_path_top1_rate"] == pytest.approx(2 / 3)
    assert final_summary["physical_condition_mae"]["power_db"] == pytest.approx(0.0)
