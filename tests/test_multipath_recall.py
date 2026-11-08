from __future__ import annotations

import torch

from physics_csi.evaluation.multipath_recall import (
    FeatureStandardizer,
    manifold_metrics,
    multipath_view_features,
)


def _payload(shift: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "path_mask": torch.ones(3, 2, dtype=torch.bool),
        "path_delays_ns": torch.tensor([[0.0, 10.0], [0.0, 20.0], [0.0, 30.0]]),
        "path_powers_linear": torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]]),
        "path_phases_rad": torch.zeros(3, 2),
        "aoa_az_deg": torch.zeros(3, 2),
        "aoa_el_deg": torch.zeros(3, 2),
        "aod_az_deg": torch.tensor(
            [[10.0, 30.0], [20.0, 50.0], [-20.0, 20.0]]
        )
        + shift,
        "aod_el_deg": torch.zeros(3, 2),
    }


def test_self_centered_view_is_invariant_to_common_aod_rotation() -> None:
    first = multipath_view_features(_payload(), view="self_centered")
    second = multipath_view_features(_payload(75.0), view="self_centered")
    assert torch.allclose(first, second, atol=1e-5, rtol=0.0)


def test_absolute_view_retains_common_aod_rotation() -> None:
    first = multipath_view_features(_payload(), view="absolute")
    second = multipath_view_features(_payload(75.0), view="absolute")
    assert not torch.allclose(first, second)


def test_identical_manifolds_have_full_precision_recall_and_coverage() -> None:
    features = multipath_view_features(_payload(), view="absolute")
    standardizer = FeatureStandardizer.fit(features)
    normalized = standardizer.transform(features)
    result = manifold_metrics(normalized, normalized, requested_k=1)
    assert result.metrics["precision"] == 1.0
    assert result.metrics["recall"] == 1.0
    assert result.metrics["coverage"] == 1.0
