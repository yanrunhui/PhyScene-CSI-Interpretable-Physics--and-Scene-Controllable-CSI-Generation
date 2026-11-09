from __future__ import annotations

import torch

from physics_csi.evaluation.diversity_selection import (
    select_mild_diversity_candidates,
    select_random_candidates,
)


def test_mild_diversity_selection_is_unique_and_eligible() -> None:
    features = torch.tensor(
        [[[0.0], [0.1], [0.2], [2.0], [2.1], [9.0]]]
    )
    eligible = torch.tensor([[True, True, True, True, True, False]])
    selection = select_mild_diversity_candidates(
        features,
        eligible,
        outputs_per_query=3,
        diversity_strength=0.35,
        generator=torch.Generator().manual_seed(7),
    )
    assert bool(selection.success[0])
    assert torch.unique(selection.local_indices[0]).numel() == 3
    assert bool(eligible[0, selection.local_indices[0]].all())


def test_random_selection_is_reproducible_without_replacement() -> None:
    eligible = torch.ones(2, 8, dtype=torch.bool)
    first = select_random_candidates(
        eligible,
        outputs_per_query=4,
        generator=torch.Generator().manual_seed(11),
    )
    second = select_random_candidates(
        eligible,
        outputs_per_query=4,
        generator=torch.Generator().manual_seed(11),
    )
    assert torch.equal(first.local_indices, second.local_indices)
    assert all(torch.unique(row).numel() == 4 for row in first.local_indices)


def test_selection_shortfall_returns_unique_outputs_and_reports_failure() -> None:
    features = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    eligible = torch.tensor([[True, False, True, False, False, False]])
    diversity = select_mild_diversity_candidates(
        features,
        eligible,
        outputs_per_query=4,
        diversity_strength=0.35,
        generator=torch.Generator().manual_seed(9),
    )
    assert not bool(diversity.success[0])
    assert diversity.local_indices[0].unique().numel() == 4
    assert {0, 2}.issubset(set(diversity.local_indices[0].tolist()))

    random = select_random_candidates(
        eligible,
        outputs_per_query=4,
        generator=torch.Generator().manual_seed(9),
    )
    assert not bool(random.success[0])
    assert random.local_indices[0].unique().numel() == 4
    assert {0, 2}.issubset(set(random.local_indices[0].tolist()))
