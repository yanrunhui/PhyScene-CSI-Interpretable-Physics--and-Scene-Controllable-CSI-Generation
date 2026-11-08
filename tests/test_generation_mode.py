from __future__ import annotations

import pytest

from physics_csi.generation_mode import (
    CandidateBudgetPolicy,
    CSIGenerationMode,
    resolve_generation_route,
)


def test_no_mean_aod_uses_five_condition_b3_dp_link_route() -> None:
    route = resolve_generation_route(
        target_mean_aod_az_deg=None,
        path_count=3,
    )

    assert route.mode is CSIGenerationMode.FIVE_CONDITION
    assert route.use_natural_b3
    assert route.use_delay_power_residual
    assert route.use_link_global_aod
    assert not route.use_aod_conditioned_b3


def test_mean_aod_uses_cfg2_aod_b3_dp_without_link_global() -> None:
    route = resolve_generation_route(
        target_mean_aod_az_deg=90.0,
        path_count=5,
    )

    assert route.mode is CSIGenerationMode.SIX_CONDITION_MEAN_AOD
    assert route.use_aod_conditioned_b3
    assert route.use_delay_power_residual
    assert not route.use_link_global_aod
    assert not route.use_natural_b3
    assert route.guidance_scale == 2.0


def test_mean_aod_route_rejects_unvalidated_path_count() -> None:
    with pytest.raises(ValueError, match="validated only for L=5"):
        resolve_generation_route(
            target_mean_aod_az_deg=45.0,
            path_count=4,
        )


def test_candidate_budget_starts_at_16_and_tops_up_to_64() -> None:
    policy = CandidateBudgetPolicy(requested_outputs=8)

    assert policy.initial_candidates == 16
    assert policy.maximum_candidates == 64
    assert policy.next_round_size(0) == 16
    assert policy.next_round_size(16) == 8
    assert policy.next_round_size(56) == 8
    assert policy.next_round_size(64) == 0


def test_candidate_budget_never_exceeds_maximum() -> None:
    policy = CandidateBudgetPolicy(
        requested_outputs=8,
        initial_multiplier=2,
        top_up_candidates=10,
        maximum_multiplier=3,
    )

    assert policy.next_round_size(16) == 8
