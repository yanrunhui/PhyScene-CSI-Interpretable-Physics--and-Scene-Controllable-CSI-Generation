from __future__ import annotations

import pytest

from physics_csi.generation_mode import (
    AoDControlMode,
    CSIGenerationMode,
    resolve_generation_route,
)


def test_mean_mode_has_frozen_cfg2_and_16_to_8_policy() -> None:
    route = resolve_generation_route(
        aod_mode="mean_power_weighted",
        target_aod_deg=30.0,
        path_count=5,
    )

    assert route.mode is CSIGenerationMode.SIX_CONDITION_MEAN_AOD
    assert route.aod_mode is AoDControlMode.MEAN_POWER_WEIGHTED
    assert route.aod_condition_type == "mean"
    assert route.guidance_scale == 2.0
    assert route.candidate_multiplier == 2
    assert route.apply_natural_angle_filter
    assert route.strongest_rebind_strategy is None
    assert route.use_delay_power_residual
    assert not route.use_link_global_aod


def test_strongest_mode_has_frozen_cfg1_swap_and_direct_8_policy() -> None:
    route = resolve_generation_route(
        aod_mode="strongest_path",
        target_aod_deg=-75.0,
        path_count=5,
    )

    assert route.mode is CSIGenerationMode.SIX_CONDITION_STRONGEST_AOD
    assert route.aod_mode is AoDControlMode.STRONGEST_PATH
    assert route.aod_condition_type == "strongest"
    assert route.guidance_scale == 1.0
    assert route.candidate_multiplier == 1
    assert not route.apply_natural_angle_filter
    assert route.strongest_rebind_strategy == "swap"
    assert route.use_delay_power_residual
    assert not route.use_link_global_aod


def test_route_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="aod_mode must be one of"):
        resolve_generation_route(
            aod_mode="combined",
            target_aod_deg=30.0,
            path_count=5,
        )


def test_route_rejects_unvalidated_path_count() -> None:
    with pytest.raises(ValueError, match="validated only for L=5"):
        resolve_generation_route(
            aod_mode="mean_power_weighted",
            target_aod_deg=45.0,
            path_count=4,
        )
