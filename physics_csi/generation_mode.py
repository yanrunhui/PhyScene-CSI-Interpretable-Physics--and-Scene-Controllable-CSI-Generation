from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class CSIGenerationMode(str, Enum):
    """Supported production inference routes."""

    FIVE_CONDITION = "five_condition"
    SIX_CONDITION_MEAN_AOD = "six_condition_mean_aod"


@dataclass(frozen=True)
class GenerationRoute:
    """Resolved model route for one user generation request."""

    mode: CSIGenerationMode
    use_natural_b3: bool
    use_aod_conditioned_b3: bool
    use_delay_power_residual: bool
    use_link_global_aod: bool
    target_mean_aod_az_deg: float | None
    required_path_count: int | None
    guidance_scale: float | None


@dataclass(frozen=True)
class CandidateBudgetPolicy:
    """Bounded dynamic-candidate policy for six-condition generation."""

    requested_outputs: int
    initial_multiplier: int = 2
    top_up_candidates: int = 8
    maximum_multiplier: int = 8

    def __post_init__(self) -> None:
        if self.requested_outputs < 1:
            raise ValueError("requested_outputs must be positive")
        if self.initial_multiplier < 1:
            raise ValueError("initial_multiplier must be positive")
        if self.top_up_candidates < 1:
            raise ValueError("top_up_candidates must be positive")
        if self.maximum_multiplier < self.initial_multiplier:
            raise ValueError(
                "maximum_multiplier cannot be smaller than initial_multiplier"
            )

    @property
    def initial_candidates(self) -> int:
        return self.requested_outputs * self.initial_multiplier

    @property
    def maximum_candidates(self) -> int:
        return self.requested_outputs * self.maximum_multiplier

    def next_round_size(self, generated_candidates: int) -> int:
        """Return the next bounded generation round size, or zero at budget."""
        if generated_candidates < 0:
            raise ValueError("generated_candidates cannot be negative")
        if generated_candidates >= self.maximum_candidates:
            return 0
        requested = (
            self.initial_candidates
            if generated_candidates == 0
            else self.top_up_candidates
        )
        return min(requested, self.maximum_candidates - generated_candidates)


def resolve_generation_route(
    *,
    target_mean_aod_az_deg: float | None,
    path_count: int,
    aod_trained_path_count: int = 5,
    aod_guidance_scale: float = 2.0,
) -> GenerationRoute:
    """Choose the frozen production pipeline without silently changing intent.

    The current AoD-conditioned checkpoint was trained only on ``L=5``.  An
    explicit error is safer than pretending that it supports arbitrary path
    counts.  The five-condition route remains valid for every path count that
    the original B3 supports.
    """
    if path_count < 1:
        raise ValueError("path_count must be positive")
    if aod_trained_path_count < 1:
        raise ValueError("aod_trained_path_count must be positive")
    if target_mean_aod_az_deg is None:
        return GenerationRoute(
            mode=CSIGenerationMode.FIVE_CONDITION,
            use_natural_b3=True,
            use_aod_conditioned_b3=False,
            use_delay_power_residual=True,
            use_link_global_aod=True,
            target_mean_aod_az_deg=None,
            required_path_count=None,
            guidance_scale=None,
        )

    if not math.isfinite(aod_guidance_scale) or aod_guidance_scale <= 0.0:
        raise ValueError("aod_guidance_scale must be finite and positive")
    target = float(target_mean_aod_az_deg)
    if not math.isfinite(target):
        raise ValueError("target Mean AoD azimuth must be finite")
    if path_count != aod_trained_path_count:
        raise ValueError(
            "six-condition Mean-AoD generation is currently validated only for "
            f"L={aod_trained_path_count}; received L={path_count}"
        )
    return GenerationRoute(
        mode=CSIGenerationMode.SIX_CONDITION_MEAN_AOD,
        use_natural_b3=False,
        use_aod_conditioned_b3=True,
        use_delay_power_residual=True,
        use_link_global_aod=False,
        target_mean_aod_az_deg=target,
        required_path_count=aod_trained_path_count,
        guidance_scale=float(aod_guidance_scale),
    )
