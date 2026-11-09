from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class AoDControlMode(str, Enum):
    """The two supported user-facing AoD semantics."""

    MEAN_POWER_WEIGHTED = "mean_power_weighted"
    STRONGEST_PATH = "strongest_path"


class CSIGenerationMode(str, Enum):
    """Stable artifact labels for the two production inference routes."""

    SIX_CONDITION_MEAN_AOD = "six_condition_mean_aod"
    SIX_CONDITION_STRONGEST_AOD = "six_condition_strongest_aod"


@dataclass(frozen=True)
class GenerationRoute:
    """Frozen checkpoint and post-processing contract for one request."""

    mode: CSIGenerationMode
    aod_mode: AoDControlMode
    use_delay_power_residual: bool
    use_link_global_aod: bool
    target_mean_aod_az_deg: float | None
    target_strongest_aod_az_deg: float | None
    required_path_count: int
    guidance_scale: float
    candidate_multiplier: int
    apply_natural_angle_filter: bool
    strongest_rebind_strategy: Literal["swap"] | None

    @property
    def target_aod_az_deg(self) -> float:
        target = (
            self.target_mean_aod_az_deg
            if self.target_mean_aod_az_deg is not None
            else self.target_strongest_aod_az_deg
        )
        if target is None:
            raise RuntimeError("AoD production route has no target")
        return target

    @property
    def aod_condition_type(self) -> Literal["mean", "strongest"]:
        return (
            "mean"
            if self.aod_mode is AoDControlMode.MEAN_POWER_WEIGHTED
            else "strongest"
        )


def resolve_generation_route(
    *,
    aod_mode: str | AoDControlMode,
    target_aod_deg: float,
    path_count: int,
    aod_trained_path_count: int = 5,
) -> GenerationRoute:
    """Resolve one of the two frozen, mutually exclusive AoD pipelines."""
    try:
        resolved_mode = AoDControlMode(aod_mode)
    except ValueError as error:
        supported = ", ".join(mode.value for mode in AoDControlMode)
        raise ValueError(f"aod_mode must be one of: {supported}") from error
    if path_count < 1 or aod_trained_path_count < 1:
        raise ValueError("path counts must be positive")
    if path_count != aod_trained_path_count:
        raise ValueError(
            "AoD generation is currently validated only for "
            f"L={aod_trained_path_count}; received L={path_count}"
        )
    target = float(target_aod_deg)
    if not math.isfinite(target):
        raise ValueError("target AoD azimuth must be finite")

    if resolved_mode is AoDControlMode.MEAN_POWER_WEIGHTED:
        return GenerationRoute(
            mode=CSIGenerationMode.SIX_CONDITION_MEAN_AOD,
            aod_mode=resolved_mode,
            use_delay_power_residual=True,
            use_link_global_aod=False,
            target_mean_aod_az_deg=target,
            target_strongest_aod_az_deg=None,
            required_path_count=aod_trained_path_count,
            guidance_scale=2.0,
            candidate_multiplier=2,
            apply_natural_angle_filter=True,
            strongest_rebind_strategy=None,
        )
    return GenerationRoute(
        mode=CSIGenerationMode.SIX_CONDITION_STRONGEST_AOD,
        aod_mode=resolved_mode,
        use_delay_power_residual=True,
        use_link_global_aod=False,
        target_mean_aod_az_deg=None,
        target_strongest_aod_az_deg=target,
        required_path_count=aod_trained_path_count,
        guidance_scale=1.0,
        candidate_multiplier=1,
        apply_natural_angle_filter=False,
        strongest_rebind_strategy="swap",
    )
