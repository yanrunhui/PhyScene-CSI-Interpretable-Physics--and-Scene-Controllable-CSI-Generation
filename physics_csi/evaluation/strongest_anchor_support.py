from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from physics_csi.evaluation.aod_support import (
    circular_distance_degrees,
    conditioned_aod_measurement,
    normalized_condition_features,
)


@dataclass(frozen=True)
class AnchorSupportClassification:
    """A/B/C classification for requested Strongest-Anchor conditions."""

    labels: tuple[str, ...]
    map_support_count: torch.Tensor
    cross_scene_support_count: torch.Tensor
    physically_feasible: torch.Tensor
    conflict_reasons: tuple[tuple[str, ...], ...]


def classify_strongest_anchor_requests(
    reference: Mapping[str, torch.Tensor],
    *,
    target_conditions: torch.Tensor,
    target_validity: torch.Tensor,
    target_map_ids: torch.Tensor,
    target_aod_deg: torch.Tensor,
    target_path_count: int | torch.Tensor,
    condition_std: torch.Tensor,
    maximum_condition_distance: float = 1.5,
    maximum_angle_distance_deg: float = 10.0,
    minimum_map_support: int = 8,
    known_los_direct_aod_deg: torch.Tensor | None = None,
    los_direct_tolerance_deg: float = 10.0,
) -> AnchorSupportClassification:
    """Classify requests without treating low empirical support as impossible.

    A: enough joint Map/condition/angle neighbors.
    B: low Map support, but the declared physical conditions are feasible.
    C: invalid conditions or a known LoS-direct geometry conflict.

    A known direct direction is optional because Map ID and aggregate conditions
    alone do not identify link geometry.  Without it, the classifier abstains
    from declaring an otherwise valid low-support LoS angle impossible.
    """
    if minimum_map_support < 1:
        raise ValueError("minimum_map_support must be positive")
    if maximum_condition_distance <= 0.0 or maximum_angle_distance_deg <= 0.0:
        raise ValueError("support radii must be positive")
    if not 0.0 <= los_direct_tolerance_deg <= 180.0:
        raise ValueError("los_direct_tolerance_deg must lie in [0, 180]")
    conditions = target_conditions.float()
    validity = target_validity.bool()
    map_ids = target_map_ids.long().reshape(-1)
    target = target_aod_deg.float().reshape(-1)
    if conditions.ndim != 2 or conditions.shape[1] != 4:
        raise ValueError("target_conditions must have shape [batch, 4]")
    batch = conditions.shape[0]
    if validity.shape != conditions.shape:
        raise ValueError("target_validity must match target_conditions")
    if map_ids.shape != (batch,) or target.shape != (batch,):
        raise ValueError("Map IDs and target AoD must have shape [batch]")
    count = torch.as_tensor(target_path_count, dtype=torch.long).reshape(-1)
    if count.numel() == 1:
        count = count.expand(batch)
    if count.shape != (batch,):
        raise ValueError("target_path_count must be scalar or have shape [batch]")
    known_direct = None
    if known_los_direct_aod_deg is not None:
        known_direct = known_los_direct_aod_deg.float().reshape(-1)
        if known_direct.numel() == 1:
            known_direct = known_direct.expand(batch)
        if known_direct.shape != (batch,):
            raise ValueError(
                "known_los_direct_aod_deg must be scalar or have shape [batch]"
            )

    real_features = normalized_condition_features(
        reference["conditions"], reference["condition_validity"], condition_std
    )
    request_features = normalized_condition_features(conditions, validity, condition_std)
    real_angle, real_angle_valid, _ = conditioned_aod_measurement(
        reference, condition_type="strongest"
    )
    real_map = reference["map_id"].long()
    real_los = reference["conditions"][:, 3] >= 0.5
    real_k_valid = reference["condition_validity"][:, 2].bool()
    real_count = reference["path_mask"].bool().sum(-1)
    observed_maps = set(int(value) for value in torch.unique(real_map).tolist())

    map_support = torch.zeros(batch, dtype=torch.long)
    cross_support = torch.zeros(batch, dtype=torch.long)
    feasible = torch.ones(batch, dtype=torch.bool)
    reasons: list[tuple[str, ...]] = []
    labels: list[str] = []
    for index in range(batch):
        current: list[str] = []
        los_value = float(conditions[index, 3])
        los = los_value >= 0.5
        k_valid = bool(validity[index, 2])
        if not bool(torch.isfinite(conditions[index]).all()) or not bool(
            torch.isfinite(target[index])
        ):
            current.append("non_finite_condition")
        if float(conditions[index, 1]) < 0.0:
            current.append("negative_delay_spread")
        if los_value not in (0.0, 1.0):
            current.append("los_status_not_binary")
        if int(count[index]) < 1:
            current.append("path_count_below_one")
        if float(conditions[index, 1]) > 0.0 and int(count[index]) < 2:
            current.append("positive_delay_spread_requires_two_paths")
        if k_valid and not los:
            current.append("k_factor_is_invalid_for_nlos")
        if k_valid and int(count[index]) < 2:
            current.append("k_factor_requires_direct_and_scatter_paths")
        if int(map_ids[index]) not in observed_maps:
            current.append("unseen_map")
        if known_direct is not None and los and k_valid and float(conditions[index, 2]) >= 0.0:
            direct_error = float(
                circular_distance_degrees(target[index], known_direct[index])
            )
            if direct_error > los_direct_tolerance_deg:
                current.append("strongest_aod_conflicts_with_known_los_direct_geometry")

        feasible[index] = not current
        reasons.append(tuple(current))
        if current:
            labels.append("C")
            continue

        discrete = (
            (real_los == los)
            & (real_k_valid == k_valid)
            & (real_count == count[index])
            & real_angle_valid
        )
        condition_distance = torch.linalg.vector_norm(
            real_features - request_features[index], dim=-1
        )
        angle_distance = circular_distance_degrees(real_angle, target[index])
        nearby = (
            discrete
            & (condition_distance <= maximum_condition_distance)
            & (angle_distance <= maximum_angle_distance_deg)
        )
        cross_support[index] = int(
            (nearby & (real_map != map_ids[index])).sum()
        )
        map_support[index] = int((nearby & (real_map == map_ids[index])).sum())
        labels.append(
            "A" if int(map_support[index]) >= minimum_map_support else "B"
        )

    return AnchorSupportClassification(
        labels=tuple(labels),
        map_support_count=map_support,
        cross_scene_support_count=cross_support,
        physically_feasible=feasible,
        conflict_reasons=tuple(reasons),
    )
