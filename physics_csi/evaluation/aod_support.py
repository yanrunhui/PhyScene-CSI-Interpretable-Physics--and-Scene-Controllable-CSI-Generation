from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch

from physics_csi.data.aod_conditioned import power_weighted_mean_aod


def circular_distance_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    difference = torch.deg2rad(first.float() - second.float())
    return torch.rad2deg(torch.atan2(difference.sin(), difference.cos())).abs()


def normalized_condition_features(
    conditions: torch.Tensor,
    validity: torch.Tensor,
    condition_std: torch.Tensor,
) -> torch.Tensor:
    scale = condition_std.float().flatten()[:3].clamp_min(1e-6)
    features = conditions[:, :3].float() / scale
    features = features.clone()
    features[:, 2] = torch.where(
        validity[:, 2].bool(), features[:, 2], torch.zeros_like(features[:, 2])
    )
    return features


def support_strata(
    payload: Mapping[str, torch.Tensor],
) -> dict[tuple[int, bool, bool, int], torch.Tensor]:
    conditions = payload["conditions"]
    validity = payload["condition_validity"].bool()
    map_ids = payload["map_id"].long()
    path_count = payload["path_mask"].bool().sum(-1)
    grouped: dict[tuple[int, bool, bool, int], list[int]] = defaultdict(list)
    for index in range(conditions.shape[0]):
        grouped[
            (
                int(map_ids[index]),
                bool(conditions[index, 3] >= 0.5),
                bool(validity[index, 2]),
                int(path_count[index]),
            )
        ].append(index)
    return {
        key: torch.tensor(indices, dtype=torch.long)
        for key, indices in grouped.items()
    }


def mean_aod(payload: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return power_weighted_mean_aod(
        payload["path_powers_linear"], payload["aod_az_deg"], payload["path_mask"]
    )


def local_support_counts(
    payload: Mapping[str, torch.Tensor],
    *,
    condition_std: torch.Tensor,
    maximum_condition_distance: float,
    maximum_angle_distance_deg: float,
) -> torch.Tensor:
    """Count real neighbors with the same discrete conditions and nearby c/AoD."""
    features = normalized_condition_features(
        payload["conditions"], payload["condition_validity"], condition_std
    )
    angles, concentration = mean_aod(payload)
    valid_angle = concentration >= 0.10
    counts = torch.zeros(features.shape[0], dtype=torch.long)
    for indices in support_strata(payload).values():
        if indices.numel() < 2:
            continue
        condition_distance = torch.cdist(features[indices], features[indices])
        angle_distance = circular_distance_degrees(
            angles[indices].unsqueeze(1), angles[indices].unsqueeze(0)
        )
        eligible = (
            (condition_distance <= maximum_condition_distance)
            & (angle_distance <= maximum_angle_distance_deg)
            & valid_angle[indices].unsqueeze(1)
            & valid_angle[indices].unsqueeze(0)
        )
        eligible.fill_diagonal_(False)
        counts[indices] = eligible.sum(-1).cpu()
    return counts


def match_independent_supported_pairs(
    payload: Mapping[str, torch.Tensor],
    *,
    condition_std: torch.Tensor,
    maximum_condition_distance: float,
    maximum_angle_distance_deg: float,
    minimum_support: int,
    path_count: int,
    maximum_pairs_per_map_los: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build disjoint query/donor pairs within Map, LoS, K-validity and L."""
    if minimum_support < 1 or maximum_pairs_per_map_los < 1:
        raise ValueError("support and pair limits must be positive")
    features = normalized_condition_features(
        payload["conditions"], payload["condition_validity"], condition_std
    )
    angles, concentration = mean_aod(payload)
    valid_angle = concentration >= 0.10
    support = local_support_counts(
        payload,
        condition_std=condition_std,
        maximum_condition_distance=maximum_condition_distance,
        maximum_angle_distance_deg=maximum_angle_distance_deg,
    )
    generator = torch.Generator().manual_seed(seed)
    query_chunks: list[torch.Tensor] = []
    donor_chunks: list[torch.Tensor] = []
    distance_chunks: list[torch.Tensor] = []
    strata_audit: list[dict[str, Any]] = []
    for key, indices in sorted(support_strata(payload).items()):
        map_id, los, k_valid, current_l = key
        if current_l != path_count or indices.numel() < 2:
            continue
        indices = indices[valid_angle[indices]]
        permutation = indices[
            torch.randperm(indices.numel(), generator=generator)
        ]
        midpoint = permutation.numel() // 2
        queries = permutation[:midpoint]
        donors = permutation[midpoint : midpoint * 2]
        queries = queries[support[queries] >= minimum_support]
        if not queries.numel() or not donors.numel():
            continue
        condition_distance = torch.cdist(features[queries], features[donors])
        angle_distance = circular_distance_degrees(
            angles[queries].unsqueeze(1), angles[donors].unsqueeze(0)
        )
        valid = (
            (condition_distance <= maximum_condition_distance)
            & (angle_distance <= maximum_angle_distance_deg)
        )
        # Joint score is dimensionless: one unit means reaching either limit.
        score = condition_distance / max(maximum_condition_distance, 1e-12)
        score = score + angle_distance / max(maximum_angle_distance_deg, 1e-12)
        score = score.masked_fill(~valid, torch.inf)
        order = torch.argsort(score.amin(-1))
        used = torch.zeros(donors.numel(), dtype=torch.bool)
        matched_query: list[int] = []
        matched_donor: list[int] = []
        matched_distance: list[float] = []
        for row in order.tolist():
            for column in torch.argsort(score[row]).tolist():
                if not math_isfinite(float(score[row, column])):
                    break
                if not bool(used[column]):
                    used[column] = True
                    matched_query.append(int(queries[row]))
                    matched_donor.append(int(donors[column]))
                    matched_distance.append(float(condition_distance[row, column]))
                    break
            if len(matched_query) >= maximum_pairs_per_map_los:
                break
        if matched_query:
            query_chunks.append(torch.tensor(matched_query, dtype=torch.long))
            donor_chunks.append(torch.tensor(matched_donor, dtype=torch.long))
            distance_chunks.append(torch.tensor(matched_distance, dtype=torch.float32))
        strata_audit.append(
            {
                "map_id": map_id,
                "los_status": int(los),
                "k_factor_valid": k_valid,
                "path_count": current_l,
                "real_samples": int(indices.numel()),
                "high_support_queries": int(queries.numel()),
                "independent_pairs": len(matched_query),
            }
        )
    if not query_chunks:
        raise RuntimeError("no independent high-support Map-condition-AoD pairs found")
    query = torch.cat(query_chunks)
    donor = torch.cat(donor_chunks)
    distances = torch.cat(distance_chunks)
    if torch.isin(query, donor).any():
        raise RuntimeError("query and donor halves are not independent")
    return query, donor, support, {
        "pairs": int(query.numel()),
        "maps": int(torch.unique(payload["map_id"][query]).numel()),
        "condition_distance": {
            "mean": float(distances.mean()),
            "p95": float(torch.quantile(distances, 0.95)),
            "maximum": float(distances.max()),
        },
        "strata": strata_audit,
    }


def math_isfinite(value: float) -> bool:
    # Kept local to avoid converting score matrices through Python/NumPy.
    return value != float("inf") and value != float("-inf") and value == value


def generated_support_counts(
    generated: Mapping[str, torch.Tensor],
    target_conditions: torch.Tensor,
    target_validity: torch.Tensor,
    target_map_ids: torch.Tensor,
    real_payload: Mapping[str, torch.Tensor],
    *,
    condition_std: torch.Tensor,
    maximum_condition_distance: float,
    maximum_angle_distance_deg: float,
) -> torch.Tensor:
    """Count real neighbors around each generated output in its requested stratum."""
    real_features = normalized_condition_features(
        real_payload["conditions"], real_payload["condition_validity"], condition_std
    )
    generated_features = normalized_condition_features(
        target_conditions, target_validity, condition_std
    )
    real_angles, real_concentration = mean_aod(real_payload)
    generated_angles, generated_concentration = mean_aod(generated)
    real_valid = real_concentration >= 0.10
    generated_valid = generated_concentration >= 0.10
    real_l = real_payload["path_mask"].bool().sum(-1)
    generated_l = generated["path_mask"].bool().sum(-1)
    real_los = real_payload["conditions"][:, 3] >= 0.5
    generated_los = target_conditions[:, 3] >= 0.5
    real_k_valid = real_payload["condition_validity"][:, 2].bool()
    generated_k_valid = target_validity[:, 2].bool()
    counts = torch.zeros(generated_angles.numel(), dtype=torch.long)
    for index in range(generated_angles.numel()):
        eligible = (
            (real_payload["map_id"].long() == int(target_map_ids[index]))
            & (real_l == generated_l[index])
            & (real_los == generated_los[index])
            & (real_k_valid == generated_k_valid[index])
            & real_valid
        )
        candidates = torch.nonzero(eligible, as_tuple=False).flatten()
        if not candidates.numel() or not bool(generated_valid[index]):
            continue
        condition_distance = torch.linalg.vector_norm(
            real_features[candidates] - generated_features[index], dim=-1
        )
        angle_distance = circular_distance_degrees(
            real_angles[candidates], generated_angles[index]
        )
        counts[index] = int(
            (
                (condition_distance <= maximum_condition_distance)
                & (angle_distance <= maximum_angle_distance_deg)
            ).sum()
        )
    return counts
