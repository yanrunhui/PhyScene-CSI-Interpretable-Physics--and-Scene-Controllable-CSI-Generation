from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from physics_csi.evaluation.aod_support import strongest_aod
from physics_csi.physics.aod_control import wrap_degrees


@dataclass(frozen=True)
class DiversitySelection:
    local_indices: torch.Tensor
    flat_indices: torch.Tensor
    eligible_count: torch.Tensor
    success: torch.Tensor


def candidate_diversity_features(
    payload: Mapping[str, torch.Tensor],
    target_aod_deg: torch.Tensor,
) -> torch.Tensor:
    """Build relative multipath and compact Tx-spatial selection features."""
    mask = payload["path_mask"].bool()
    samples, paths = mask.shape
    target = target_aod_deg.to(payload["aod_az_deg"]).reshape(-1)
    if target.shape != (samples,):
        raise ValueError("target_aod_deg must have one value per candidate")
    delay = payload["path_delays_ns"].float()
    first_delay = torch.where(mask, delay, torch.full_like(delay, torch.inf)).amin(-1)
    relative_delay = torch.log1p((delay - first_delay.unsqueeze(-1)).clamp_min(0.0))
    power = payload["path_powers_linear"].float().clamp_min(0.0) * mask
    power_fraction = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
    relative_aod = torch.deg2rad(
        wrap_degrees(payload["aod_az_deg"].float() - target.unsqueeze(-1))
    )
    multipath = torch.cat(
        [
            mask.float(),
            relative_delay * mask,
            power_fraction,
            relative_aod.sin() * mask,
            relative_aod.cos() * mask,
        ],
        dim=-1,
    )
    csi = payload.get("csi")
    if not isinstance(csi, torch.Tensor) or csi.ndim != 4:
        raise ValueError("candidate diversity features require CSI with [B,R,T,F]")
    tx_energy = csi.abs().square().mean((1, 3))
    tx_energy = tx_energy / tx_energy.sum(-1, keepdim=True).clamp_min(1e-20)
    if csi.shape[2] > 1:
        adjacent = (csi[:, :, 1:, :] * csi[:, :, :-1, :].conj()).mean((1, 3))
        adjacent = adjacent / adjacent.abs().mean(-1, keepdim=True).clamp_min(1e-20)
        spatial = torch.cat([tx_energy, adjacent.real, adjacent.imag], dim=-1)
    else:
        spatial = tx_energy
    return torch.cat([multipath, spatial.float()], dim=-1)


def _facility_select_one(
    features: torch.Tensor,
    eligible: torch.Tensor,
    outputs: int,
    diversity_strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    eligible_candidates = torch.nonzero(eligible, as_tuple=False).flatten()
    enough_eligible = eligible_candidates.numel() >= outputs
    candidates = (
        eligible_candidates
        if enough_eligible
        else torch.arange(features.shape[0], device=features.device)
    )
    values = features[candidates].float()
    mean = values.mean(0, keepdim=True)
    scale = values.std(0, unbiased=False, keepdim=True).clamp_min(1e-4)
    values = (values - mean) / scale
    distance = torch.cdist(values, values)
    positive = distance[distance > 0.0]
    bandwidth = (
        positive.median()
        if positive.numel()
        else torch.ones((), device=values.device)
    )
    similarity = torch.exp(-distance / bandwidth.clamp_min(1e-6))
    random_priority = torch.rand(
        candidates.numel(), generator=generator, device=features.device
    )
    covered = torch.zeros(candidates.numel(), device=features.device)
    selected: list[int] = [] if enough_eligible else eligible_candidates.tolist()
    available = torch.ones(candidates.numel(), dtype=torch.bool, device=features.device)
    if selected:
        # Preserve every eligible output when the pool misses the 8-output
        # contract, then use representativeness only to fill the shortfall.
        selected_tensor = torch.tensor(selected, device=features.device)
        available[selected_tensor] = False
        covered = similarity[:, selected_tensor].amax(-1)
    for _ in range(outputs - len(selected)):
        gains = (
            torch.maximum(covered.unsqueeze(-1), similarity)
            - covered.unsqueeze(-1)
        ).sum(0)
        gains = gains / gains.max().clamp_min(1e-12)
        score = (
            diversity_strength * gains
            + (1.0 - diversity_strength) * random_priority
        )
        score = score.masked_fill(~available, -torch.inf)
        chosen = int(score.argmax())
        selected.append(chosen)
        available[chosen] = False
        covered = torch.maximum(covered, similarity[:, chosen])
    return candidates[torch.tensor(selected, device=features.device)]


def select_mild_diversity_candidates(
    features: torch.Tensor,
    eligible: torch.Tensor,
    *,
    outputs_per_query: int,
    diversity_strength: float,
    generator: torch.Generator,
) -> DiversitySelection:
    """Select representative candidates without maximizing extreme distances."""
    if features.ndim != 3 or eligible.shape != features.shape[:2]:
        raise ValueError("features must be [Q,C,D] and eligible must be [Q,C]")
    if not 0.0 <= diversity_strength <= 1.0:
        raise ValueError("diversity_strength must lie in [0, 1]")
    queries, candidates, _ = features.shape
    if not 1 <= outputs_per_query <= candidates:
        raise ValueError("outputs_per_query must lie in [1, candidates]")
    eligible_count = eligible.sum(-1)
    success = eligible_count >= outputs_per_query
    local = torch.stack(
        [
            _facility_select_one(
                features[index],
                eligible[index],
                outputs_per_query,
                diversity_strength,
                generator,
            )
            for index in range(queries)
        ]
    )
    offsets = torch.arange(queries, device=features.device).unsqueeze(-1) * candidates
    return DiversitySelection(
        local_indices=local,
        flat_indices=local + offsets,
        eligible_count=eligible_count,
        success=success,
    )


def select_random_candidates(
    eligible: torch.Tensor,
    *,
    outputs_per_query: int,
    generator: torch.Generator,
) -> DiversitySelection:
    if eligible.ndim != 2:
        raise ValueError("eligible must have shape [Q,C]")
    queries, candidates = eligible.shape
    if not 1 <= outputs_per_query <= candidates:
        raise ValueError("outputs_per_query must lie in [1, candidates]")
    eligible_count = eligible.sum(-1)
    success = eligible_count >= outputs_per_query
    rows: list[torch.Tensor] = []
    for index in range(queries):
        available = torch.nonzero(eligible[index], as_tuple=False).flatten()
        if available.numel() >= outputs_per_query:
            order = torch.randperm(
                available.numel(), generator=generator, device=eligible.device
            )
            rows.append(available[order[:outputs_per_query]])
        else:
            ineligible = torch.nonzero(~eligible[index], as_tuple=False).flatten()
            fill_order = torch.randperm(
                ineligible.numel(), generator=generator, device=eligible.device
            )
            rows.append(
                torch.cat(
                    [
                        available,
                        ineligible[
                            fill_order[: outputs_per_query - available.numel()]
                        ],
                    ]
                )
            )
    local = torch.stack(rows)
    offsets = torch.arange(queries, device=eligible.device).unsqueeze(-1) * candidates
    return DiversitySelection(
        local_indices=local,
        flat_indices=local + offsets,
        eligible_count=eligible_count,
        success=success,
    )


def strongest_candidate_eligibility(
    payload: Mapping[str, torch.Tensor],
    target_aod_deg: torch.Tensor,
    *,
    queries: int,
    candidates_per_query: int,
    tolerance_deg: float = 10.0,
) -> torch.Tensor:
    measured = strongest_aod(payload)
    target = target_aod_deg.to(measured.angle_deg).reshape(queries)
    error = wrap_degrees(
        measured.angle_deg - target.repeat_interleave(candidates_per_query)
    ).abs()
    feasible = payload.get("constraint_feasible")
    valid = measured.valid
    if isinstance(feasible, torch.Tensor):
        valid = valid & feasible.bool()
    return (valid & (error <= tolerance_deg)).reshape(queries, candidates_per_query)
