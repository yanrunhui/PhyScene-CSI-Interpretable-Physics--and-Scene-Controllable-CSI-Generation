from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch

from physics_csi.evaluation.aod_support import (
    circular_distance_degrees,
    mean_aod,
    strongest_aod,
)
from physics_csi.physics.aod_control import wrap_degrees


@dataclass(frozen=True)
class FilteredCandidateSelection:
    selected_flat_indices: torch.Tensor
    success: torch.Tensor
    eligible_count: torch.Tensor
    selected_error_deg: torch.Tensor
    eligible: torch.Tensor


def select_first_eligible_candidates(
    payload: Mapping[str, torch.Tensor],
    target_mean_aod_deg: torch.Tensor,
    *,
    queries: int,
    candidates_per_query: int,
    outputs_per_query: int,
    tolerance_deg: float,
    minimum_concentration: float = 0.10,
    mode: Literal["mean", "strongest"] = "mean",
    exclude_strongest_near_ties: bool = False,
    strongest_near_tie_gap_db: float = 0.5,
) -> FilteredCandidateSelection:
    """Select natural Mean or Strongest AoD candidates in generation order."""
    if queries < 1 or candidates_per_query < 1 or outputs_per_query < 1:
        raise ValueError("query, candidate and output counts must be positive")
    if outputs_per_query > candidates_per_query:
        raise ValueError("outputs_per_query cannot exceed candidates_per_query")
    if not 0.0 <= tolerance_deg <= 180.0:
        raise ValueError("tolerance_deg must lie in [0, 180]")
    if not 0.0 <= minimum_concentration <= 1.0:
        raise ValueError("minimum_concentration must lie in [0, 1]")
    if mode not in {"mean", "strongest"}:
        raise ValueError("mode must be 'mean' or 'strongest'")
    if strongest_near_tie_gap_db < 0.0:
        raise ValueError("strongest_near_tie_gap_db must be non-negative")
    samples = int(payload["path_mask"].shape[0])
    if samples != queries * candidates_per_query:
        raise ValueError("payload size does not match query and candidate counts")
    target = target_mean_aod_deg.to(payload["aod_az_deg"]).reshape(-1)
    if target.shape != (queries,):
        raise ValueError("target AoD must have shape [queries]")

    if mode == "mean":
        actual, concentration = mean_aod(payload)
        measurement_valid = concentration >= minimum_concentration
    else:
        measured = strongest_aod(
            payload, near_tie_gap_db=strongest_near_tie_gap_db
        )
        actual = measured.angle_deg
        measurement_valid = measured.valid
        if exclude_strongest_near_ties:
            measurement_valid = measurement_valid & ~measured.near_tie
    expanded_target = target.repeat_interleave(candidates_per_query)
    error = circular_distance_degrees(actual, expanded_target).reshape(
        queries, candidates_per_query
    )
    eligible = (error <= tolerance_deg) & measurement_valid.reshape(
        queries, candidates_per_query
    )
    eligible_count = eligible.sum(-1)
    success = eligible_count >= outputs_per_query
    selected_local = torch.zeros(
        queries,
        outputs_per_query,
        dtype=torch.long,
        device=eligible.device,
    )
    for query in range(queries):
        accepted = torch.nonzero(eligible[query], as_tuple=False).flatten()
        if accepted.numel() >= outputs_per_query:
            selected_local[query] = accepted[:outputs_per_query]
        elif accepted.numel():
            # Failed rows remain explicitly invalid, but retaining accepted
            # indices makes the saved audit easier to inspect.
            selected_local[query, : accepted.numel()] = accepted
    offsets = (
        torch.arange(queries, device=eligible.device).unsqueeze(-1)
        * candidates_per_query
    )
    selected_flat = selected_local + offsets
    selected_error = error.gather(1, selected_local)
    selected_error = torch.where(
        success.unsqueeze(-1),
        selected_error,
        torch.full_like(selected_error, torch.nan),
    )
    return FilteredCandidateSelection(
        selected_flat_indices=selected_flat,
        success=success,
        eligible_count=eligible_count,
        selected_error_deg=selected_error,
        eligible=eligible,
    )


def circular_angle_bin(
    angle_deg: torch.Tensor, *, width_deg: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable circular bin IDs and centers over [-180, 180)."""
    if not 0.0 < width_deg <= 360.0:
        raise ValueError("width_deg must lie in (0, 360]")
    bins = max(1, int(round(360.0 / width_deg)))
    effective_width = 360.0 / bins
    wrapped = wrap_degrees(angle_deg.float())
    index = torch.floor((wrapped + 180.0) / effective_width).long().clamp_max(
        bins - 1
    )
    center = -180.0 + (index.float() + 0.5) * effective_width
    return index, wrap_degrees(center)
