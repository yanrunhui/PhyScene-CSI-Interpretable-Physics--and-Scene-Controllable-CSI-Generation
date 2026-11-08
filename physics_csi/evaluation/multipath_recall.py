from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch

from physics_csi.evaluation.aod_support import mean_aod
from physics_csi.evaluation.distribution import full_multipath_features
from physics_csi.physics.aod_control import wrap_degrees


FeatureView = Literal[
    "absolute",
    "target_centered",
    "self_centered",
    "without_aod",
]


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: torch.Tensor
    scale: torch.Tensor
    scale_floor: float
    dimensions_using_floor: int

    @classmethod
    def fit(cls, reference: torch.Tensor) -> "FeatureStandardizer":
        if reference.ndim != 2 or reference.shape[0] < 2:
            raise ValueError("reference features must have shape [samples, features]")
        mean = reference.mean(0, keepdim=True)
        raw_scale = reference.std(0, unbiased=False, keepdim=True)
        positive = raw_scale[torch.isfinite(raw_scale) & (raw_scale > 0)]
        typical = (
            positive.median()
            if positive.numel()
            else torch.ones((), device=reference.device)
        )
        floor = torch.maximum(
            typical * 0.05,
            torch.tensor(1e-6, device=reference.device),
        )
        finite = torch.where(
            torch.isfinite(raw_scale), raw_scale, torch.zeros_like(raw_scale)
        )
        return cls(
            mean=mean,
            scale=finite.clamp_min(floor),
            scale_floor=float(floor),
            dimensions_using_floor=int((finite <= floor).sum()),
        )

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class ManifoldResult:
    metrics: dict[str, float]
    recalled_reference: torch.Tensor
    covered_reference: torch.Tensor
    reference_radius: torch.Tensor
    candidate_radius: torch.Tensor
    nearest_candidate_distance: torch.Tensor


def multipath_view_features(
    payload: Mapping[str, torch.Tensor],
    *,
    view: FeatureView,
    target_mean_aod_deg: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build link-level multipath features for a recall interpretation view."""
    if view not in {
        "absolute",
        "target_centered",
        "self_centered",
        "without_aod",
    }:
        raise ValueError(f"unsupported multipath feature view: {view}")
    working = dict(payload)
    if view == "target_centered":
        if target_mean_aod_deg is None:
            raise ValueError("target_centered features require target Mean AoD")
        center = target_mean_aod_deg.to(payload["aod_az_deg"]).reshape(-1)
        if center.shape != (payload["aod_az_deg"].shape[0],):
            raise ValueError("target Mean AoD must have shape [samples]")
        working["aod_az_deg"] = wrap_degrees(
            payload["aod_az_deg"].float() - center.unsqueeze(-1)
        )
    elif view == "self_centered":
        center, _ = mean_aod(payload)
        working["aod_az_deg"] = wrap_degrees(
            payload["aod_az_deg"].float() - center.unsqueeze(-1)
        )

    features = full_multipath_features(working)
    if view == "without_aod":
        # full_multipath_features appends AoD sin, cos and scaled elevation last.
        aod_dimensions = 3 * int(payload["path_mask"].shape[1])
        features = features[:, :-aod_dimensions]
    return features


def manifold_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    requested_k: int,
) -> ManifoldResult:
    """Precision/recall manifold metrics with per-reference membership masks."""
    if candidate.ndim != 2 or reference.ndim != 2:
        raise ValueError("candidate and reference features must be matrices")
    if candidate.shape[1] != reference.shape[1]:
        raise ValueError("candidate and reference features must share dimensions")
    k = min(requested_k, candidate.shape[0] - 1, reference.shape[0] - 1)
    if k < 1:
        raise ValueError("manifold metrics require at least two samples per set")

    reference_self = torch.cdist(reference, reference)
    candidate_self = torch.cdist(candidate, candidate)
    reference_self.fill_diagonal_(torch.inf)
    candidate_self.fill_diagonal_(torch.inf)
    reference_radius = reference_self.kthvalue(k, dim=1).values
    candidate_radius = candidate_self.kthvalue(k, dim=1).values
    cross = torch.cdist(candidate, reference)
    precise_candidate = (
        cross <= reference_radius.unsqueeze(0)
    ).any(1)
    recalled_reference = (
        cross <= candidate_radius.unsqueeze(1)
    ).any(0)
    covered_reference = cross.amin(0) <= reference_radius
    density = (cross <= reference_radius.unsqueeze(0)).float().sum(1).mean() / k
    rare = reference_radius >= torch.quantile(reference_radius, 0.90)
    rare_coverage = (
        covered_reference[rare].float().mean()
        if bool(rare.any())
        else covered_reference.float().mean()
    )
    candidate_pairwise = torch.pdist(candidate).mean().clamp_min(1e-20)
    reference_pairwise = torch.pdist(reference).mean().clamp_min(1e-20)
    return ManifoldResult(
        metrics={
            "precision": float(precise_candidate.float().mean()),
            "recall": float(recalled_reference.float().mean()),
            "density": float(density),
            "coverage": float(covered_reference.float().mean()),
            "rare_reference_coverage": float(rare_coverage),
            "pairwise_log_ratio_abs": float(
                torch.log(candidate_pairwise / reference_pairwise).abs()
            ),
        },
        recalled_reference=recalled_reference,
        covered_reference=covered_reference,
        reference_radius=reference_radius,
        candidate_radius=candidate_radius,
        nearest_candidate_distance=cross.amin(0),
    )


def masked_rate(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    selected = values[mask]
    return float(selected.float().mean()) if selected.numel() else None


def reference_partition_metrics(
    result: ManifoldResult,
    *,
    reference_angle_error_deg: torch.Tensor,
) -> dict[str, float | int | None]:
    """Explain which real target-angle and density regions are missed."""
    angle_error = reference_angle_error_deg.to(result.reference_radius).reshape(-1)
    if angle_error.shape != result.reference_radius.shape:
        raise ValueError("reference angle error must have one value per reference")
    q50 = torch.quantile(result.reference_radius, 0.50)
    q90 = torch.quantile(result.reference_radius, 0.90)
    partitions = {
        "angle_core_le_5deg": angle_error <= 5.0,
        "angle_shoulder_5_to_10deg": (angle_error > 5.0) & (angle_error <= 10.0),
        "angle_outside_10deg": angle_error > 10.0,
        "density_dense_0_to_50pct": result.reference_radius <= q50,
        "density_middle_50_to_90pct": (result.reference_radius > q50)
        & (result.reference_radius < q90),
        "density_rare_top_10pct": result.reference_radius >= q90,
    }
    output: dict[str, float | int | None] = {}
    for name, mask in partitions.items():
        output[f"{name}_samples"] = int(mask.sum())
        output[f"{name}_recall"] = masked_rate(result.recalled_reference, mask)
        output[f"{name}_coverage"] = masked_rate(result.covered_reference, mask)
    output["nearest_candidate_distance_mean"] = float(
        result.nearest_candidate_distance.mean()
    )
    output["candidate_radius_mean"] = float(result.candidate_radius.mean())
    output["reference_radius_mean"] = float(result.reference_radius.mean())
    return output
