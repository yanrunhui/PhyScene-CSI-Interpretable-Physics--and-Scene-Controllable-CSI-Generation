from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset


def power_weighted_mean_aod(
    powers_linear: torch.Tensor,
    aod_az_deg: torch.Tensor,
    path_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return circular power-weighted Mean AoD and resultant concentration."""
    powers = powers_linear.float() * path_mask.bool()
    total = powers.sum(-1).clamp_min(1e-20)
    weights = powers / total.unsqueeze(-1)
    angle = torch.deg2rad(aod_az_deg.float())
    sine = (weights * angle.sin()).sum(-1)
    cosine = (weights * angle.cos()).sum(-1)
    concentration = torch.sqrt(sine.square() + cosine.square()).clamp(0.0, 1.0)
    mean_deg = torch.rad2deg(torch.atan2(sine, cosine))
    return mean_deg, concentration


def strongest_path_aod(
    powers_linear: torch.Tensor,
    aod_az_deg: torch.Tensor,
    path_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the active first-argmax AoD, index, top-two dB gap and validity.

    The path index follows the record's current ordering; it is deliberately
    not assumed to be the first or earliest path. Equal maximum powers use
    PyTorch's deterministic first-argmax rule. A link with only one active path
    has an infinite top-two gap because no competing path exists.
    """
    powers = powers_linear.float()
    angles = aod_az_deg.float()
    mask = path_mask.bool()
    if powers.ndim < 1 or powers.shape != angles.shape or powers.shape != mask.shape:
        raise ValueError("path power, AoD azimuth and path mask must share [..., path]")
    if powers.shape[-1] < 1:
        raise ValueError("strongest-path AoD requires at least one path slot")

    masked_power = powers.masked_fill(~mask, -torch.inf)
    strongest_index = masked_power.argmax(dim=-1)
    strongest_angle = angles.gather(
        -1, strongest_index.unsqueeze(-1)
    ).squeeze(-1)
    strongest_power = masked_power.gather(
        -1, strongest_index.unsqueeze(-1)
    ).squeeze(-1)

    active_count = mask.sum(dim=-1)
    top = masked_power.topk(min(2, powers.shape[-1]), dim=-1).values
    second_power = (
        top[..., 1]
        if top.shape[-1] > 1
        else torch.full_like(strongest_power, -torch.inf)
    )
    finite_ratio = 10.0 * torch.log10(
        strongest_power.clamp_min(1e-20) / second_power.clamp_min(1e-20)
    )
    top2_gap_db = torch.where(
        (active_count >= 2) & (second_power > 0.0),
        finite_ratio,
        torch.full_like(finite_ratio, torch.inf),
    )

    active_finite = torch.where(
        mask, torch.isfinite(powers), torch.ones_like(mask)
    )
    active_nonnegative = torch.where(mask, powers >= 0.0, torch.ones_like(mask))
    valid = (
        mask.any(dim=-1)
        & active_finite.all(dim=-1)
        & active_nonnegative.all(dim=-1)
        & (strongest_power > 0.0)
        & torch.isfinite(strongest_angle)
    )
    return strongest_angle, strongest_index, top2_gap_db, valid


class AoDConditionedDataset(Dataset):
    """Attach a continuous Mean- or strongest-path AoD condition label."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        path_count: int | None = 5,
        minimum_concentration: float = 0.10,
        reject_low_concentration: bool = True,
        condition_type: str = "mean",
    ) -> None:
        if path_count is not None and path_count < 1:
            raise ValueError("path_count must be positive or None")
        if not 0.0 <= minimum_concentration <= 1.0:
            raise ValueError("minimum_concentration must lie in [0, 1]")
        if condition_type not in {"mean", "strongest"}:
            raise ValueError("condition_type must be 'mean' or 'strongest'")
        self.dataset = dataset
        self.condition_type = condition_type
        self.required_path_count = path_count
        self.minimum_concentration = float(minimum_concentration)
        self.reject_low_concentration = bool(reject_low_concentration)
        self.indices: list[int] = []
        self.mean_aod_az_deg: list[float] = []
        self.mean_aod_concentration: list[float] = []
        self.strongest_path_indices: list[int] = []
        self.strongest_top2_power_gap_db: list[float] = []
        self.map_ids: list[int] = []
        self.los_status: list[int] = []
        self.dropped_path_count = 0
        self.dropped_low_concentration = 0
        self.dropped_invalid_strongest = 0
        for index in range(len(dataset)):
            record = dataset[index]
            count = int(record["path_count"])
            if path_count is not None and count != path_count:
                self.dropped_path_count += 1
                continue
            if self.condition_type == "mean":
                angle, concentration = power_weighted_mean_aod(
                    record["path_powers_linear"],
                    record["aod_az_deg"],
                    record["path_mask"],
                )
                angle_value = float(angle)
                concentration_value = float(concentration)
                valid = (
                    math.isfinite(angle_value)
                    and math.isfinite(concentration_value)
                    and concentration_value >= self.minimum_concentration
                )
                if self.reject_low_concentration and not valid:
                    self.dropped_low_concentration += 1
                    continue
            else:
                angle, strongest_index, gap_db, validity = strongest_path_aod(
                    record["path_powers_linear"],
                    record["aod_az_deg"],
                    record["path_mask"],
                )
                angle_value = float(angle)
                valid = bool(validity)
                if not valid:
                    self.dropped_invalid_strongest += 1
                    continue
                concentration_value = 0.0
                self.strongest_path_indices.append(int(strongest_index))
                self.strongest_top2_power_gap_db.append(float(gap_db))
            self.indices.append(index)
            self.mean_aod_az_deg.append(angle_value if valid else 0.0)
            self.mean_aod_concentration.append(concentration_value)
            self.map_ids.append(int(record["map_id"]))
            self.los_status.append(int(float(record["conditions"][3]) >= 0.5))
        if not self.indices:
            raise ValueError("AoD-conditioned dataset contains no retained samples")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.dataset[self.indices[index]])
        if self.condition_type == "strongest":
            record["strongest_aod_az_deg"] = torch.tensor(
                self.mean_aod_az_deg[index], dtype=torch.float32
            )
            record["strongest_path_index"] = torch.tensor(
                self.strongest_path_indices[index], dtype=torch.long
            )
            record["strongest_top2_power_gap_db"] = torch.tensor(
                self.strongest_top2_power_gap_db[index], dtype=torch.float32
            )
            record["strongest_aod_valid"] = torch.tensor(True, dtype=torch.bool)
            return record
        concentration = self.mean_aod_concentration[index]
        valid = concentration >= self.minimum_concentration
        record["mean_aod_az_deg"] = torch.tensor(
            self.mean_aod_az_deg[index], dtype=torch.float32
        )
        record["mean_aod_concentration"] = torch.tensor(
            concentration, dtype=torch.float32
        )
        record["mean_aod_valid"] = torch.tensor(valid, dtype=torch.bool)
        return record

    def audit(self) -> dict[str, Any]:
        angles = torch.tensor(self.mean_aod_az_deg)
        concentrations = torch.tensor(self.mean_aod_concentration)
        map_histogram = Counter(self.map_ids)
        angle_bin = torch.floor((angles + 180.0) / 30.0).long().clamp(0, 11)
        result: dict[str, Any] = {
            "input_samples": len(self.dataset),
            "retained_samples": len(self),
            "condition_type": self.condition_type,
            "required_path_count": self.required_path_count,
            "minimum_concentration": self.minimum_concentration,
            "reject_low_concentration": self.reject_low_concentration,
            "dropped_path_count": self.dropped_path_count,
            "dropped_low_concentration": self.dropped_low_concentration,
            "dropped_invalid_strongest": self.dropped_invalid_strongest,
            "los_samples": sum(self.los_status),
            "nlos_samples": len(self.los_status) - sum(self.los_status),
            "maps": len(map_histogram),
            "per_map_samples": {
                "minimum": min(map_histogram.values()),
                "median": float(torch.tensor(list(map_histogram.values())).median()),
                "maximum": max(map_histogram.values()),
            },
            "angle_histogram_30deg": {
                f"[{lower},{lower + 30})": int((angle_bin == number).sum())
                for number, lower in enumerate(range(-180, 180, 30))
            },
        }
        if self.condition_type == "mean":
            result["concentration"] = {
                "mean": float(concentrations.mean()),
                "p05": float(torch.quantile(concentrations, 0.05)),
                "median": float(torch.quantile(concentrations, 0.50)),
                "p95": float(torch.quantile(concentrations, 0.95)),
            }
            return result

        gaps = torch.tensor(self.strongest_top2_power_gap_db, dtype=torch.float32)
        los = torch.tensor(self.los_status, dtype=torch.bool)

        def gap_summary(values: torch.Tensor) -> dict[str, Any]:
            finite = values[torch.isfinite(values)]
            summary: dict[str, Any] = {
                "samples": int(values.numel()),
                "finite_samples": int(finite.numel()),
                "infinite_samples": int(torch.isinf(values).sum()),
            }
            if finite.numel():
                summary.update(
                    {
                        "mean": float(finite.mean()),
                        "p05": float(torch.quantile(finite, 0.05)),
                        "median": float(torch.quantile(finite, 0.50)),
                        "p95": float(torch.quantile(finite, 0.95)),
                    }
                )
            else:
                summary.update(
                    {"mean": None, "p05": None, "median": None, "p95": None}
                )
            return summary

        per_map_gap = {
            str(map_id): gap_summary(
                gaps[torch.tensor(self.map_ids, dtype=torch.long) == map_id]
            )
            for map_id in sorted(map_histogram)
        }
        result["top2_power_gap_db"] = {
            "overall": gap_summary(gaps),
            "los": gap_summary(gaps[los]),
            "nlos": gap_summary(gaps[~los]),
            "per_map": per_map_gap,
        }
        return result


def make_aod_conditioned_datasets(
    datasets: Mapping[str, Dataset],
    *,
    path_count: int | None = 5,
    minimum_concentration: float = 0.10,
    reject_low_concentration: bool = True,
    condition_type: str = "mean",
) -> dict[str, AoDConditionedDataset]:
    return {
        split: AoDConditionedDataset(
            dataset,
            path_count=path_count,
            minimum_concentration=minimum_concentration,
            reject_low_concentration=reject_low_concentration,
            condition_type=condition_type,
        )
        for split, dataset in datasets.items()
    }
