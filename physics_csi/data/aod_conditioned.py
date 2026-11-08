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


class AoDConditionedDataset(Dataset):
    """Attach continuous Mean-AoD labels and optionally retain one path count."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        path_count: int | None = 5,
        minimum_concentration: float = 0.10,
        reject_low_concentration: bool = True,
    ) -> None:
        if path_count is not None and path_count < 1:
            raise ValueError("path_count must be positive or None")
        if not 0.0 <= minimum_concentration <= 1.0:
            raise ValueError("minimum_concentration must lie in [0, 1]")
        self.dataset = dataset
        self.required_path_count = path_count
        self.minimum_concentration = float(minimum_concentration)
        self.reject_low_concentration = bool(reject_low_concentration)
        self.indices: list[int] = []
        self.mean_aod_az_deg: list[float] = []
        self.mean_aod_concentration: list[float] = []
        self.map_ids: list[int] = []
        self.los_status: list[int] = []
        self.dropped_path_count = 0
        self.dropped_low_concentration = 0
        for index in range(len(dataset)):
            record = dataset[index]
            count = int(record["path_count"])
            if path_count is not None and count != path_count:
                self.dropped_path_count += 1
                continue
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
        return {
            "input_samples": len(self.dataset),
            "retained_samples": len(self),
            "required_path_count": self.required_path_count,
            "minimum_concentration": self.minimum_concentration,
            "reject_low_concentration": self.reject_low_concentration,
            "dropped_path_count": self.dropped_path_count,
            "dropped_low_concentration": self.dropped_low_concentration,
            "los_samples": sum(self.los_status),
            "nlos_samples": len(self.los_status) - sum(self.los_status),
            "maps": len(map_histogram),
            "per_map_samples": {
                "minimum": min(map_histogram.values()),
                "median": float(torch.tensor(list(map_histogram.values())).median()),
                "maximum": max(map_histogram.values()),
            },
            "concentration": {
                "mean": float(concentrations.mean()),
                "p05": float(torch.quantile(concentrations, 0.05)),
                "median": float(torch.quantile(concentrations, 0.50)),
                "p95": float(torch.quantile(concentrations, 0.95)),
            },
            "angle_histogram_30deg": {
                f"[{lower},{lower + 30})": int((angle_bin == number).sum())
                for number, lower in enumerate(range(-180, 180, 30))
            },
        }


def make_aod_conditioned_datasets(
    datasets: Mapping[str, Dataset],
    *,
    path_count: int | None = 5,
    minimum_concentration: float = 0.10,
    reject_low_concentration: bool = True,
) -> dict[str, AoDConditionedDataset]:
    return {
        split: AoDConditionedDataset(
            dataset,
            path_count=path_count,
            minimum_concentration=minimum_concentration,
            reject_low_concentration=reject_low_concentration,
        )
        for split, dataset in datasets.items()
    }
