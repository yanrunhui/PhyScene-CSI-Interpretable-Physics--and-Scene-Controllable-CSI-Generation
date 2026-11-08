from __future__ import annotations

import torch
from torch.utils.data import Dataset

from physics_csi.data.aod_conditioned import (
    AoDConditionedDataset,
    power_weighted_mean_aod,
)
from physics_csi.data.scene_id import LocalityAwareBlockShuffleSampler


class _Records(Dataset):
    def __init__(self, records: list[dict[str, torch.Tensor]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.records[index]


def _record(angles: list[float], powers: list[float], count: int = 5) -> dict[str, torch.Tensor]:
    return {
        "path_count": torch.tensor(count),
        "path_mask": torch.ones(5, dtype=torch.bool),
        "path_powers_linear": torch.tensor(powers),
        "aod_az_deg": torch.tensor(angles),
        "map_id": torch.tensor(10),
        "conditions": torch.tensor([-80.0, 30.0, 0.0, 0.0]),
    }


def test_power_weighted_mean_aod_wraps_at_180_degrees() -> None:
    angle, concentration = power_weighted_mean_aod(
        torch.tensor([0.5, 0.5]),
        torch.tensor([179.0, -179.0]),
        torch.tensor([True, True]),
    )
    assert abs(abs(float(angle)) - 180.0) < 1e-4
    assert float(concentration) > 0.99


def test_aod_conditioned_dataset_filters_unstable_and_non_l5_records() -> None:
    stable = _record([30.0] * 5, [0.4, 0.2, 0.15, 0.15, 0.1])
    cancelling = _record([0.0, 180.0, 0.0, 180.0, 0.0], [0.25, 0.25, 0.25, 0.25, 0.0])
    wrong_count = _record([45.0] * 5, [0.2] * 5, count=4)
    dataset = AoDConditionedDataset(
        _Records([stable, cancelling, wrong_count]),
        path_count=5,
        minimum_concentration=0.1,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert abs(float(sample["mean_aod_az_deg"]) - 30.0) < 1e-4
    assert bool(sample["mean_aod_valid"])
    audit = dataset.audit()
    assert audit["dropped_low_concentration"] == 1
    assert audit["dropped_path_count"] == 1


def test_locality_aware_sampler_is_reproducible_and_complete() -> None:
    records = _Records([_record([30.0] * 5, [0.2] * 5) for _ in range(17)])
    sampler = LocalityAwareBlockShuffleSampler(records, block_size=5, seed=11)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    second = list(sampler)
    assert first == second
    assert sorted(first) == list(range(17))
    sampler.set_epoch(4)
    assert list(sampler) != first
