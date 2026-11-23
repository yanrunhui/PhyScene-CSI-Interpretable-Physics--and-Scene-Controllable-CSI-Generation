import numpy as np
import torch
import json
import subprocess
import sys
from pathlib import Path

from physics_csi.downstream.compressibility.core import K_VALUES, analyze, forward_transform, intervene_delays, physical_summary
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig


def test_best_k_transform_and_grid_aligned_single_path():
    system = SystemConfig()
    # Broadside has constant spatial steering; zero delay is one IFFT bin.
    model = MIMOOFDMSynthesizer(system)
    one = torch.ones(1, 1)
    zero = torch.zeros(1, 1)
    h = model(zero, one, zero, zero, zero, zero, zero, one.bool())[0, 0].numpy()
    ad = forward_transform(h, system)
    assert np.count_nonzero(np.abs(ad) > 1e-5) == 1
    stats, curves, _ = analyze(h, system)
    assert stats["k99"] == 1
    assert stats["k999"] == 1
    assert stats["effective_sparsity"] < 1.001
    assert len(curves) == len(K_VALUES)
    assert curves[-1]["nmse_linear"] < 1e-12


def test_paired_delay_intervention_preserves_other_paths():
    system = SystemConfig()
    sample = {
        "path_delays_ns": torch.tensor([10., 30., 80., 0., 0.]),
        "path_powers_linear": torch.tensor([.8, .1, .1, 0., 0.]),
        "path_phases_rad": torch.tensor([.1, .4, -.5, 0., 0.]),
        "aod_az_deg": torch.tensor([0., 20., 40., 0., 0.]),
        "aod_el_deg": torch.zeros(5),
        "path_mask": torch.tensor([True, True, True, False, False]),
        "direct_path_mask": torch.tensor([True, False, False, False, False]),
    }
    original = physical_summary(sample)
    changed = intervene_delays(sample, 50., system)
    actual = physical_summary(changed)
    assert abs(actual["actual_delay_spread_ns"] - 50.) < .002
    assert abs(actual["k_factor_db"] - original["k_factor_db"]) < 1e-6
    assert actual["los"] == original["los"]
    assert torch.equal(changed["path_phases_rad"], sample["path_phases_rad"])
    assert torch.equal(changed["aod_az_deg"], sample["aod_az_deg"])
    assert torch.equal(changed["path_delays_ns"][3:], sample["path_delays_ns"][3:])


def test_phase1_phase2_cli_smoke(tmp_path):
    system = SystemConfig()
    prepared = tmp_path / "prepared"
    (prepared / "train").mkdir(parents=True)
    values = {
        "sample_id": ["synthetic-0"], "map_id": torch.tensor([0]),
        "path_delays_ns": torch.tensor([[10., 30., 80., 0., 0.]]),
        "path_powers_linear": torch.tensor([[.8, .1, .1, 0., 0.]]),
        "path_phases_rad": torch.tensor([[.1, .4, -.5, 0., 0.]]),
        "aod_az_deg": torch.tensor([[0., 20., 40., 0., 0.]]),
        "aod_el_deg": torch.zeros(1, 5),
        "aoa_az_deg": torch.zeros(1, 5),
        "aoa_el_deg": torch.zeros(1, 5),
        "path_mask": torch.tensor([[True, True, True, False, False]]),
        "direct_path_mask": torch.tensor([[True, False, False, False, False]]),
        "path_count_truncated": torch.tensor([False]),
    }
    model = MIMOOFDMSynthesizer(system)
    values["csi"] = model(*(values[key] for key in (
        "path_delays_ns", "path_powers_linear", "path_phases_rad",
        "aoa_az_deg", "aoa_el_deg", "aod_az_deg", "aod_el_deg", "path_mask"
    ))).detach()
    torch.save({"samples": values}, prepared / "train" / "shard_00000.pt")
    (prepared / "manifest.json").write_text(json.dumps({
        "system": system.as_dict(),
        "splits": {"train": {"shards": [{"file": "train/shard_00000.pt", "count": 1}]}}
    }))
    output = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_delay_spread_compressibility.py"
    subprocess.run([sys.executable, str(script), "--prepared-root", str(prepared),
                    "--output", str(output), "--phase", "both", "--phase1-count", "1",
                    "--base-count", "1", "--targets-ns", "20", "50",
                    "--path-count", "3", "--no-k-filter", "--bootstrap", "20"], check=True)
    assert (output / "per_sample_metrics.csv").exists()
    assert (output / "compression_curves.csv").exists()
    assert (output / "fig5_paired_k99.png").exists()
    assert (output / "paired_summary.json").exists()
