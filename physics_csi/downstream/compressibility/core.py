from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from physics_csi.data.physical_labels import compute_physical_conditions
from physics_csi.physics.synthesizer import SystemConfig


K_VALUES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
PATH_FIELDS = (
    "path_delays_ns", "path_powers_linear", "path_phases_rad",
    "aoa_az_deg", "aoa_el_deg", "aod_az_deg", "aod_el_deg", "path_mask",
)


def check_system(system: SystemConfig) -> None:
    if system.tx_array_shape != (8, 8) or system.rx_array_shape != (1, 1) or system.num_subcarriers != 64:
        raise ValueError("This protocol requires Tx 8x8, Rx 1x1, and 64 subcarriers")
    if system.tx_array_plane != "xy":
        raise ValueError("Spatial DFT protocol assumes the configured xy UPA")


def forward_transform(h: np.ndarray, system: SystemConfig) -> np.ndarray:
    check_system(system)
    h = np.asarray(h)
    if h.shape != (64, 64) or not np.iscomplexobj(h) or not np.isfinite(h).all():
        raise ValueError("CSI must be finite complex [64,64]")
    # array_positions_wavelengths uses meshgrid(indexing='ij').reshape(rows*cols,3).
    grid = h.reshape(*system.tx_array_shape, system.num_subcarriers)
    return np.fft.ifft(np.fft.fft2(grid, axes=(0, 1), norm="ortho"), axis=2, norm="ortho")


def inverse_transform(ad: np.ndarray, system: SystemConfig) -> np.ndarray:
    check_system(system)
    if ad.shape != (8, 8, 64):
        raise ValueError("Angle-delay tensor must be [8,8,64]")
    return np.fft.ifft2(np.fft.fft(ad, axis=2, norm="ortho"), axes=(0, 1), norm="ortho").reshape(64, 64)


def analyze(h: np.ndarray, system: SystemConfig, *, k_values: tuple[int, ...] = K_VALUES) -> tuple[dict[str, float], list[dict[str, float]], np.ndarray]:
    check_system(system)
    h = np.asarray(h, dtype=np.complex128)
    norm = float(np.linalg.norm(h))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("CSI has zero or invalid energy")
    h = h / norm  # common unit Frobenius norm, independently per CSI
    ad = forward_transform(h, system)
    energy = np.abs(ad.ravel()) ** 2
    total = float(energy.sum())
    if not np.isclose(total, 1.0, rtol=1e-10, atol=1e-10):
        raise AssertionError("Unitary transform did not preserve energy")
    indices = np.argsort(-energy, kind="stable")
    retained = np.cumsum(energy[indices])
    p = energy[energy > 0] / total
    stats = {
        "pre_normalization_csi_energy": norm * norm,
        "k99": int(np.searchsorted(retained, .99 * total, side="left") + 1),
        "k999": int(np.searchsorted(retained, .999 * total, side="left") + 1),
        "effective_sparsity": float(np.exp(-(p * np.log(p)).sum())),
        "transform_relative_energy_error": abs(total - 1.0),
        "full_transform_relative_error": float(np.linalg.norm(h - inverse_transform(ad, system)) ** 2),
    }
    curves = []
    previous = math.inf
    for k in k_values:
        if not 1 <= k <= energy.size:
            raise ValueError(f"Invalid k={k}")
        sparse = np.zeros(energy.size, dtype=np.complex128)
        sparse[indices[:k]] = ad.ravel()[indices[:k]]
        reconstructed = inverse_transform(sparse.reshape(ad.shape), system)
        nmse = float(np.linalg.norm(h - reconstructed) ** 2)
        discarded = float(np.sum(energy[indices[k:]]) / total)
        if not np.isclose(nmse, discarded, rtol=1e-5, atol=1e-11):
            raise AssertionError("Reconstruction and discarded-energy NMSE differ")
        if nmse > previous + 1e-11:
            raise AssertionError("Best-k error increased with k")
        previous = nmse
        curves.append({
            "k": k, "retention_ratio": k / energy.size,
            "nmse_linear": nmse,
            "nmse_db": 10 * math.log10(max(nmse, 1e-30)),
            "retained_energy_ratio": float(np.sum(energy[indices[:k]]) / total),
        })
    if stats["full_transform_relative_error"] > 1e-10:
        raise AssertionError("Full transform inverse failed")
    return stats, curves, ad


def physical_summary(sample: dict[str, Any]) -> dict[str, float | int]:
    mask = sample["path_mask"].bool()
    direct = sample["direct_path_mask"].bool()
    powers = sample["path_powers_linear"].float()
    delays = sample["path_delays_ns"].float()
    conditions, validity = compute_physical_conditions(
        powers[None], delays[None], mask[None], direct[None]
    )
    c = conditions[0]
    active_powers = powers[mask].double()
    weights = active_powers / active_powers.sum()
    az = torch.deg2rad(sample["aod_az_deg"][mask].double())
    el = torch.deg2rad(sample["aod_el_deg"][mask].double())
    # Circular azimuth spread handles angles around +/-180 degrees.
    az_spread = float(torch.sqrt(torch.clamp(1 - torch.abs(torch.sum(weights * torch.exp(1j * az))), min=0)).item())
    el_mean = (weights * el).sum()
    el_spread = float(torch.sqrt((weights * (el - el_mean).square()).sum()).item())
    return {
        "actual_delay_spread_ns": float(c[1]), "k_factor_db": float(c[2]) if validity[0, 2] else math.nan,
        "los": int(c[3].item()), "path_count": int(mask.sum()),
        "total_path_power_linear": float(active_powers.sum()),
        "angular_spread_az_rad": az_spread, "angular_spread_el_rad": el_spread,
    }


def intervene_delays(sample: dict[str, Any], target_ns: float, system: SystemConfig, *, guard_fraction: float = 0.9) -> dict[str, Any]:
    if target_ns <= 0:
        raise ValueError("Target RMS delay spread must be positive")
    result = dict(sample)
    before = physical_summary(sample)["actual_delay_spread_ns"]
    if before <= 1e-7:
        raise ValueError("Cannot scale a zero-spread path set")
    delays = sample["path_delays_ns"].clone()
    mask = sample["path_mask"].bool()
    origin = delays[mask].min()
    delays[mask] = origin + (target_ns / before) * (delays[mask] - origin)
    # OFDM frequency sample spacing is B/N; delays modulo N/B alias.
    unambiguous_ns = system.num_subcarriers / system.bandwidth_hz * 1e9
    span_ns = float((delays[mask].max() - delays[mask].min()).item())
    if span_ns >= guard_fraction * unambiguous_ns:
        raise ValueError(f"Delay span {span_ns:.2f} ns exceeds {guard_fraction:.0%} of unambiguous {unambiguous_ns:.2f} ns")
    result["path_delays_ns"] = delays
    actual = physical_summary(result)["actual_delay_spread_ns"]
    if not math.isclose(actual, target_ns, rel_tol=2e-5, abs_tol=2e-3):
        raise AssertionError(f"Delay spread intervention missed target: {actual} vs {target_ns}")
    return result
