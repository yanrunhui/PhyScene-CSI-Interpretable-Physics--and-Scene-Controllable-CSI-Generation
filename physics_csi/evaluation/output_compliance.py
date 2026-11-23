from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

from physics_csi.data.physical_labels import compute_physical_conditions
from physics_csi.generation_mode import AoDControlMode
from physics_csi.physics.aod_control import AoDControlProjector
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig


@dataclass(frozen=True)
class ComplianceTolerances:
    """Per-sample tolerances for the final generated-path audit."""

    total_power_abs_db: float = 0.1
    delay_spread_abs_ns: float = 0.5
    delay_spread_relative: float = 0.02
    k_factor_abs_db: float = 0.2
    aod_abs_deg: float = 10.0
    csi_reconstruction_relative: float = 1e-5

    def __post_init__(self) -> None:
        values = {
            "total_power_abs_db": self.total_power_abs_db,
            "delay_spread_abs_ns": self.delay_spread_abs_ns,
            "delay_spread_relative": self.delay_spread_relative,
            "k_factor_abs_db": self.k_factor_abs_db,
            "aod_abs_deg": self.aod_abs_deg,
            "csi_reconstruction_relative": self.csi_reconstruction_relative,
        }
        invalid = [
            name
            for name, value in values.items()
            if not math.isfinite(value) or value < 0
        ]
        if invalid:
            raise ValueError(f"compliance tolerances must be nonnegative: {invalid}")
        if self.aod_abs_deg > 180.0:
            raise ValueError("aod_abs_deg must not exceed 180 degrees")


@dataclass(frozen=True)
class ComplianceTargets:
    """User-requested conditions against which an artifact is audited."""

    total_power_db: float
    delay_spread_ns: float
    k_factor_db: float
    k_factor_valid: bool
    los_status: int
    path_count: int
    aod_mode: AoDControlMode
    target_aod_deg: float
    map_id: int | None = None
    expected_samples: int = 8

    def __post_init__(self) -> None:
        finite_values = {
            "total_power_db": self.total_power_db,
            "delay_spread_ns": self.delay_spread_ns,
            "target_aod_deg": self.target_aod_deg,
        }
        if self.k_factor_valid:
            finite_values["k_factor_db"] = self.k_factor_db
        invalid = [
            name for name, value in finite_values.items() if not math.isfinite(value)
        ]
        if invalid:
            raise ValueError(f"compliance targets must be finite: {invalid}")
        if self.delay_spread_ns < 0:
            raise ValueError("delay_spread_ns cannot be negative")
        if self.los_status not in {0, 1}:
            raise ValueError("los_status must be 0 or 1")
        if self.path_count < 1:
            raise ValueError("path_count must be positive")
        if self.expected_samples < 1:
            raise ValueError("expected_samples must be positive")


def _resolve_aod_mode(artifact: Mapping[str, Any]) -> AoDControlMode:
    explicit = artifact.get("aod_mode")
    if explicit is not None:
        if isinstance(explicit, AoDControlMode):
            return explicit
        try:
            return AoDControlMode(str(explicit))
        except ValueError as error:
            raise ValueError(f"unsupported artifact aod_mode: {explicit}") from error

    generation_mode = str(artifact.get("generation_mode", "")).lower()
    if "strongest" in generation_mode:
        return AoDControlMode.STRONGEST_PATH
    if "mean" in generation_mode:
        return AoDControlMode.MEAN_POWER_WEIGHTED

    for container_name in ("control", "control_audit", "generated"):
        container = artifact.get(container_name)
        if not isinstance(container, Mapping):
            continue
        condition_type = str(container.get("condition_type", "")).lower()
        if condition_type == "strongest":
            return AoDControlMode.STRONGEST_PATH
        if condition_type == "mean":
            return AoDControlMode.MEAN_POWER_WEIGHTED
        if container.get("target_strongest_aod_az_deg") is not None:
            return AoDControlMode.STRONGEST_PATH
        if container.get("target_mean_aod_az_deg") is not None:
            return AoDControlMode.MEAN_POWER_WEIGHTED
    raise ValueError(
        "artifact has no aod_mode and its legacy fields do not identify Mean or "
        "Strongest AoD; pass an explicit expected mode through the audit CLI"
    )


def _constant_scalar(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().float().flatten()
    if tensor.numel() == 0:
        return None
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"artifact {field} contains a non-finite value")
    first = tensor[0]
    if not bool(torch.allclose(tensor, first.expand_as(tensor), atol=1e-5, rtol=0.0)):
        raise ValueError(f"artifact {field} is not constant across output samples")
    return float(first)


def _resolve_target_aod(
    artifact: Mapping[str, Any], mode: AoDControlMode
) -> float:
    field = (
        "target_mean_aod_az_deg"
        if mode is AoDControlMode.MEAN_POWER_WEIGHTED
        else "target_strongest_aod_az_deg"
    )
    for container_name, container in (
        ("artifact", artifact),
        ("control_audit", artifact.get("control_audit")),
        ("control", artifact.get("control")),
        ("generated", artifact.get("generated")),
    ):
        if not isinstance(container, Mapping):
            continue
        resolved = _constant_scalar(
            container.get(field), field=f"{container_name}.{field}"
        )
        if resolved is not None:
            return resolved
    raise ValueError(f"artifact does not contain legacy or current {field}")


def targets_from_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_samples: int | None = None,
) -> ComplianceTargets:
    """Read the declared request from a dual-route generation artifact."""
    try:
        conditions = artifact["conditions"]
        validity = torch.as_tensor(artifact["condition_validity"]).bool().flatten()
        mode = _resolve_aod_mode(artifact)
        path_count = int(artifact["path_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("artifact does not contain a valid generation request") from error
    if validity.numel() != 4:
        raise ValueError("artifact condition_validity must contain four values")
    target_aod = _resolve_target_aod(artifact, mode)
    generated = artifact.get("generated")
    if not isinstance(generated, Mapping) or "path_mask" not in generated:
        raise ValueError("artifact does not contain a generated path payload")
    samples = int(torch.as_tensor(generated["path_mask"]).shape[0])
    return ComplianceTargets(
        total_power_db=float(conditions["total_power_db"]),
        delay_spread_ns=float(conditions["delay_spread_ns"]),
        k_factor_db=float(conditions["k_factor_db"]),
        k_factor_valid=bool(validity[2]),
        los_status=int(conditions["los_status"]),
        path_count=path_count,
        aod_mode=mode,
        target_aod_deg=float(target_aod),
        map_id=int(artifact["map_id"]) if artifact.get("map_id") is not None else None,
        expected_samples=samples if expected_samples is None else expected_samples,
    )


def override_targets(
    targets: ComplianceTargets,
    **overrides: Any,
) -> ComplianceTargets:
    """Replace only explicitly supplied (non-None) target values."""
    supplied = {key: value for key, value in overrides.items() if value is not None}
    if "aod_mode" in supplied:
        supplied["aod_mode"] = AoDControlMode(supplied["aod_mode"])
    return replace(targets, **supplied)


def _distribution(values: torch.Tensor) -> dict[str, float | None]:
    flattened = values.detach().float().flatten()
    finite = flattened[torch.isfinite(flattened)]
    if finite.numel() == 0:
        return {"mean": None, "p50": None, "p95": None, "maximum": None}
    return {
        "mean": float(finite.mean()),
        "p50": float(torch.quantile(finite, 0.50)),
        "p95": float(torch.quantile(finite, 0.95)),
        "maximum": float(finite.max()),
    }


def _sample_finiteness(
    payload: Mapping[str, torch.Tensor],
    samples: int,
) -> torch.Tensor:
    finite = torch.ones(samples, dtype=torch.bool)
    for value in payload.values():
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            continue
        is_numeric = value.is_floating_point() or value.is_complex()
        if value.shape[0] != samples or not is_numeric:
            continue
        flattened = value.reshape(samples, -1)
        finite &= torch.isfinite(flattened).all(dim=-1).cpu()
    return finite


def _reconstruct_csi(
    payload: Mapping[str, torch.Tensor],
    system_config: SystemConfig,
) -> torch.Tensor:
    synthesizer = MIMOOFDMSynthesizer(system_config).eval()
    return synthesizer(
        payload["path_delays_ns"].cpu(),
        payload["path_powers_linear"].cpu(),
        payload["path_phases_rad"].cpu(),
        payload["aoa_az_deg"].cpu(),
        payload["aoa_el_deg"].cpu(),
        payload["aod_az_deg"].cpu(),
        payload["aod_el_deg"].cpu(),
        payload["path_mask"].cpu(),
    )


def audit_generated_output(
    artifact: Mapping[str, Any],
    *,
    targets: ComplianceTargets | None = None,
    tolerances: ComplianceTolerances | None = None,
    system_config: SystemConfig | Mapping[str, Any] | None = None,
    check_csi_reconstruction: bool = True,
) -> dict[str, Any]:
    """Audit every final CSI sample against all user-requested scalar conditions.

    The physical conditions and AoD are recomputed from ``artifact['generated']``
    after every residual and rebinding stage.  Map ID is checked as request
    metadata only; distributional Map fidelity requires a separate batch-level
    evaluation.
    """
    requested = targets or targets_from_artifact(artifact)
    limits = tolerances or ComplianceTolerances()
    generated = artifact.get("generated")
    if not isinstance(generated, Mapping):
        raise ValueError("artifact['generated'] must be a tensor mapping")
    required = {
        "path_delays_ns",
        "path_powers_linear",
        "path_mask",
        "direct_path_mask",
        "aod_az_deg",
    }
    missing = sorted(required - generated.keys())
    if missing:
        raise ValueError(f"generated payload is missing required fields: {missing}")
    if check_csi_reconstruction:
        synthesis_fields = {
            "csi",
            "path_phases_rad",
            "aoa_az_deg",
            "aoa_el_deg",
            "aod_el_deg",
        }
        missing_synthesis = sorted(synthesis_fields - generated.keys())
        if missing_synthesis:
            raise ValueError(
                "generated payload is missing CSI reconstruction fields: "
                f"{missing_synthesis}"
            )

    payload = {
        key: value.detach().cpu()
        for key, value in generated.items()
        if isinstance(value, torch.Tensor)
    }
    path_mask = payload["path_mask"].bool()
    if path_mask.ndim != 2:
        raise ValueError("generated path_mask must have shape [sample, path]")
    samples = path_mask.shape[0]
    if samples < 1:
        raise ValueError("generated payload is empty")

    actual_conditions, actual_validity = compute_physical_conditions(
        payload["path_powers_linear"].float(),
        payload["path_delays_ns"].float(),
        path_mask,
        payload["direct_path_mask"].bool(),
    )
    power_error = (actual_conditions[:, 0] - requested.total_power_db).abs()
    delay_error = (actual_conditions[:, 1] - requested.delay_spread_ns).abs()
    delay_limit = max(
        limits.delay_spread_abs_ns,
        abs(requested.delay_spread_ns) * limits.delay_spread_relative,
    )
    k_error = (actual_conditions[:, 2] - requested.k_factor_db).abs()
    actual_k_valid = actual_validity[:, 2].bool()
    if requested.k_factor_valid:
        k_pass = actual_k_valid & (k_error <= limits.k_factor_abs_db)
    else:
        # For NLoS or otherwise invalid K, preserving invalidity is the condition.
        k_pass = ~actual_k_valid
        k_error = torch.full_like(k_error, torch.nan)

    actual_los = actual_conditions[:, 3] >= 0.5
    path_count = path_mask.sum(dim=-1)
    projector_mode = (
        "mean"
        if requested.aod_mode is AoDControlMode.MEAN_POWER_WEIGHTED
        else "strongest"
    )
    aod = AoDControlProjector()(payload, requested.target_aod_deg, mode=projector_mode)
    aod_error = aod.rotation_deg.abs().cpu()

    passes = {
        "finite": _sample_finiteness(payload, samples),
        "total_power": power_error <= limits.total_power_abs_db,
        "delay_spread": delay_error <= delay_limit,
        "k_factor": k_pass,
        "los_status": actual_los == bool(requested.los_status),
        "path_count": path_count == requested.path_count,
        "aod": aod.control_valid.cpu() & (aod_error <= limits.aod_abs_deg),
    }
    requested_pass = torch.stack(list(passes.values()), dim=-1).all(dim=-1)

    reconstruction: dict[str, Any] = {
        "checked": False,
        "available": "csi" in payload,
        "per_sample_relative_error": None,
        "error_distribution": None,
        "pass_rate": None,
        "all_pass": None,
    }
    reconstruction_pass = torch.ones(samples, dtype=torch.bool)
    if check_csi_reconstruction:
        if "csi" not in payload:
            raise ValueError("generated payload has no CSI tensor to reconstruct")
        if system_config is None:
            configured = artifact.get("system_config")
            if configured is None:
                raise ValueError(
                    "CSI reconstruction requires artifact system_config or an explicit config"
                )
            system_config = configured
        resolved_system = (
            system_config
            if isinstance(system_config, SystemConfig)
            else SystemConfig.from_mapping(system_config)
        )
        reconstructed = _reconstruct_csi(payload, resolved_system)
        saved = payload["csi"].cpu()
        if reconstructed.shape != saved.shape:
            raise ValueError(
                "reconstructed CSI shape does not match saved CSI: "
                f"{tuple(reconstructed.shape)} != {tuple(saved.shape)}"
            )
        difference = torch.linalg.vector_norm(
            (saved - reconstructed).reshape(samples, -1), dim=-1
        )
        denominator = torch.linalg.vector_norm(
            reconstructed.reshape(samples, -1), dim=-1
        ).clamp_min(1e-20)
        relative = (difference / denominator).float()
        reconstruction_pass = torch.isfinite(relative) & (
            relative <= limits.csi_reconstruction_relative
        )
        reconstruction = {
            "checked": True,
            "available": True,
            "per_sample_relative_error": relative.tolist(),
            "error_distribution": _distribution(relative),
            "pass_rate": float(reconstruction_pass.float().mean()),
            "all_pass": bool(reconstruction_pass.all()),
        }

    full_pass = requested_pass & reconstruction_pass
    artifact_map_id = artifact.get("map_id")
    map_metadata_pass = requested.map_id is None or (
        artifact_map_id is not None and int(artifact_map_id) == requested.map_id
    )
    sample_count_pass = samples == requested.expected_samples

    target_values = {
        "total_power_db": requested.total_power_db,
        "delay_spread_ns": requested.delay_spread_ns,
        "k_factor_db": requested.k_factor_db if requested.k_factor_valid else None,
        "k_factor_valid": requested.k_factor_valid,
        "los_status": requested.los_status,
        "path_count": requested.path_count,
        "aod_mode": requested.aod_mode.value,
        "target_aod_deg": requested.target_aod_deg,
        "map_id": requested.map_id,
        "expected_samples": requested.expected_samples,
    }
    tolerance_values = {
        "total_power_abs_db": limits.total_power_abs_db,
        "delay_spread_abs_ns": limits.delay_spread_abs_ns,
        "delay_spread_relative": limits.delay_spread_relative,
        "effective_delay_spread_abs_ns": delay_limit,
        "k_factor_abs_db": limits.k_factor_abs_db,
        "aod_abs_deg": limits.aod_abs_deg,
        "csi_reconstruction_relative": limits.csi_reconstruction_relative,
    }

    per_sample: list[dict[str, Any]] = []
    for index in range(samples):
        sample_passes = {name: bool(value[index]) for name, value in passes.items()}
        per_sample.append(
            {
                "sample_index": index,
                "actual": {
                    "total_power_db": float(actual_conditions[index, 0]),
                    "delay_spread_ns": float(actual_conditions[index, 1]),
                    "k_factor_db": (
                        float(actual_conditions[index, 2])
                        if bool(actual_k_valid[index])
                        else None
                    ),
                    "k_factor_valid": bool(actual_k_valid[index]),
                    "los_status": int(actual_los[index]),
                    "path_count": int(path_count[index]),
                    "aod_deg": float(aod.reference_before_deg[index]),
                },
                "absolute_error": {
                    "total_power_db": float(power_error[index]),
                    "delay_spread_ns": float(delay_error[index]),
                    "k_factor_db": (
                        float(k_error[index]) if requested.k_factor_valid else None
                    ),
                    "aod_deg": float(aod_error[index]),
                    "csi_reconstruction_relative": (
                        reconstruction["per_sample_relative_error"][index]
                        if reconstruction["checked"]
                        else None
                    ),
                },
                "pass": sample_passes,
                "all_requested_conditions_pass": bool(requested_pass[index]),
                "full_audit_pass": bool(full_pass[index]),
            }
        )

    metric_tensors = {
        "total_power_abs_error_db": power_error,
        "delay_spread_abs_error_ns": delay_error,
        "k_factor_abs_error_db": k_error,
        "aod_abs_error_deg": aod_error,
    }
    return {
        "format_version": 1,
        "audit": "generated_csi_user_condition_compliance",
        "status": (
            "pass"
            if sample_count_pass
            and map_metadata_pass
            and bool(full_pass.all())
            else "fail"
        ),
        "targets": target_values,
        "tolerances": tolerance_values,
        "sample_count": samples,
        "sample_count_pass": sample_count_pass,
        "map_metadata_pass": map_metadata_pass,
        "metric_error_distribution": {
            name: _distribution(values) for name, values in metric_tensors.items()
        },
        "condition_pass_rate": {
            name: float(value.float().mean()) for name, value in passes.items()
        },
        "all_requested_conditions_pass_rate": float(requested_pass.float().mean()),
        "all_requested_conditions_pass": bool(requested_pass.all()),
        "csi_reconstruction": reconstruction,
        "full_audit_pass_rate": float(full_pass.float().mean()),
        "full_audit_pass": bool(full_pass.all()),
        "request_pass": bool(
            sample_count_pass and map_metadata_pass and full_pass.all()
        ),
        "per_sample": per_sample,
        "map_fidelity_note": (
            "Map ID is a metadata equality check here. Map-conditioned realism must "
            "be evaluated separately on a batch with spatial-correlation, Power-AoD, "
            "PDP, Precision, Recall, and Coverage metrics."
        ),
    }
