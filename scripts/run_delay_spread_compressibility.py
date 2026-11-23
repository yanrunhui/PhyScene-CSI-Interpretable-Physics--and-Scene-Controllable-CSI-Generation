#!/usr/bin/env python3
"""Phase 1/2: prepared RayVerse CSI check and paired delay intervention."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.data.dataset import PreparedShardDataset
from physics_csi.downstream.compressibility.core import (
    K_VALUES, PATH_FIELDS, analyze, intervene_delays, physical_summary,
)
from physics_csi.physics.synthesizer import MIMOOFDMSynthesizer, SystemConfig


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthesize(sample: dict, model: MIMOOFDMSynthesizer) -> np.ndarray:
    with torch.no_grad():
        csi = model(*(sample[key][None] for key in PATH_FIELDS))[0, 0]
    return csi.cpu().numpy()


def select_indices(dataset: PreparedShardDataset, count: int, *, rng: np.random.Generator,
                   targets: tuple[float, ...] | None, system: SystemConfig,
                   los_only: bool, k_min: float | None, k_max: float | None,
                   path_count: int | None) -> tuple[list[int], dict[str, int]]:
    chosen: list[int] = []
    rejection = {"zero_spread": 0, "alias_guard": 0, "los": 0, "k_factor": 0, "path_count": 0, "truncated": 0}
    # Shuffle shard order and the examples within each shard. This avoids
    # reloading a large .pt shard for every random candidate.
    for shard in rng.permutation(len(dataset.entries)):
        start = 0 if shard == 0 else dataset.ends[shard - 1]
        stop = dataset.ends[shard]
        for index in start + rng.permutation(stop - start):
            sample = dataset[int(index)]
            if bool(sample.get("path_count_truncated", False)):
                rejection["truncated"] += 1
                continue
            summary = physical_summary(sample)
            if path_count is not None and summary["path_count"] != path_count:
                rejection["path_count"] += 1
                continue
            if los_only and not summary["los"]:
                rejection["los"] += 1
                continue
            if k_min is not None and not (k_min <= summary["k_factor_db"] <= k_max):
                rejection["k_factor"] += 1
                continue
            if targets:
                if summary["actual_delay_spread_ns"] <= 1e-7:
                    rejection["zero_spread"] += 1
                    continue
                try:
                    for target in targets:
                        intervene_delays(sample, target, system)
                except ValueError:
                    rejection["alias_guard"] += 1
                    continue
            chosen.append(int(index))
            if len(chosen) >= count:
                return chosen, rejection
    if len(chosen) < count:
        raise RuntimeError(f"Only {len(chosen)}/{count} eligible samples; rejection counts: {rejection}. Relax filters or target delays.")
    return chosen, rejection


def add_result(sample: dict, h: np.ndarray, experiment: str, target: float,
               system: SystemConfig, per_sample: list[dict], curves: list[dict]) -> np.ndarray:
    stats, rows, ad = analyze(h, system)
    sample_id = str(sample["sample_id"])
    common = {
        "sample_id": sample_id if experiment == "phase1" else f"{sample_id}-ds_{target:g}",
        "base_path_id": sample_id, "map_id": int(sample["map_id"]),
        "experiment_type": experiment,
        "target_delay_spread_ns": target,
    }
    per_sample.append({**common, **physical_summary(sample), **stats})
    curves.extend({**common, **row} for row in rows)
    return ad


def _summary(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "q05": float(np.quantile(values, .05)), "q95": float(np.quantile(values, .95))}


def paired_report(rows: list[dict], curves: list[dict], targets: tuple[float, ...],
                  rng: np.random.Generator, bootstrap: int) -> dict:
    by_base = {(r["base_path_id"], r["target_delay_spread_ns"]): r for r in rows}
    base_ids = sorted({r["base_path_id"] for r in rows})
    result = {"groups": {}, "paired_against_first_target": {}, "nmse_by_target_and_k": {}}
    for target in targets:
        values = np.asarray([by_base[(base, target)]["k99"] for base in base_ids], dtype=float)
        result["groups"][str(target)] = _summary(values)
        result["nmse_by_target_and_k"][str(target)] = {}
        for k in K_VALUES:
            linear = np.asarray([r["nmse_linear"] for r in curves if r["target_delay_spread_ns"] == target and r["k"] == k])
            db = 10 * np.log10(np.maximum(linear, 1e-30))
            result["nmse_by_target_and_k"][str(target)][str(k)] = {
                "sample_nmse_db": _summary(db),
                "mean_linear_nmse_then_db": float(10 * np.log10(max(linear.mean(), 1e-30))),
            }
        if target == targets[0]:
            continue
        first = np.asarray([by_base[(base, targets[0])]["k99"] for base in base_ids], dtype=float)
        diff = values - first
        boot = np.mean(diff[rng.integers(0, len(diff), size=(bootstrap, len(diff)))], axis=1)
        result["paired_against_first_target"][str(target)] = {
            **_summary(diff), "mean_bootstrap_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
        }
    return result


def make_plots(output: Path, rows: list[dict], curves: list[dict], targets: tuple[float, ...],
               example_ad: dict[float, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [[r for r in rows if r["target_delay_spread_ns"] == target] for target in targets]
    fig, ax = plt.subplots()
    ax.boxplot([[r["k99"] for r in group] for group in groups], labels=[str(x) for x in targets])
    ax.set(xlabel="Target RMS delay spread (ns)", ylabel="k99")
    fig.savefig(output / "fig1_delay_vs_k99.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots()
    for k in (64, 128, 256, 512):
        med = [np.median([r["nmse_db"] for r in curves if r["k"] == k and r["target_delay_spread_ns"] == t]) for t in targets]
        ax.plot(targets, med, marker="o", label=f"k={k}")
    ax.set(xlabel="Target RMS delay spread (ns)", ylabel="Median sample NMSE (dB)"); ax.legend()
    fig.savefig(output / "fig2_delay_vs_nmse.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots()
    for t in targets:
        med = [np.median([r["nmse_db"] for r in curves if r["k"] == k and r["target_delay_spread_ns"] == t]) for k in K_VALUES]
        ax.plot(np.asarray(K_VALUES) / 4096, med, marker=".", label=f"{t:g} ns")
    ax.set(xscale="log", xlabel="Retained coefficient fraction", ylabel="Median sample NMSE (dB)"); ax.legend()
    fig.savefig(output / "fig3_retention_vs_nmse.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, len(targets), figsize=(3 * len(targets), 3), sharey=True)
    for ax, t in zip(np.atleast_1d(axs), targets):
        image = ax.imshow(10 * np.log10(np.abs(example_ad[t].reshape(64, 64)) ** 2 + 1e-14), origin="lower", vmin=-80, vmax=0, aspect="auto")
        ax.set_title(f"{t:g} ns"); ax.set(xlabel="Delay bin", ylabel="Flattened angle bin")
    fig.colorbar(image, ax=np.atleast_1d(axs).tolist(), label="Coefficient energy (dB)")
    fig.savefig(output / "fig4_angle_delay_energy.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    # Retain the full angle-delay tensors for inspecting delay leakage, not just the projection.
    np.savez_compressed(output / "example_angle_delay.npz", **{f"ds_{t:g}": ad for t, ad in example_ad.items()})

    lookup = {(r["base_path_id"], r["target_delay_spread_ns"]): r["k99"] for r in rows}
    ids = sorted({r["base_path_id"] for r in rows})
    fig, ax = plt.subplots()
    for t in targets[1:]:
        differences = [lookup[(base, t)] - lookup[(base, targets[0])] for base in ids]
        ax.boxplot(differences, positions=[t], widths=max(2, .045 * (max(targets) - min(targets))))
    ax.axhline(0, color="black", lw=.8); ax.set(xlabel=f"Target spread (ns), paired with {targets[0]:g} ns", ylabel="Paired k99 difference")
    fig.savefig(output / "fig5_paired_k99.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", default="artifacts/prepared_d2los")
    parser.add_argument("--output", default="artifacts/compressibility/delay_spread")
    parser.add_argument("--phase", choices=["1", "2", "both"], default="both")
    parser.add_argument("--phase1-count", type=int, default=10)
    parser.add_argument("--base-count", type=int, default=200)
    parser.add_argument("--phase1-split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--phase2-split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--targets-ns", type=float, nargs="+", default=[20, 50, 80, 100, 150])
    parser.add_argument("--los-only", action="store_true", default=True)
    parser.add_argument("--allow-nlos", action="store_false", dest="los_only")
    parser.add_argument("--path-count", type=int, default=5, help="Exact active paths for Phase 2; 0 disables filter")
    parser.add_argument("--k-center-db", type=float, default=10.0, help="LoS K-factor center for Phase 2")
    parser.add_argument("--no-k-filter", action="store_true", help="Disable the Phase 2 K-factor filter")
    parser.add_argument("--k-tolerance-db", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    targets = tuple(sorted(set(args.targets_ns)))
    if not targets or min(targets) <= 0 or args.bootstrap <= 0:
        parser.error("Positive targets and bootstrap count required")
    if args.path_count < 0 or args.k_tolerance_db < 0:
        parser.error("Path count and K tolerance must be nonnegative")
    prepared = Path(args.prepared_root)
    with (prepared / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    system = SystemConfig.from_mapping(manifest["system"])
    model = MIMOOFDMSynthesizer(system).eval()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metadata = {"seed": args.seed, "prepared_root": str(prepared.resolve()), "system": system.as_dict(),
                "phase1_split": args.phase1_split, "phase2_split": args.phase2_split,
                "targets_ns": targets, "k_values": K_VALUES, "bootstrap": args.bootstrap,
                "los_only": args.los_only, "k_center_db": args.k_center_db,
                "path_count_filter": args.path_count or None,
                "k_tolerance_db": args.k_tolerance_db,
                "interpretation": "Physical path intervention; best-k oracle coefficients, no index/quantization/feedback-bit cost."}
    k_min = args.k_center_db - args.k_tolerance_db if not args.no_k_filter else None
    k_max = args.k_center_db + args.k_tolerance_db if not args.no_k_filter else None
    if args.phase in ("1", "both"):
        data = PreparedShardDataset(prepared, args.phase1_split)
        indices, rejected = select_indices(data, args.phase1_count, rng=rng, targets=None, system=system,
                                           los_only=False, k_min=None, k_max=None, path_count=None)
        samples, curves = [], []
        for index in sorted(indices):
            sample = data[index]
            add_result(sample, sample["csi"][0].numpy(), "phase1", math.nan, system, samples, curves)
            # Verify the stored CSI agrees with the actual synthesizer and array mapping.
            regenerated = synthesize(sample, model)
            rel = float(np.linalg.norm(sample["csi"][0].numpy() - regenerated) / np.linalg.norm(regenerated))
            if rel > 1e-5:
                raise AssertionError(f"Stored CSI synthesis mismatch: {rel}")
        write_csv(output / "phase1_per_sample_metrics.csv", samples)
        write_csv(output / "phase1_compression_curves.csv", curves)
        metadata["phase1"] = {"indices": indices, "rejected": rejected, "count": len(samples)}
    if args.phase in ("2", "both"):
        data = PreparedShardDataset(prepared, args.phase2_split)
        indices, rejected = select_indices(data, args.base_count, rng=rng, targets=targets, system=system,
                                           los_only=args.los_only, k_min=k_min, k_max=k_max,
                                           path_count=args.path_count or None)
        samples, curves, example_ad = [], [], {}
        for base_index, index in enumerate(sorted(indices)):
            original = data[index]
            for target in targets:
                changed = intervene_delays(original, target, system)
                ad = add_result(changed, synthesize(changed, model), "paired_delay_intervention", target,
                                system, samples, curves)
                if base_index == 0:
                    example_ad[target] = ad
        write_csv(output / "per_sample_metrics.csv", samples)
        write_csv(output / "compression_curves.csv", curves)
        report = paired_report(samples, curves, targets, rng, args.bootstrap)
        with (output / "paired_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        make_plots(output, samples, curves, targets, example_ad)
        metadata["phase2"] = {"indices": indices, "rejected": rejected, "count": len(samples)}
    with (output / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    print(f"Wrote Phase {args.phase} results to {output}")


if __name__ == "__main__":
    main()
