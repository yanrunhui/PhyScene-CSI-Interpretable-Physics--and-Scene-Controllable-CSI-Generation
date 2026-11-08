#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact AoD-conditioning diagnostics.")
    parser.add_argument("--cfg-report", required=True)
    parser.add_argument("--gradient-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.cfg_report).open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    with Path(args.gradient_report).open("r", encoding="utf-8") as handle:
        gradient = json.load(handle)

    rows: list[dict[str, Any]] = []
    summaries = cfg["summaries_by_guidance_scale"]
    for scale, section in summaries.items():
        for los_status, stages in section["by_los"].items():
            for stage, metrics in stages.items():
                if not metrics:
                    continue
                rows.append(
                    {
                        "guidance_scale": float(scale),
                        "los_status": int(los_status),
                        "stage": stage,
                        "mean_error_deg": metrics["mean_aod_error_deg"]["mean"],
                        "p95_cell_weighted_deg": metrics["mean_aod_error_deg"][
                            "p95_cell_weighted"
                        ],
                        "hit_at_5deg": metrics["hit_at_5deg"],
                        "hit_at_10deg": metrics["hit_at_10deg"],
                        "relative_aod_spread_deg": metrics[
                            "within_link_relative_aod_spread_deg"
                        ],
                        "csi_dispersion": metrics["normalized_csi_rms_dispersion"],
                        "physical_feasible_rate": metrics[
                            "constraint_feasible_rate"
                        ],
                    }
                )
    best_by_los: dict[str, Any] = {}
    for los_status in sorted({row["los_status"] for row in rows}):
        base_rows = [
            row
            for row in rows
            if row["los_status"] == los_status
            and row["stage"] == "aod_conditioned_b3"
        ]
        if base_rows:
            best_by_los[str(los_status)] = min(
                base_rows, key=lambda item: item["mean_error_deg"]
            )

    aggregate = gradient["aggregate_mean"]
    gradient_groups = aggregate["gradient_groups"]
    summary = {
        "format_version": 1,
        "diagnostic": "aod_conditioning_decision_summary",
        "cfg_scope": cfg["cfg_implementation_audit"],
        "gradient_audit": {
            "diffusion_loss": aggregate["diffusion_loss"],
            "aod_loss_unweighted": aggregate["aod_loss_unweighted"],
            "aod_loss_weighted": aggregate["aod_loss_weighted"],
            "aod_supervision_fraction": aggregate["aod_supervision_fraction"],
            "all_parameters": gradient_groups["all"],
            "denoiser": gradient_groups.get("denoiser"),
            "aod_conditioner": gradient_groups.get("aod_conditioner"),
            "implementation": gradient["implementation_audit"],
        },
        "best_cfg_by_los_for_direct_aod_conditioned_b3": best_by_los,
        "comparison_rows": rows,
        "stage_interpretation": cfg["stage_interpretation"],
        "source_reports": {
            "cfg": str(Path(args.cfg_report).resolve()),
            "gradients": str(Path(args.gradient_report).resolve()),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
