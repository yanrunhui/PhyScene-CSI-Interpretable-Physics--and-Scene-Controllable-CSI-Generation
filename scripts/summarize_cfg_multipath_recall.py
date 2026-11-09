#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _model_summary(report: dict[str, Any], label: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for view in ("absolute", "target_centered", "self_centered", "without_aod"):
        section = report["macro_per_map"][label][view]
        output[view] = {
            "manifold": {
                key: section["manifold"][key]
                for key in (
                    "precision",
                    "density",
                    "recall",
                    "coverage",
                    "rare_reference_coverage",
                )
            },
            "reference_partitions": section["reference_partitions"],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.report).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    comparisons = report["cfg3_vs_cfg2_paired_per_map"]
    summary = {
        "format_version": 1,
        "diagnostic": "cfg2_cfg3_multipath_recall_decision_summary",
        "condition_type": report.get("condition_type", "mean"),
        "samples": report["samples"],
        "maps": report["maps"],
        "decision": report["decision"],
        "macro_per_map": {
            label: _model_summary(report, label)
            for label in ("real_query", "aod_cfg2_dp", "aod_cfg3_dp")
        },
        "cfg3_minus_cfg2_key_deltas": {
            view: {
                metric: comparisons[view]["manifold"][metric]
                for metric in (
                    "precision",
                    "density",
                    "recall",
                    "coverage",
                    "rare_reference_coverage",
                )
            }
            for view in (
                "absolute",
                "target_centered",
                "self_centered",
                "without_aod",
            )
        },
        "absolute_reference_partition_deltas": comparisons["absolute"][
            "reference_partitions"
        ],
        "by_los_available": sorted(report["by_los"]),
        "source_report": str(Path(args.report).resolve()),
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
