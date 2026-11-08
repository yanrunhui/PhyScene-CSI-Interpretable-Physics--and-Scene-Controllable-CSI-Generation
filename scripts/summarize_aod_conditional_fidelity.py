#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEY_METRICS = (
    "path_level_power_aod_sliced_wasserstein",
    "link_level_power_aod_sliced_wasserstein",
    "power_weighted_aod_sliced_wasserstein",
    "tx_spatial_correlation_relative_frobenius",
    "phase_invariant_full_csi_projected_mmd_rbf",
    "phase_invariant_full_csi_projected_sliced_wasserstein",
    "pdp_sliced_wasserstein",
    "multipath_recall",
    "multipath_coverage",
)


def _compact(report: dict[str, Any], labels: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in labels:
        section = report["macro_per_map"].get(label)
        if not section:
            continue
        control = dict(section["control"])
        if "hit_at_10deg" in control:
            control["expected_candidates_per_accepted_at_10deg"] = 1.0 / max(
                float(control["hit_at_10deg"]), 1e-12
            )
        result[label] = {
            "control": control,
            "fidelity": {
                metric: section["fidelity"][metric] for metric in KEY_METRICS
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-validation", required=True)
    parser.add_argument("--support-test", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    values = {}
    for name, path in (
        ("support_validation", args.support_validation),
        ("support_test", args.support_test),
        ("validation", args.validation),
        ("test", args.test),
    ):
        with Path(path).open("r", encoding="utf-8") as handle:
            values[name] = json.load(handle)
    selection = values["validation"]["validation_selection"]
    selected = {
        "baseline_b3_dp_link",
        "aod_cfg1_dp",
        "aod_cfg2_dp",
        "aod_cfg3_dp",
        "aod_cfg2_dp_link",
        "aod_cfg3_dp_link",
        selection["unified"]["selected_label"],
        selection["by_los"]["0"]["selected_label"],
        selection["by_los"]["1"]["selected_label"],
        "real_query",
    }
    summary = {
        "format_version": 1,
        "diagnostic": "aod_conditional_fidelity_validation_test_summary",
        "support": {
            split: values[f"support_{split}"][
                "support_summary_for_selected_path_count"
            ]
            for split in ("validation", "test")
        },
        "validation_selection": selection,
        "test_used_frozen_validation_selection": values["test"][
            "frozen_validation_selection_used_for_test"
        ],
        "validation_key_results": _compact(values["validation"], selected),
        "test_key_results": _compact(values["test"], selected),
        "source_reports": {
            "support_validation": str(Path(args.support_validation).resolve()),
            "support_test": str(Path(args.support_test).resolve()),
            "validation": str(Path(args.validation).resolve()),
            "test": str(Path(args.test).resolve()),
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
