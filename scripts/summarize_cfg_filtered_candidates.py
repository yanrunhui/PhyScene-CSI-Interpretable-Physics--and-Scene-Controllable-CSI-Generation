#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NONREGRESSION_METRICS = (
    "path_level_power_aod_sliced_wasserstein",
    "link_level_power_aod_sliced_wasserstein",
    "power_weighted_aod_sliced_wasserstein",
    "tx_spatial_correlation_relative_frobenius",
    "phase_invariant_full_csi_projected_mmd_rbf",
    "phase_invariant_full_csi_projected_sliced_wasserstein",
    "pdp_sliced_wasserstein",
    "multipath_precision",
    "multipath_density",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.report).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    comparisons = report["cfg2_vs_cfg3_cluster_bootstrap"]
    cfg2_success = report["success"]["cfg2_filtered"]["output_success_rate"]
    recall = comparisons["multipath_recall"]
    self_recall = comparisons["self_centered_multipath_recall"]
    regressions = [
        metric
        for metric in NONREGRESSION_METRICS
        if comparisons[metric]["evidence"] == "cfg2_worse"
    ]
    if (
        cfg2_success >= 0.99
        and recall["evidence"] == "cfg2_better"
        and self_recall["evidence"] == "cfg2_better"
        and not regressions
    ):
        classification = "cfg2_filtered_preferred"
    elif cfg2_success < 0.95:
        classification = "cfg2_filtered_success_rate_insufficient"
    else:
        classification = "cfg2_cfg3_filtered_tradeoff_or_inconclusive"
    summary: dict[str, Any] = {
        "format_version": 1,
        "experiment": "cfg2_cfg3_unbiased_16_to_8_decision_summary",
        "classification": classification,
        "success": report["success"],
        "common_success_rate": report["common_success_rate"],
        "maps": report["maps_in_fidelity_macro"],
        "map_angle_groups": report["fidelity_map_angle_groups"],
        "key_comparisons": {
            metric: comparisons[metric]
            for metric in (
                "multipath_precision",
                "multipath_density",
                "multipath_recall",
                "multipath_coverage",
                "self_centered_multipath_precision",
                "self_centered_multipath_density",
                "self_centered_multipath_recall",
                "self_centered_multipath_coverage",
                *NONREGRESSION_METRICS[:7],
            )
        },
        "statistically_detected_cfg2_nonregression_failures": regressions,
        "macro_per_map": report["macro_per_map"],
        "lowest_cfg2_map_angle_success_groups": sorted(
            report["map_angle_success"],
            key=lambda item: item["success_rate"]["cfg2_filtered"],
        )[:20],
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
