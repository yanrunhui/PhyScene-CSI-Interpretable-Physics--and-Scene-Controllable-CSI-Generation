#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_csi.utils.io import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the two locked Strongest Anchor Coverage experiments."
    )
    parser.add_argument("--coverage-report", required=True)
    parser.add_argument("--diversity-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    coverage_path = Path(args.coverage_report)
    diversity_path = Path(args.diversity_report)
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    diversity = json.loads(diversity_path.read_text(encoding="utf-8"))
    summary = {
        "experiment": "strongest_anchor_coverage_closeout",
        "cfg": 1,
        "coverage_diagnosis": coverage["diagnosis"],
        "diversity_decision": diversity["decision"],
        "budget_decisions": diversity["budget_decisions"],
        "reports": {
            "coverage_strata": str(coverage_path.resolve()),
            "diversity_selection": str(diversity_path.resolve()),
        },
    }
    atomic_json_dump(summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
