#!/usr/bin/env python3

"""Summarise the structured RQ2 qualitative review sheet."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "rq2_qualitative_review.csv"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "rq2_qualitative_review_summary.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "rq2_qualitative_review_method_summary.csv"

SCORE_FIELDS = (
    "privacy_visual_score",
    "utility_visual_score",
    "demographic_consistency_score",
    "pose_context_score",
    "scene_coherence_score",
)

FAIL_FIELDS = (
    "ghost_or_donor_artifact",
    "palette_or_blotchy_artifact",
    "partial_face_coverage",
    "demographic_drift_flag",
    "scene_break_flag",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def to_int(row: dict[str, str], field: str) -> int:
    return int(row[field])


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    method_rows: list[dict[str, object]] = []
    methods_summary: dict[str, object] = {}
    for method, method_entries in sorted(grouped.items()):
        score_means = {
            field: mean([to_int(row, field) for row in method_entries])
            for field in SCORE_FIELDS
        }
        fail_counts = {
            field: sum(to_int(row, field) for row in method_entries)
            for field in FAIL_FIELDS
        }
        promoted_count = sum(1 for row in method_entries if row["promotion_decision"] == "promote")
        demoted_count = sum(1 for row in method_entries if row["promotion_decision"] == "demote")
        retain_count = sum(1 for row in method_entries if row["promotion_decision"] == "retain_bounded")
        methods_summary[method] = {
            "sample_count": len(method_entries),
            "score_means": score_means,
            "failure_counts": fail_counts,
            "promote_count": promoted_count,
            "retain_bounded_count": retain_count,
            "demote_count": demoted_count,
            "final_recommendation": method_entries[0]["method_recommendation"],
            "notes": [row["notes"] for row in method_entries if row["notes"].strip()],
        }
        method_rows.append(
            {
                "method": method,
                "sample_count": len(method_entries),
                **score_means,
                **fail_counts,
                "promote_count": promoted_count,
                "retain_bounded_count": retain_count,
                "demote_count": demoted_count,
                "final_recommendation": method_entries[0]["method_recommendation"],
            }
        )

    summary = {
        "input": str(args.input.relative_to(PROJECT_ROOT)),
        "sample_count": len(rows),
        "methods": methods_summary,
    }
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "sample_count",
                *SCORE_FIELDS,
                *FAIL_FIELDS,
                "promote_count",
                "retain_bounded_count",
                "demote_count",
                "final_recommendation",
            ],
        )
        writer.writeheader()
        writer.writerows(method_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
