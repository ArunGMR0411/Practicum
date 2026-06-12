#!/usr/bin/env python3
"""Classify retained structured visual-review rows by truthful provenance.

The historical expansion mixed manual, inherited, and image-heuristic rows.
This migration is deliberately table-only: it does not inspect images, create
ratings, or calculate inter-rater agreement.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/03_anonymisation/14_group2_comparison"
REVIEW = OUT / "13_expanded_structured_visual_review.csv"
ROLLUP = OUT / "14_expanded_review_provenance_rollup.csv"
SUMMARY = OUT / "15_expanded_review_provenance_summary.json"


def classify(source: str, reviewer: str) -> tuple[str, str, str]:
    source = source.lower()
    if source.startswith(("partner_residual_edge", "author_residual_edge")):
        return "heuristic_or_automatically_generated", "no", "yes"
    if source.startswith("partner_align_author_fail"):
        return "inherited_or_reused_review_result", "no", "yes"
    if source in {"dual20_partner_human", "diffusion_quality_pass2", "partner_five_case_independent"}:
        return "genuine_manual_partner_review", "not_established", "no"
    if reviewer == "author":
        return "structured_author_inspection", "no", "no"
    return "joint_or_adjudicated_manual_review", "no", "no"


def main() -> None:
    rows = list(csv.DictReader(REVIEW.open(encoding="utf-8")))
    for row in rows:
        pclass, independent, automated = classify(row.get("rating_source", ""), row.get("reviewer_id", ""))
        row["review_scope"] = "expanded_structured_visual_review"
        row["provenance_class"] = pclass
        row["independent"] = independent
        row["automated_assistance"] = automated
    fields = list(rows[0])
    with REVIEW.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    by_method: dict[str, Counter[str]] = defaultdict(Counter)
    author_manual: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        method = row["method"]
        by_method[method][row["provenance_class"]] += 1
        if row["provenance_class"] == "structured_author_inspection":
            author_manual[method].append(int(row["obvious_artifact"]))
    rollup = []
    for method in sorted(by_method):
        counts = by_method[method]
        vals = author_manual.get(method, [])
        rollup.append({
            "method": method,
            "n_records": sum(counts.values()),
            "manual_author_records": counts["structured_author_inspection"],
            "manual_partner_records": counts["genuine_manual_partner_review"],
            "joint_records": counts["joint_or_adjudicated_manual_review"],
            "inherited_records": counts["inherited_or_reused_review_result"],
            "heuristic_records": counts["heuristic_or_automatically_generated"],
            "structured_author_fail_rate": round(sum(vals) / len(vals), 6) if vals else "",
            "agreement_statistics": "not_computed",
            "default_route_recommendation": "RESEARCH_ONLY_NOT_DEFAULT",
        })
    with ROLLUP.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rollup[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rollup)

    total = Counter(row["provenance_class"] for row in rows)
    payload = {
        "status": "provenance_corrected",
        "n_records": len(rows),
        "methods": sorted(by_method),
        "provenance_counts": dict(sorted(total.items())),
        "agreement_statistics": "removed_mixed_non_independent_sources",
        "default_blocking": "All generative methods remain RESEARCH_ONLY_NOT_DEFAULT.",
        "canonical_provenance": "outputs/03_anonymisation/19_visual_eligibility_gate/05_manual_review_provenance.csv",
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
