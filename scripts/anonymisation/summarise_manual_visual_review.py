#!/usr/bin/env python3
"""Summarise manual visual-review records without inferring rater reliability.

This utility intentionally reports provenance counts only. It must not compute
agreement statistics unless a future, independently rated common subset has
retained source records proving that design.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "outputs/03_anonymisation/14_group2_comparison/13_expanded_structured_visual_review.csv"
DEFAULT_OUT = ROOT / "outputs/03_anonymisation/14_group2_comparison/15_expanded_review_provenance_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    df = pd.read_csv(args.ratings)
    if "provenance_class" not in df.columns:
        raise SystemExit("Missing provenance_class; run expand_structured_visual_review.py first")
    counts = Counter(df["provenance_class"].fillna("unknown").astype(str))
    payload = {
        "status": "provenance_only",
        "n_records": int(len(df)),
        "provenance_counts": dict(sorted(counts.items())),
        "agreement_statistics": "not_computed_mixed_non_independent_sources",
        "canonical_claim": (
            "The author and project partner manually inspected and corrected annotations and relevant "
            "outputs using dedicated review applications. This was a structured project review process, "
            "not a blinded human-participant study."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
