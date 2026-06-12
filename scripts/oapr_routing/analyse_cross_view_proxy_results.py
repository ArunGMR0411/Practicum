#!/usr/bin/env python3

"""Analyse cross-view proxy residual-linkability results with paired statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Run a paired t-test and return mean difference plus confidence interval."""
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal length")
    if len(a) < 2:
        return {
            "n": int(len(a)),
            "mean_diff": float(np.mean(a - b)) if len(a) else 0.0,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }

    diffs = a - b
    result = ttest_rel(a, b)
    mean_diff = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    tcrit = float(t_dist.ppf(0.975, df=len(diffs) - 1))
    ci_low = mean_diff - tcrit * se
    ci_high = mean_diff + tcrit * se
    return {
        "n": int(len(a)),
        "mean_diff": mean_diff,
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def summarise_slice(df: pd.DataFrame, slice_col: str) -> list[dict[str, object]]:
    """Summarise mean similarities and evaluated counts by one slice column."""
    rows: list[dict[str, object]] = []
    for key, group in df.groupby(slice_col, dropna=False):
        evaluated = group[group["evaluated"] == True].copy()  # noqa: E712
        rows.append(
            {
                slice_col: "missing" if pd.isna(key) else str(key),
                "pairs_total": int(len(group)),
                "pairs_evaluated": int(len(evaluated)),
                "evaluation_rate": float(len(evaluated) / len(group)) if len(group) else 0.0,
                "mean_original_max_cross_view_cosine": float(evaluated["original_max_cross_view_cosine"].mean())
                if not evaluated.empty
                else 0.0,
                "mean_blur_max_cross_view_cosine": float(evaluated["blur_max_cross_view_cosine"].mean())
                if not evaluated.empty
                else 0.0,
                "mean_pixelate_max_cross_view_cosine": float(evaluated["pixelate_max_cross_view_cosine"].mean())
                if not evaluated.empty
                else 0.0,
                "mean_blur_delta_vs_original": float(
                    (evaluated["blur_max_cross_view_cosine"] - evaluated["original_max_cross_view_cosine"]).mean()
                )
                if not evaluated.empty
                else 0.0,
                "mean_pixelate_delta_vs_original": float(
                    (evaluated["pixelate_max_cross_view_cosine"] - evaluated["original_max_cross_view_cosine"]).mean()
                )
                if not evaluated.empty
                else 0.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="outputs/05_oapr/cross_view_analysis/01_cross_view_proxy_pair_results.csv",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/runs/cross_view/cross_view_proxy_analysis.json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/05_oapr/cross_view_analysis/02_cross_view_proxy_summary_table.csv",
    )
    args = parser.parse_args()

    input_csv = PROJECT_ROOT / args.input_csv
    output_json = PROJECT_ROOT / args.output_json
    output_csv = PROJECT_ROOT / args.output_csv

    df = pd.read_csv(input_csv)
    evaluated = df[df["evaluated"] == True].copy()  # noqa: E712

    original = evaluated["original_max_cross_view_cosine"].to_numpy(dtype=float)
    blur = evaluated["blur_max_cross_view_cosine"].to_numpy(dtype=float)
    pixelate = evaluated["pixelate_max_cross_view_cosine"].to_numpy(dtype=float)
    original_gap = evaluated["original_gap_vs_control"].dropna().to_numpy(dtype=float)
    blur_gap = evaluated["blur_gap_vs_control"].dropna().to_numpy(dtype=float)
    pixelate_gap = evaluated["pixelate_gap_vs_control"].dropna().to_numpy(dtype=float)

    payload = {
        "summary": {
            "pairs_total": int(len(df)),
            "pairs_evaluated": int(len(evaluated)),
            "pairs_skipped": int(len(df) - len(evaluated)),
            "evaluation_rate": float(len(evaluated) / len(df)) if len(df) else 0.0,
            "skip_reason_counts": {
                str(key): int(value)
                for key, value in df["skip_reason"].fillna("").replace("", "evaluated").value_counts().items()
            },
        },
        "paired_tests": {
            "blur_vs_original_cross_view_cosine": paired_test(blur, original),
            "pixelate_vs_original_cross_view_cosine": paired_test(pixelate, original),
            "blur_vs_pixelate_cross_view_cosine": paired_test(blur, pixelate),
        },
        "control_gap_tests": {
            "blur_vs_original_gap": paired_test(blur_gap, original_gap),
            "pixelate_vs_original_gap": paired_test(pixelate_gap, original_gap),
            "blur_vs_pixelate_gap": paired_test(blur_gap, pixelate_gap),
        },
        "slices": {
            "by_day": summarise_slice(df, "day_id"),
            "by_exocentric_stream": summarise_slice(df, "exocentric_stream_id"),
            "by_egocentric_stream": summarise_slice(df, "egocentric_stream_id"),
        },
    }

    summary_table = pd.DataFrame(
        [
            {
                "metric": "mean_original_max_cross_view_cosine",
                "value": float(np.mean(original)) if len(original) else 0.0,
            },
            {
                "metric": "mean_blur_max_cross_view_cosine",
                "value": float(np.mean(blur)) if len(blur) else 0.0,
            },
            {
                "metric": "mean_pixelate_max_cross_view_cosine",
                "value": float(np.mean(pixelate)) if len(pixelate) else 0.0,
            },
            {
                "metric": "mean_original_gap_vs_control",
                "value": float(np.mean(original_gap)) if len(original_gap) else 0.0,
            },
            {
                "metric": "mean_blur_gap_vs_control",
                "value": float(np.mean(blur_gap)) if len(blur_gap) else 0.0,
            },
            {
                "metric": "mean_pixelate_gap_vs_control",
                "value": float(np.mean(pixelate_gap)) if len(pixelate_gap) else 0.0,
            },
        ]
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_table.to_csv(output_csv, index=False)
    print(json.dumps({"analysis_json": str(output_json), "summary_csv": str(output_csv), **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
