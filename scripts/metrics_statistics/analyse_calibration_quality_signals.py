#!/usr/bin/env python3

"""Analyse calibration quality-signal separability before threshold sweeping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def summarise_group(df: pd.DataFrame, label_col: str, signal_col: str) -> dict[str, dict[str, float | int]]:
    grouped = df.groupby(label_col)[signal_col].agg(["count", "median", "mean"]).sort_index()
    summary: dict[str, dict[str, float | int]] = {}
    for label, row in grouped.iterrows():
        summary[str(label)] = {
            "count": int(row["count"]),
            "median": float(row["median"]),
            "mean": float(row["mean"]),
        }
    return summary


def get_medians(summary: dict[str, dict[str, float | int]]) -> dict[str, float]:
    return {label: float(values["median"]) for label, values in summary.items()}


def is_strictly_ordered(medians: dict[str, float], ordered_labels: list[str]) -> bool:
    values = [medians.get(label) for label in ordered_labels]
    if any(value is None for value in values):
        return False
    return all(float(left) < float(right) for left, right in zip(values, values[1:], strict=False))


def gap_ratio(medians: dict[str, float], lower_label: str, upper_label: str) -> float:
    lower = float(medians.get(lower_label, 0.0))
    upper = float(medians.get(upper_label, 0.0))
    if lower <= 0.0:
        return float("inf") if upper > 0.0 else 1.0
    return upper / lower


def gap_delta(medians: dict[str, float], lower_label: str, upper_label: str) -> float:
    lower = float(medians.get(lower_label, 0.0))
    upper = float(medians.get(upper_label, 0.0))
    return upper - lower


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="outputs/calibration_quality_signals.csv")
    parser.add_argument(
        "--output-json",
        default="outputs/calibration_quality_diagnostic_summary.json",
    )
    args = parser.parse_args()

    df = pd.read_csv(PROJECT_ROOT / args.input_csv)
    face_present_df = df[df["face_box_count"] > 0].copy()
    zero_face_df = df[df["face_box_count"] == 0].copy()

    all_blur = summarise_group(df, "blur_level_label", "blur_score")
    all_face_size = summarise_group(df, "face_size_label", "face_size_px")
    all_occlusion = summarise_group(df, "occlusion_ratio_label", "occlusion_ratio")
    all_webp = summarise_group(df, "webp_artefact_severity_label", "webp_artifact_score")
    face_present_face_size = summarise_group(face_present_df, "face_size_label", "face_size_px")
    face_present_occlusion = summarise_group(face_present_df, "occlusion_ratio_label", "occlusion_ratio")

    blur_medians = get_medians(all_blur)
    face_size_medians = get_medians(face_present_face_size)
    occlusion_medians = get_medians(face_present_occlusion)
    webp_medians = get_medians(all_webp)

    blur_ready = is_strictly_ordered(blur_medians, ["low", "medium", "high"])
    face_signal_reliable = float(len(zero_face_df) / len(df)) <= 0.10 if len(df) else False
    face_size_ready = (
        face_signal_reliable
        and is_strictly_ordered(face_size_medians, ["small", "medium", "large"])
        and gap_ratio(face_size_medians, "small", "medium") >= 1.25
        and gap_ratio(face_size_medians, "medium", "large") >= 1.5
    )
    occlusion_ready = (
        face_signal_reliable
        and is_strictly_ordered(occlusion_medians, ["low", "medium", "high"])
        and gap_delta(occlusion_medians, "low", "medium") >= 0.03
        and gap_delta(occlusion_medians, "medium", "high") >= 0.03
    )
    webp_ready = (
        is_strictly_ordered(webp_medians, ["low", "medium", "high"])
        and gap_delta(webp_medians, "low", "medium") >= 0.02
        and gap_delta(webp_medians, "medium", "high") >= 0.02
    )

    notes: list[str] = []
    notes.append(
        "Blur is suitable for threshold sensitivity analysis."
        if blur_ready
        else "Blur is not yet directionally consistent enough for threshold sensitivity analysis."
    )
    notes.append(
        "Face-size is threshold-ready on face-present rows with the current detector source."
        if face_size_ready
        else "Face-size remains operationally unreliable because detector misses are too high or small/medium/large separation is still too weak."
    )
    notes.append(
        "Occlusion is threshold-ready."
        if occlusion_ready
        else "Occlusion medians remain too flat, too weakly separated, or too detector-dependent across labels."
    )
    notes.append(
        "WebP artifact severity is threshold-ready."
        if webp_ready
        else "WebP artifact medians remain too flat or non-monotonic across labels."
    )

    summary = {
        "input_csv": args.input_csv,
        "rows": int(len(df)),
        "face_present_rows": int(len(face_present_df)),
        "zero_face_rows": int(len(zero_face_df)),
        "zero_face_rate": float(len(zero_face_df) / len(df)) if len(df) else 0.0,
        "all_rows": {
            "blur": all_blur,
            "face_size": all_face_size,
            "occlusion": all_occlusion,
            "webp": all_webp,
        },
        "face_present_only": {
            "face_size": face_present_face_size,
            "occlusion": face_present_occlusion,
        },
        "zero_face_by_face_size_label": {
            str(label): int(count)
            for label, count in zero_face_df["face_size_label"].value_counts().sort_index().items()
        },
        "zero_face_by_occlusion_label": {
            str(label): int(count)
            for label, count in zero_face_df["occlusion_ratio_label"].value_counts().sort_index().items()
        },
        "diagnostic_conclusion": {
            "blur_signal_directional": blur_ready,
            "face_size_signal_ready_for_threshold_sweep": face_size_ready,
            "occlusion_signal_ready_for_threshold_sweep": occlusion_ready,
            "webp_signal_ready_for_threshold_sweep": webp_ready,
            "notes": notes,
        },
    }

    output_path = PROJECT_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
