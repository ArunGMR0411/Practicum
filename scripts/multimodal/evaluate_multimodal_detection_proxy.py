#!/usr/bin/env python3

"""Proxy audit of multimodal detector quality against defended dev-set flags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def summarize_flag(df: pd.DataFrame, flag_col: str, count_col: str) -> dict[str, float | int]:
    pos = df[df[flag_col]]
    neg = df[~df[flag_col]]
    return {
        "positive_images": int(len(pos)),
        "negative_images": int(len(neg)),
        "positive_hit_rate": float((pos[count_col] > 0).mean()) if len(pos) else 0.0,
        "negative_trigger_rate": float((neg[count_col] > 0).mean()) if len(neg) else 0.0,
        "median_regions_positive": float(pos[count_col].median()) if len(pos) else 0.0,
        "median_regions_negative": float(neg[count_col].median()) if len(neg) else 0.0,
        "p90_regions_positive": float(pos[count_col].quantile(0.9)) if len(pos) else 0.0,
        "max_regions_positive": int(pos[count_col].max()) if len(pos) else 0,
        "max_regions_negative": int(neg[count_col].max()) if len(neg) else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/01_protocol/supporting_protocols/01_development_300.csv")
    parser.add_argument("--multimodal-results", default="outputs/multimodal_dev_results.json")
    parser.add_argument("--output-json", default="outputs/multimodal_detection_proxy_audit.json")
    args = parser.parse_args()

    manifest = pd.read_csv(PROJECT_ROOT / args.manifest)
    manifest["visible_text_flag"] = to_bool(manifest["visible_text_flag"])
    manifest["visible_screen_flag"] = to_bool(manifest["visible_screen_flag"])

    results = json.loads((PROJECT_ROOT / args.multimodal_results).read_text(encoding="utf-8"))
    per_image = pd.DataFrame(results["per_image"])

    df = manifest[
        ["relative_path", "visible_text_flag", "visible_screen_flag", "condition_label"]
    ].merge(per_image, on="relative_path", how="inner")

    text_summary = summarize_flag(df, "visible_text_flag", "text_region_count")
    screen_summary = summarize_flag(df, "visible_screen_flag", "screen_region_count")

    extreme_text_rows = (
        df.sort_values("text_region_count", ascending=False)
        .head(20)[["relative_path", "condition_label", "text_region_count", "screen_region_count"]]
        .to_dict(orient="records")
    )

    payload = {
        "version": "1.0",
        "text_detector_name": results.get("text_detector"),
        "screen_detector_name": results.get("screen_detector"),
        "text_detection_proxy": text_summary,
        "screen_detection_proxy": screen_summary,
        "text_detector_baseline_acceptable": bool(
            text_summary["positive_hit_rate"] >= 0.8
            and text_summary["negative_trigger_rate"] <= 0.2
        ),
        "screen_detector_baseline_acceptable": bool(
            screen_summary["positive_hit_rate"] >= 0.8
            and screen_summary["negative_trigger_rate"] <= 0.2
        ),
        "extreme_text_region_examples": extreme_text_rows,
    }

    output_path = PROJECT_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
