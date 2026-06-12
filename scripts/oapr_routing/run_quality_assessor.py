#!/usr/bin/env python3

"""Run the routing quality assessor on a manifest-backed image set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.routing.quality_assessor import QualityAssessor


def save_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_detection_boxes(path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    import pandas as pd

    df = pd.read_csv(path)
    required = {"image_id", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Detections CSV is missing required columns: {sorted(missing)}")

    grouped: dict[str, list[tuple[int, int, int, int]]] = {}
    for row in df.itertuples(index=False):
        grouped.setdefault(str(row.image_id), []).append(
            (int(row.x1), int(row.y1), int(row.x2), int(row.y2))
        )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/01_protocol/supporting_protocols/02_calibration_200.csv")
    parser.add_argument("--output-csv", default="outputs/calibration_quality_signals.csv")
    parser.add_argument("--output-json", default="outputs/calibration_quality_signals_summary.json")
    parser.add_argument(
        "--detections-csv",
        default="",
        help="Optional face detections CSV from run_detector_inference.py",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0 means full manifest")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    output_csv_path = PROJECT_ROOT / args.output_csv
    output_json_path = PROJECT_ROOT / args.output_json

    import pandas as pd

    manifest_df = pd.read_csv(manifest_path)
    assessor = QualityAssessor()
    detections_lookup = (
        load_detection_boxes(PROJECT_ROOT / args.detections_csv)
        if args.detections_csv
        else {}
    )

    rows: list[dict[str, object]] = []
    for idx, row in manifest_df.iterrows():
        if args.max_images and idx >= args.max_images:
            break
        image_path = PROJECT_ROOT / "data" / "castle2024" / "raw" / str(row["relative_path"])
        face_boxes = detections_lookup.get(str(row["relative_path"]), [])
        with Image.open(image_path) as image:
            assessment = assessor.assess(image=image, face_boxes=face_boxes)

        rows.append(
            {
                "relative_path": str(row["relative_path"]),
                "day_id": str(row.get("day_id", "")),
                "camera_stream_id": str(row.get("camera_stream_id", "")),
                "view_type": str(row.get("view_type", "")),
                "blur_level_label": str(row.get("blur_level", "")),
                "face_size_label": str(row.get("face_size", "")),
                "occlusion_ratio_label": str(row.get("occlusion_ratio", "")),
                "webp_artefact_severity_label": str(row.get("webp_artefact_severity", "")),
                **assessment.to_dict(),
            }
        )

    save_csv(rows, output_csv_path)

    zero_face_rows = sum(1 for row in rows if int(row.get("face_box_count", 0)) == 0)
    signal_by_label: dict[str, dict[str, float]] = {}
    for label_key, signal_key in (
        ("blur_level_label", "blur_score"),
        ("face_size_label", "face_size_px"),
        ("occlusion_ratio_label", "occlusion_ratio"),
        ("webp_artefact_severity_label", "webp_artifact_score"),
    ):
        grouped: dict[str, list[float]] = {}
        for row in rows:
            label_value = str(row.get(label_key, ""))
            signal_value = float(row.get(signal_key, 0.0))
            grouped.setdefault(label_value, []).append(signal_value)
        signal_by_label[f"{label_key}_median_{signal_key}"] = {
            key: float(pd.Series(values).median()) for key, values in sorted(grouped.items())
        }

    summary = {
        "version": "1.0",
        "manifest": args.manifest,
        "detections_csv": args.detections_csv,
        "rows_written": len(rows),
        "zero_face_rows": zero_face_rows,
        "output_csv": str(output_csv_path.relative_to(PROJECT_ROOT)),
        "label_signal_medians": signal_by_label,
        "sample": rows[:3],
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
