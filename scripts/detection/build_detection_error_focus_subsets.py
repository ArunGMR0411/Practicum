#!/usr/bin/env python3

"""Build narrow reviewed focus subsets from the full detection error audit."""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["relative_path"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-summary",
        default="outputs/detection_error_audit_yolo_scrfd_fallback/image_summary.csv",
    )
    parser.add_argument(
        "--annotation-manifest",
        default="outputs/01_protocol/thesis_manifests/final_face_detection_500.csv",
    )
    parser.add_argument(
        "--zero-face-output",
        default="outputs/intermediate/detection_error_focus/detection_zero_face_fp_cleanup_subset.csv",
    )
    parser.add_argument(
        "--missed-face-output",
        default="outputs/intermediate/detection_error_focus/detection_missed_face_cleanup_subset.csv",
    )
    args = parser.parse_args()

    image_summary = load_rows(PROJECT_ROOT / args.image_summary)
    manifest_rows = load_rows(PROJECT_ROOT / args.annotation_manifest)
    manifest_by_path = {row["relative_path"]: row for row in manifest_rows}

    zero_face_rows: list[dict[str, str]] = []
    missed_face_rows: list[dict[str, str]] = []

    for row in image_summary:
        image_id = row["image_id"]
        manifest_row = dict(manifest_by_path.get(image_id, {"relative_path": image_id}))
        manifest_row["audit_false_positive_count"] = row["false_positive_count"]
        manifest_row["audit_false_negative_count"] = row["false_negative_count"]
        manifest_row["audit_gt_count"] = row["gt_count"]
        manifest_row["audit_prediction_count"] = row["prediction_count"]

        if row["zero_face_image"] == "True" and int(row["false_positive_count"]) > 0:
            zero_face_rows.append(dict(manifest_row))

        false_negative_count = int(row["false_negative_count"])
        if false_negative_count <= 0:
            continue
        if false_negative_count >= 2 or int(row["prediction_count"]) == 0:
            missed_face_rows.append(dict(manifest_row))

    zero_face_rows.sort(key=lambda item: (int(item["audit_false_positive_count"]), item["relative_path"]), reverse=True)
    missed_face_rows.sort(key=lambda item: (int(item["audit_false_negative_count"]), int(item["audit_gt_count"]), item["relative_path"]), reverse=True)

    save_rows(zero_face_rows, PROJECT_ROOT / args.zero_face_output)
    save_rows(missed_face_rows, PROJECT_ROOT / args.missed_face_output)

    print(
        {
            "zero_face_fp_subset_count": len(zero_face_rows),
            "missed_face_cleanup_subset_count": len(missed_face_rows),
            "zero_face_output": args.zero_face_output,
            "missed_face_output": args.missed_face_output,
        }
    )


if __name__ == "__main__":
    main()
