#!/usr/bin/env python3

"""Run a text detector over a reviewed pack and save per-image counts."""

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

from src.detection.text_detector import TextDetector


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--backend", choices=["easyocr", "east", "doctr"], default="easyocr")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--east-model", default="data/models/frozen_east_text_detection.pb")
    parser.add_argument(
        "--easyocr-scales",
        default="1.0",
        help="Comma-separated EasyOCR scales, for example '1.0,1.5'.",
    )
    args = parser.parse_args()

    review_csv_path = PROJECT_ROOT / args.review_csv
    rows = load_rows(review_csv_path)
    pack_root = review_csv_path.parent
    easyocr_scales = [float(value.strip()) for value in args.easyocr_scales.split(",") if value.strip()]
    detector = TextDetector(
        backend=args.backend,
        device=args.device,
        east_model_path=str(PROJECT_ROOT / args.east_model),
        easyocr_multiscale_scales=easyocr_scales,
    )

    per_image: list[dict[str, object]] = []
    total_regions = 0
    detector_name = "unknown"
    for row in rows:
        image_path = pack_root / "images" / row["relative_path"]
        if not image_path.is_file():
            image_path = PROJECT_ROOT / "data" / "castle2024" / "raw" / row["relative_path"]
        with Image.open(image_path) as image:
            loaded = image.copy()
        result = detector.detect(loaded)
        detector_name = result.metadata.get("detector_name", detector_name)
        region_count = len(result.detections)
        total_regions += region_count
        per_image.append(
            {
                "relative_path": row["relative_path"],
                "text_region_count": region_count,
                "screen_region_count": 0,
            }
        )

    payload = {
        "version": "1.0",
        "review_csv": args.review_csv,
        "images_processed": len(rows),
        "text_detector": detector_name,
        "text_detector_backend": args.backend,
        "easyocr_multiscale_scales": easyocr_scales,
        "text_region_count_total": total_regions,
        "screen_region_count_total": 0,
        "per_image": per_image,
    }

    output_json_path = PROJECT_ROOT / args.output_json
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
