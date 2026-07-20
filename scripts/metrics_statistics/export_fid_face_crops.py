#!/usr/bin/env python3

"""Export deterministic face crops for the Phase 3 WebP FID baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detection.run_detector_inference import build_detector
from src.data.castle_loader import CASTLEDataset


DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "01_protocol" / "supporting_protocols" / "03_fid_source_50000.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "fid_webp_baseline_crops"
DEFAULT_METADATA = PROJECT_ROOT / "outputs" / "fid_webp_baseline_crops.csv"
DEFAULT_MODEL = PROJECT_ROOT / "data" / "models" / "yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)))
    parser.add_argument("--detector", choices=["mtcnn", "yolo", "retinaface", "yolo_scrfd_fallback"], default="yolo_scrfd_fallback")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL.relative_to(PROJECT_ROOT)))
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--min-face-size-threshold-px", type=float, default=96.0)
    parser.add_argument("--center-y-threshold", type=float, default=0.65)
    parser.add_argument("--text-score-threshold", type=int, default=12)
    parser.add_argument("--target-crops", type=int, default=10000)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output-metadata", default=str(DEFAULT_METADATA.relative_to(PROJECT_ROOT)))
    return parser.parse_args()


def validate_box(box: tuple[int, int, int, int], image: Image.Image) -> tuple[int, int, int, int] | None:
    """Clamp one box to the image and reject degenerate crops."""
    x1, y1, x2, y2 = box
    left = max(0, min(int(x1), image.width))
    top = max(0, min(int(y1), image.height))
    right = max(0, min(int(x2), image.width))
    bottom = max(0, min(int(y2), image.height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def save_metadata(rows: list[dict[str, object]], output_path: Path) -> None:
    """Atomically persist crop metadata as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "crop_index",
        "split",
        "crop_path",
        "source_relative_path",
        "box_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
        "detector_name",
    ]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    args = parse_args()
    if args.target_crops < 2 or args.target_crops % 2 != 0:
        raise ValueError("--target-crops must be an even integer >= 2")

    dataset = CASTLEDataset(PROJECT_ROOT / args.manifest, return_format="pil", filters={})
    detector = build_detector(args)
    output_root = PROJECT_ROOT / args.output_root
    split_size = args.target_crops // 2
    rows: list[dict[str, object]] = []
    crops_written = 0
    images_processed = 0

    for item in dataset:
        if args.max_images is not None and images_processed >= args.max_images:
            break
        if crops_written >= args.target_crops:
            break

        image = item["image"]
        relative_path = item["metadata"]["relative_path"]
        result = detector.detect(image)
        images_processed += 1

        for box_index, detection in enumerate(result.detections):
            if crops_written >= args.target_crops:
                break
            valid_box = validate_box(detection.box, image)
            if valid_box is None:
                continue
            left, top, right, bottom = valid_box
            crop = image.crop((left, top, right, bottom)).convert("RGB")
            split = "reference" if crops_written < split_size else "comparison"
            crop_path = output_root / split / f"{crops_written:05d}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path)
            rows.append(
                {
                    "crop_index": crops_written,
                    "split": split,
                    "crop_path": str(crop_path.relative_to(PROJECT_ROOT)),
                    "source_relative_path": relative_path,
                    "box_index": box_index,
                    "x1": left,
                    "y1": top,
                    "x2": right,
                    "y2": bottom,
                    "score": float(detection.confidence),
                    "detector_name": detector.detector_name,
                }
            )
            crops_written += 1

    metadata_path = PROJECT_ROOT / args.output_metadata
    save_metadata(rows, metadata_path)
    print(
        json.dumps(
            {
                "images_processed": images_processed,
                "crops_written": crops_written,
                "target_crops": args.target_crops,
                "reference_crops": min(crops_written, split_size),
                "comparison_crops": max(0, crops_written - split_size),
                "output_root": str(output_root.relative_to(PROJECT_ROOT)),
                "output_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
                "detector": detector.detector_name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
