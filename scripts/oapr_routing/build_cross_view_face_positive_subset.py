#!/usr/bin/env python3

"""Build a face-positive cross-view evaluation subset using the active detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.oapr_routing.build_cross_view_eval_subset import build_summary, sample_cross_view_pairs
from src.detection.yolo_scrfd_fallback_detector import YOLOSCRFDFallbackDetector


RAW_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"
YOLO_MODEL = PROJECT_ROOT / "data" / "models" / "yolov8n.pt"


def load_image(relative_path: str) -> Image.Image:
    with Image.open(RAW_ROOT / relative_path) as image:
        return image.convert("RGB")


def face_counts_for_pair(detector: YOLOSCRFDFallbackDetector, row: pd.Series) -> tuple[int, int]:
    """Return detected face counts for the ego and exo images in one pair row."""
    ego_image = load_image(str(row["egocentric_relative_path"]))
    exo_image = load_image(str(row["exocentric_relative_path"]))
    ego_count = len(detector.detect(ego_image).detections)
    exo_count = len(detector.detect(exo_image).detections)
    return ego_count, exo_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/castle2024/raw_dataset_index.csv",
    )
    parser.add_argument("--target-pairs", type=int, default=120)
    parser.add_argument("--candidate-pairs", type=int, default=400)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        default="outputs/01_protocol/supporting_protocols/05_cross_view_face_positive_pairs_80.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/runs/cross_view/cross_view_eval_face_positive_subset_summary.json",
    )
    args = parser.parse_args()

    manifest_df = pd.read_csv(PROJECT_ROOT / args.manifest)
    candidate_df = sample_cross_view_pairs(
        manifest_df=manifest_df,
        target_pairs=args.candidate_pairs,
        random_seed=args.random_seed,
    )

    detector = YOLOSCRFDFallbackDetector(
        yolo_model_path=str(YOLO_MODEL),
        yolo_device=args.device,
    )

    kept_rows: list[pd.Series] = []
    for _, row in candidate_df.iterrows():
        ego_count, exo_count = face_counts_for_pair(detector, row)
        row = row.copy()
        row["ego_face_count_seed"] = ego_count
        row["exo_face_count_seed"] = exo_count
        if ego_count > 0 and exo_count > 0:
            kept_rows.append(row)
        if len(kept_rows) >= args.target_pairs:
            break

    if not kept_rows:
        raise SystemExit("No face-positive matched pairs were found in the sampled candidate pool.")

    subset_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    summary = build_summary(subset_df, args.target_pairs, args.random_seed)
    summary.update(
        {
            "candidate_pairs_scanned": int(len(candidate_df)),
            "face_positive_pairs_found": int(len(subset_df)),
            "mean_seed_ego_face_count": float(subset_df["ego_face_count_seed"].mean()),
            "mean_seed_exo_face_count": float(subset_df["exo_face_count_seed"].mean()),
        }
    )

    output_path = PROJECT_ROOT / args.output
    summary_path = PROJECT_ROOT / args.summary_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output_path), "summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
