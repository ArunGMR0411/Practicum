#!/usr/bin/env python3

"""Run a proxy cross-view residual-linkability evaluation on matched CASTLE pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.blur_anonymiser import BlurAnonymiser
from src.anonymisation.pixelate_anonymiser import PixelateAnonymiser
from src.detection.yolo_scrfd_fallback_detector import YOLOSCRFDFallbackDetector
from src.evaluation.cross_view import build_control_groups, max_similarity_to_gallery
from src.evaluation.reid_evaluator import ReIDEvaluator


RAW_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"
YOLO_MODEL = PROJECT_ROOT / "data" / "models" / "yolov8n.pt"
ADAFACE_CKPT = PROJECT_ROOT / "data/models/adaface_ir50_ms1mv2.ckpt"
ARCFACE_ONNX = Path("~/.insightface/models/buffalo_l/w600k_r50.onnx").expanduser()


def load_image(relative_path: str) -> Image.Image:
    with Image.open(RAW_ROOT / relative_path) as image:
        return image.convert("RGB")


def boxes_from_result(result) -> list[tuple[int, int, int, int]]:
    return [tuple(map(int, detection.box)) for detection in result.detections]


def crop_faces(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> list[Image.Image]:
    crops: list[Image.Image] = []
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append(image.crop((x1, y1, x2, y2)))
    return crops


def evaluate_pair_rows(
    pair_df: pd.DataFrame,
    detector: YOLOSCRFDFallbackDetector,
    evaluator: ReIDEvaluator,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate matched and mismatched cross-view similarity on one sampled pair manifest."""
    blur = BlurAnonymiser()
    pixelate = PixelateAnonymiser()
    pair_rows: list[dict[str, object]] = []

    for record in pair_df.to_dict(orient="records"):
        ego_image = load_image(str(record["egocentric_relative_path"]))
        exo_image = load_image(str(record["exocentric_relative_path"]))

        ego_boxes = boxes_from_result(detector.detect(ego_image))
        exo_boxes = boxes_from_result(detector.detect(exo_image))
        if not ego_boxes or not exo_boxes:
            pair_rows.append(
                {
                    **record,
                    "ego_face_count": len(ego_boxes),
                    "exo_face_count": len(exo_boxes),
                    "evaluated": False,
                    "skip_reason": "missing_detected_faces",
                }
            )
            continue

        ego_crops = crop_faces(ego_image, ego_boxes)
        exo_crops = crop_faces(exo_image, exo_boxes)
        if not ego_crops or not exo_crops:
            pair_rows.append(
                {
                    **record,
                    "ego_face_count": len(ego_boxes),
                    "exo_face_count": len(exo_boxes),
                    "evaluated": False,
                    "skip_reason": "empty_crops_after_validation",
                }
            )
            continue

        blurred_ego = blur.anonymise(ego_image, ego_boxes).image
        pixelated_ego = pixelate.anonymise(ego_image, ego_boxes).image
        blur_crops = crop_faces(blurred_ego, ego_boxes)
        pixelate_crops = crop_faces(pixelated_ego, ego_boxes)

        ego_embeddings = evaluator.extract_embeddings_adaface(ego_crops)
        blur_embeddings = evaluator.extract_embeddings_adaface(blur_crops)
        pixelate_embeddings = evaluator.extract_embeddings_adaface(pixelate_crops)
        exo_embeddings = evaluator.extract_embeddings_adaface(exo_crops)

        pair_rows.append(
            {
                **record,
                "ego_face_count": len(ego_crops),
                "exo_face_count": len(exo_crops),
                "evaluated": True,
                "skip_reason": "",
                "original_max_cross_view_cosine": max(
                    max_similarity_to_gallery(query_embedding, exo_embeddings) for query_embedding in ego_embeddings
                ),
                "blur_max_cross_view_cosine": max(
                    max_similarity_to_gallery(query_embedding, exo_embeddings) for query_embedding in blur_embeddings
                ),
                "pixelate_max_cross_view_cosine": max(
                    max_similarity_to_gallery(query_embedding, exo_embeddings) for query_embedding in pixelate_embeddings
                ),
            }
        )

    control_groups = build_control_groups(pair_rows)
    evaluated_rows = [row for row in pair_rows if row.get("evaluated")]
    for row in evaluated_rows:
        controls = [
            idx
            for idx in control_groups[(str(row["day_id"]), str(row["exocentric_stream_id"]))]
            if pair_rows[idx]["timestamp_id"] != row["timestamp_id"] and pair_rows[idx].get("evaluated")
        ]
        if not controls:
            row["control_original_max_cross_view_cosine"] = 0.0
            row["control_blur_max_cross_view_cosine"] = 0.0
            row["control_pixelate_max_cross_view_cosine"] = 0.0
            continue
        control_row = pair_rows[controls[0]]
        row["control_original_max_cross_view_cosine"] = float(control_row["original_max_cross_view_cosine"])
        row["control_blur_max_cross_view_cosine"] = float(control_row["blur_max_cross_view_cosine"])
        row["control_pixelate_max_cross_view_cosine"] = float(control_row["pixelate_max_cross_view_cosine"])
        row["original_gap_vs_control"] = float(
            row["original_max_cross_view_cosine"] - row["control_original_max_cross_view_cosine"]
        )
        row["blur_gap_vs_control"] = float(
            row["blur_max_cross_view_cosine"] - row["control_blur_max_cross_view_cosine"]
        )
        row["pixelate_gap_vs_control"] = float(
            row["pixelate_max_cross_view_cosine"] - row["control_pixelate_max_cross_view_cosine"]
        )

    original_scores = [row["original_max_cross_view_cosine"] for row in evaluated_rows]
    blur_scores = [row["blur_max_cross_view_cosine"] for row in evaluated_rows]
    pixelate_scores = [row["pixelate_max_cross_view_cosine"] for row in evaluated_rows]
    summary = {
        "pairs_total": int(len(pair_rows)),
        "pairs_evaluated": int(len(evaluated_rows)),
        "pairs_skipped": int(len(pair_rows) - len(evaluated_rows)),
        "mean_original_max_cross_view_cosine": float(np.mean(original_scores)) if original_scores else 0.0,
        "mean_blur_max_cross_view_cosine": float(np.mean(blur_scores)) if blur_scores else 0.0,
        "mean_pixelate_max_cross_view_cosine": float(np.mean(pixelate_scores)) if pixelate_scores else 0.0,
        "mean_blur_delta_vs_original": float(np.mean(np.array(blur_scores) - np.array(original_scores)))
        if original_scores
        else 0.0,
        "mean_pixelate_delta_vs_original": float(np.mean(np.array(pixelate_scores) - np.array(original_scores)))
        if original_scores
        else 0.0,
    }
    return pair_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="outputs/01_protocol/supporting_protocols/04_cross_view_pairs_200.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/05_oapr/cross_view_analysis/01_cross_view_proxy_pair_results.csv",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/runs/cross_view/cross_view_proxy_results.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    pair_df = pd.read_csv(PROJECT_ROOT / args.manifest)
    if args.max_pairs is not None:
        pair_df = pair_df.head(args.max_pairs).copy()

    detector = YOLOSCRFDFallbackDetector(
        yolo_model_path=str(YOLO_MODEL),
        yolo_device=args.device,
    )
    evaluator = ReIDEvaluator(
        adaface_ckpt_path=str(ADAFACE_CKPT),
        arcface_onnx_path=str(ARCFACE_ONNX),
        device=args.device,
    )

    pair_rows, summary = evaluate_pair_rows(pair_df, detector, evaluator)
    output_csv = PROJECT_ROOT / args.output_csv
    output_json = PROJECT_ROOT / args.output_json
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(pair_rows).to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(output_csv), "json": str(output_json), **summary}, indent=2))


if __name__ == "__main__":
    main()
