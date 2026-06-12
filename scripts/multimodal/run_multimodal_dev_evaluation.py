#!/usr/bin/env python3

"""Run text/screen detection, redaction, and OCR suppression evaluation on the dev set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.screen_redactor import ScreenRedactor
from src.anonymisation.text_redactor import TextRedactor
from src.data.castle_loader import CASTLEDataset
from src.detection.screen_detector import ScreenDetector
from src.detection.text_detector import TextDetector
from src.evaluation.ocr_evaluator import OCREvaluator
from src.utils.compute_policy import build_compute_policy


def save_csv(rows: list[dict[str, object]], fieldnames: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output_path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    if len(values) == 0:
        return (0.0, 0.0)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    se = std / np.sqrt(len(values))
    from scipy.stats import t as t_dist

    alpha = 1.0 - confidence
    t_crit = float(t_dist.ppf(1 - alpha / 2, df=len(values) - 1))
    return (mean - t_crit * se, mean + t_crit * se)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/01_protocol/supporting_protocols/01_development_300.csv")
    parser.add_argument("--output-root", default="outputs/multimodal_dev")
    parser.add_argument("--output-manifest", default="outputs/multimodal_dev_manifest.csv")
    parser.add_argument("--output-json", default="outputs/multimodal_dev_results.json")
    parser.add_argument("--text-backend", choices=["easyocr", "east", "doctr"], default="easyocr")
    parser.add_argument("--text-device", default="cuda")
    parser.add_argument("--screen-device", default="0")
    parser.add_argument("--ocr-device", default="cuda")
    parser.add_argument("--screen-model", default="data/models/yolov8n.pt")
    parser.add_argument("--east-model", default="data/models/frozen_east_text_detection.pb")
    parser.add_argument("--text-redaction-mode", choices=["blur", "fill"], default="fill")
    parser.add_argument("--screen-redaction-mode", choices=["blur", "fill"], default="blur")
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=0, help="0 means full manifest")
    args = parser.parse_args()
    policy = build_compute_policy()

    manifest_path = PROJECT_ROOT / args.manifest
    dataset = CASTLEDataset(str(manifest_path), return_format="pil", filters={})

    text_detector = TextDetector(
        backend=args.text_backend,
        device=args.text_device,
        east_model_path=str(PROJECT_ROOT / args.east_model),
    )
    screen_detector = ScreenDetector(model_path=args.screen_model, device=args.screen_device)
    text_redactor = TextRedactor(mode=args.text_redaction_mode)
    screen_redactor = ScreenRedactor(mode=args.screen_redaction_mode)
    ocr_evaluator = OCREvaluator(device=args.ocr_device, region_batch_size=policy.ocr_region_batch_size)

    output_root = PROJECT_ROOT / args.output_root
    output_manifest_path = PROJECT_ROOT / args.output_manifest
    output_json_path = PROJECT_ROOT / args.output_json

    image_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    redaction_manifest_rows: list[dict[str, object]] = []
    post_similarities: list[float] = []
    pre_similarities: list[float] = []
    suppressed_count = 0

    processed = 0
    for item in dataset:
        if args.max_images and processed >= args.max_images:
            break
        processed += 1
        image: Image.Image = item["image"]
        relative_path = item["metadata"]["relative_path"]

        text_result = text_detector.detect(image)
        text_boxes = [d.box for d in text_result.detections]
        screen_result = screen_detector.detect(image)
        screen_boxes = [d.box for d in screen_result.detections]

        text_redacted = text_redactor.anonymise(image, text_boxes)
        fully_redacted = screen_redactor.anonymise(text_redacted.image, screen_boxes)

        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fully_redacted.image.save(output_path)

        original_ocr = ocr_evaluator.recognise_regions(image, text_boxes)
        redacted_ocr = ocr_evaluator.recognise_regions(fully_redacted.image, text_boxes)

        image_suppression = ocr_evaluator.suppression_rate(
            original_ocr, redacted_ocr, similarity_threshold=args.similarity_threshold
        )
        image_similarity_summary = ocr_evaluator.paired_accuracy_summary(
            original_ocr, redacted_ocr
        )

        image_rows.append(
            {
                "relative_path": relative_path,
                "text_region_count": len(text_boxes),
                "screen_region_count": len(screen_boxes),
                "ocr_suppression_rate": image_suppression,
                "ocr_mean_similarity": image_similarity_summary["mean_similarity"],
                "output_path": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )
        redaction_manifest_rows.append(
            {
                "relative_path": relative_path,
                "output_path": str(output_path.relative_to(PROJECT_ROOT)),
                "text_region_count": len(text_boxes),
                "screen_region_count": len(screen_boxes),
                "text_redaction_mode": args.text_redaction_mode,
                "screen_redaction_mode": args.screen_redaction_mode,
            }
        )

        for idx, (before, after) in enumerate(zip(original_ocr, redacted_ocr, strict=True)):
            similarity = ocr_evaluator.text_similarity(before.text, after.text)
            pre_similarities.append(1.0)
            post_similarities.append(similarity)
            if similarity < args.similarity_threshold:
                suppressed_count += 1
            region_rows.append(
                {
                    "relative_path": relative_path,
                    "region_index": idx,
                    "box": list(before.box),
                    "original_text": before.text,
                    "redacted_text": after.text,
                    "similarity": similarity,
                    "suppressed": similarity < args.similarity_threshold,
                }
            )

    save_csv(
        redaction_manifest_rows,
        fieldnames=[
            "relative_path",
            "output_path",
            "text_region_count",
            "screen_region_count",
            "text_redaction_mode",
            "screen_redaction_mode",
        ],
        output_path=output_manifest_path,
    )

    pre = np.array(pre_similarities, dtype=np.float32)
    post = np.array(post_similarities, dtype=np.float32)
    if len(post) == 0:
        stats = {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "mean_pre_similarity": 0.0,
            "mean_post_similarity": 0.0,
            "n_regions": 0,
            "ci_lower_post_similarity": 0.0,
            "ci_upper_post_similarity": 0.0,
        }
    else:
        test = ttest_rel(pre, post)
        ci_lower, ci_upper = confidence_interval(post)
        stats = {
            "t_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "mean_pre_similarity": float(pre.mean()),
            "mean_post_similarity": float(post.mean()),
            "n_regions": int(len(post)),
            "ci_lower_post_similarity": float(ci_lower),
            "ci_upper_post_similarity": float(ci_upper),
        }

    payload = {
        "version": "1.0",
        "manifest": args.manifest,
        "images_processed": processed,
        "text_detector": text_result.metadata.get("detector_name", "craft_easyocr")
        if processed
        else "craft_easyocr",
        "text_detector_backend": args.text_backend,
        "screen_detector": screen_result.metadata.get("detector_name", "screen_yolo_fallback")
        if processed
        else "screen_yolo_fallback",
        "text_redaction_mode": args.text_redaction_mode,
        "screen_redaction_mode": args.screen_redaction_mode,
        "compute_policy": {
            "device": policy.device,
            "ocr_region_batch_size": policy.ocr_region_batch_size,
            "accelerator_total_gb": policy.accelerator_total_gb,
        },
        "text_region_count_total": len(region_rows),
        "screen_region_count_total": int(sum(row["screen_region_count"] for row in image_rows)),
        "suppressed_region_fraction": float(suppressed_count / len(region_rows)) if region_rows else 0.0,
        "ocr_similarity_stats": stats,
        "per_image": image_rows,
        "per_region": region_rows,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "images_processed": processed,
        "text_regions": len(region_rows),
        "screen_regions": int(sum(row["screen_region_count"] for row in image_rows)),
        "suppressed_fraction": payload["suppressed_region_fraction"],
        "output_json": str(output_json_path),
        "output_manifest": str(output_manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
