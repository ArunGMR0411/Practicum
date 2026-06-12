#!/usr/bin/env python3
"""Export per-detector candidate boxes for condition-profile telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detection.run_face_detector_hardening_experiment import (  # noqa: E402
    Candidate,
    Protocol,
    load_records,
    nms_fusion,
    run_candidate,
    write_csv,
)
from scripts.detection.run_low_compute_detector_policy_experiment import (  # noqa: E402
    SCENE_PREDICTIONS,
    cluster_predictions,
    cross_validated_reranker,
    load_scene_predictions,
)
from scripts.detection.run_sliced_and_rfdetr_detector_experiment import (  # noqa: E402
    run_rfdetr,
    run_sliced_yolo,
)
from src.evaluation.detection_metrics import ScoredBox  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/02_face_detection/11_detector_candidate_box_telemetry"
RFDETR_CHECKPOINT = (
    PROJECT_ROOT
    / "data/models/face_detection_candidates/rfdetr_hf_cache/models--Herojayjay--RFDETR-Face-Detection/"
    / "snapshots/597fcce941997900080ce8127b53a5d24e330225/rfdetr_medium_face.pth"
)


def write_union_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(model: str, predictions: list[ScoredBox]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        protocol, relative_path = prediction.image_id.split("::", 1)
        x1, y1, x2, y2 = prediction.box
        rows.append(
            {
                "protocol": protocol,
                "relative_path": relative_path,
                "scoped_image_id": prediction.image_id,
                "detector_name": model,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "score": round(float(prediction.score), 8),
                "source_detector": str(prediction.metadata.get("source_detector", model)) if prediction.metadata else model,
            }
        )
    return rows


def build_records(max_images: int | None) -> list[Any]:
    protocols = [
        Protocol("01_baseline_500", PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/01_baseline_500/manifest.csv"),
        Protocol("02_egocentric_stress_500", PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"),
    ]
    records = load_records(protocols)
    if max_images is not None:
        return records[:max_images]
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-sliced", action="store_true")
    parser.add_argument("--skip-rfdetr", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(args.max_images)
    records_by_id = {record.scoped_id: record for record in records}
    started = perf_counter()

    candidates = [
        Candidate("yolo11s_widerface_1280", "yolo", "detector", "data/models/face_detection_candidates/yolo11s_widerface.pt", 1280),
        Candidate("scrfd_10g_current_640", "scrfd", "detector", "/home/arun-gmr/.insightface/models/buffalo_l/det_10g.onnx", 640),
        Candidate("yolo8s_widerface_repo_640", "yolo", "detector", "data/models/face_detection_candidates/yolov8s_widerface.pt", 640),
        Candidate("yolo11n_pose_widerface_640", "yolo", "detector", "data/models/face_detection_candidates/yolo11n-pose_widerface.pt", 640),
        Candidate("yolo11s_widerface_640", "yolo", "detector", "data/models/face_detection_candidates/yolo11s_widerface.pt", 640),
    ]

    predictions_by_model: dict[str, list[ScoredBox]] = {}
    runtime_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        print(f"\n=== Running {candidate.name} ===", flush=True)
        predictions, runtime = run_candidate(candidate, records)
        predictions_by_model[candidate.name] = predictions
        runtime_rows.append(runtime)
        candidate_rows.extend(prediction_rows(candidate.name, predictions))

    if not args.skip_sliced:
        print("\n=== Running sliced_yolo11s_widerface_1280 ===", flush=True)
        sliced, sliced_runtime = run_sliced_yolo(
            records=records,
            model_path="data/models/face_detection_candidates/yolo11s_widerface.pt",
            model_name="sliced_yolo11s_widerface_1280",
        )
        predictions_by_model["sliced_yolo11s_widerface_1280"] = sliced
        runtime_rows.append(sliced_runtime)
        candidate_rows.extend(prediction_rows("sliced_yolo11s_widerface_1280", sliced))

    if not args.skip_rfdetr:
        print("\n=== Running rfdetr_medium_face_030 ===", flush=True)
        try:
            rfdetr, rfdetr_runtime = run_rfdetr(
                records=records,
                checkpoint_path=str(RFDETR_CHECKPOINT),
                model_name="rfdetr_medium_face_030",
            )
            predictions_by_model["rfdetr_medium_face_030"] = rfdetr
            runtime_rows.append(rfdetr_runtime)
            candidate_rows.extend(prediction_rows("rfdetr_medium_face_030", rfdetr))
        except Exception as exc:
            runtime_rows.append(
                {
                    "candidate": "rfdetr_medium_face_030",
                    "family": "rfdetr",
                    "model_path": str(RFDETR_CHECKPOINT),
                    "runtime_seconds": "",
                    "images": len(records),
                    "detections": 0,
                    "failures": len(records),
                    "failure_examples": f"{type(exc).__name__}: {exc}",
                }
            )

    fusion_specs = [
        ("fusion_yolo11s1280_scrfd10g", ["yolo11s_widerface_1280", "scrfd_10g_current_640"], 0.50, 0.22, 0.08, 0.90),
        ("fusion_yolo11s_scrfd10g", ["yolo11s_widerface_640", "scrfd_10g_current_640"], 0.50, 0.22, 0.08, 0.90),
        ("fusion_rfdetr_scrfd10g", ["rfdetr_medium_face_030", "scrfd_10g_current_640"], 0.50, 0.20, 0.08, 0.90),
    ]
    for name, sources, iou, min_score, bonus, penalty in fusion_specs:
        if not all(source in predictions_by_model for source in sources):
            continue
        predictions = nms_fusion(predictions_by_model, sources, name, iou, min_score, bonus, penalty)
        predictions_by_model[name] = predictions
        runtime_rows.append(
            {
                "candidate": name,
                "family": "fusion",
                "model_path": "+".join(sources),
                "confidence": min_score,
                "runtime_seconds": "",
                "images": len(records),
                "detections": len(predictions),
                "failures": 0,
                "failure_examples": "",
            }
        )
        candidate_rows.extend(prediction_rows(name, predictions))

    reranker_sources = [
        source
        for source in [
            "rfdetr_medium_face_030",
            "yolo11s_widerface_1280",
            "scrfd_10g_current_640",
            "yolo8s_widerface_repo_640",
            "yolo11n_pose_widerface_640",
        ]
        if source in predictions_by_model
    ]
    if len(reranker_sources) >= 2:
        scene_predictions = load_scene_predictions(SCENE_PREDICTIONS)
        clusters = cluster_predictions(predictions_by_model, reranker_sources, records_by_id)
        reranked, thresholds = cross_validated_reranker(
            clusters=clusters,
            records=records,
            source_names=reranker_sources,
            scene_predictions=scene_predictions,
            use_oracle_conditions=False,
        )
        predictions_by_model["cv_box_reranker_with_rfdetr_predicted_conditions"] = reranked
        candidate_rows.extend(prediction_rows("cv_box_reranker_with_rfdetr_predicted_conditions", reranked))
        for threshold in thresholds:
            threshold["reranker"] = "cv_box_reranker_with_rfdetr_predicted_conditions"
        write_csv(output_dir / "detector_candidate_reranker_thresholds.csv", thresholds)

    write_union_csv(output_dir / "detector_candidate_boxes.csv", candidate_rows)
    write_union_csv(output_dir / "detector_candidate_runtime.csv", runtime_rows)
    summary = [
        {
            "image_count": len(records),
            "detector_or_policy_count": len(predictions_by_model),
            "box_count": len(candidate_rows),
            "runtime_total_seconds": round(perf_counter() - started, 3),
            "detectors": "|".join(sorted(predictions_by_model)),
        }
    ]
    write_csv(output_dir / "detector_candidate_box_export_summary.csv", summary)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Detector Candidate Box Telemetry",
                "",
                "This folder retains per-image candidate boxes from the detector families used by the face-detection policy evidence.",
                "",
                "Purpose:",
                "",
                "- Preserve true per-detector candidate boxes for detector-disagreement telemetry.",
                "- Support the Step 4 condition-profile model before anonymisation routing.",
                "- Avoid losing evidence by keeping only aggregate detector scores.",
                "",
                "Main files:",
                "",
                "- `detector_candidate_boxes.csv`: per-detector and policy boxes.",
                "- `detector_candidate_runtime.csv`: runtime/failure evidence.",
                "- `detector_candidate_box_export_summary.csv`: export summary.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote detector candidate boxes to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
