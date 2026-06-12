#!/usr/bin/env python3

"""Low-compute sliced-inference and optional RF-DETR face-detector experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detection.run_face_detector_hardening_experiment import (  # noqa: E402
    Candidate,
    Protocol,
    build_subgroup_membership,
    load_records,
    nms_fusion,
    run_candidate,
    score_predictions,
    write_csv,
)
from scripts.detection.run_low_compute_detector_policy_experiment import (  # noqa: E402
    cluster_predictions,
    cross_validated_reranker,
    load_scene_predictions,
    score_model,
    SCENE_PREDICTIONS,
)
from src.evaluation.detection_metrics import ScoredBox, compute_iou  # noqa: E402


def tile_offsets(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    offsets = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if offsets[-1] != final:
        offsets.append(final)
    return sorted(set(offsets))


def nms_boxes(boxes: list[ScoredBox], iou_threshold: float) -> list[ScoredBox]:
    pending = sorted(boxes, key=lambda item: item.score, reverse=True)
    kept: list[ScoredBox] = []
    while pending:
        seed = pending.pop(0)
        kept.append(seed)
        pending = [item for item in pending if compute_iou(seed.box, item.box) < iou_threshold]
    return kept


def run_sliced_yolo(
    records,
    model_path: str,
    model_name: str,
    tile_size: int = 1280,
    overlap: int = 320,
    confidence: float = 0.20,
    nms_iou: float = 0.50,
) -> tuple[list[ScoredBox], dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    predictions: list[ScoredBox] = []
    started = perf_counter()
    tile_count = 0
    failures: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        try:
            with Image.open(record.path) as loaded:
                image = loaded.convert("RGB")
            width, height = image.size
            image_boxes: list[ScoredBox] = []
            for top in tile_offsets(height, tile_size, overlap):
                for left in tile_offsets(width, tile_size, overlap):
                    right = min(width, left + tile_size)
                    bottom = min(height, top + tile_size)
                    tile = image.crop((left, top, right, bottom))
                    result = model.predict(
                        source=tile,
                        conf=confidence,
                        iou=nms_iou,
                        device="0",
                        imgsz=tile_size,
                        verbose=False,
                    )[0]
                    tile_count += 1
                    if result.boxes is None:
                        continue
                    boxes = result.boxes.xyxy.cpu().tolist()
                    scores = result.boxes.conf.cpu().tolist()
                    classes = result.boxes.cls.cpu().tolist()
                    for box, score, class_id in zip(boxes, scores, classes, strict=False):
                        if int(class_id) != 0:
                            continue
                        x1, y1, x2, y2 = [int(round(v)) for v in box]
                        image_boxes.append(
                            ScoredBox(
                                image_id=record.scoped_id,
                                box=(x1 + left, y1 + top, x2 + left, y2 + top),
                                score=float(score),
                                metadata={"source_detector": model_name},
                            )
                        )
            predictions.extend(nms_boxes(image_boxes, iou_threshold=nms_iou))
        except Exception as exc:  # pragma: no cover - evidence logging path
            failures.append({"image_id": record.scoped_id, "error": f"{type(exc).__name__}: {exc}"})
        if index % 25 == 0:
            elapsed = perf_counter() - started
            rate = index / max(elapsed, 1e-9)
            remaining = (len(records) - index) / max(rate, 1e-9)
            print(
                f"[{model_name}] {index}/{len(records)} images, {tile_count} tiles, "
                f"{rate:.2f} img/s, ETA {remaining/60:.1f} min",
                flush=True,
            )
    runtime = perf_counter() - started
    return predictions, {
        "candidate": model_name,
        "family": "sliced_yolo",
        "model_path": model_path,
        "tile_size": tile_size,
        "overlap": overlap,
        "confidence": confidence,
        "runtime_seconds": round(runtime, 3),
        "images": len(records),
        "tiles": tile_count,
        "detections": len(predictions),
        "failures": len(failures),
        "failure_examples": json.dumps(failures[:5]),
    }


def run_rfdetr(
    records,
    checkpoint_path: str,
    model_name: str,
    threshold: float = 0.30,
) -> tuple[list[ScoredBox], dict[str, Any]]:
    import torch
    from rfdetr import RFDETRMedium

    if not torch.cuda.is_available():
        raise RuntimeError("RF-DETR requires CUDA for this experiment; CPU fallback is disabled.")

    started = perf_counter()
    model = RFDETRMedium(device="cuda", pretrain_weights=checkpoint_path)
    model.optimize_for_inference()
    load_seconds = perf_counter() - started

    predictions: list[ScoredBox] = []
    failures: list[dict[str, str]] = []
    infer_started = perf_counter()
    for index, record in enumerate(records, start=1):
        try:
            with Image.open(record.path) as loaded:
                image = loaded.convert("RGB")
            detections = model.predict(image, threshold=threshold, include_source_image=False)
            xyxy = getattr(detections, "xyxy", [])
            confidences = getattr(detections, "confidence", [])
            class_ids = getattr(detections, "class_id", [])
            for box, score, class_id in zip(xyxy, confidences, class_ids, strict=False):
                if int(class_id) != 0:
                    continue
                x1, y1, x2, y2 = [int(round(float(v))) for v in box]
                predictions.append(
                    ScoredBox(
                        image_id=record.scoped_id,
                        box=(x1, y1, x2, y2),
                        score=float(score),
                        metadata={"source_detector": model_name},
                    )
                )
        except Exception as exc:  # pragma: no cover - evidence logging path
            failures.append({"image_id": record.scoped_id, "error": f"{type(exc).__name__}: {exc}"})
        if index % 25 == 0:
            elapsed = perf_counter() - infer_started
            rate = index / max(elapsed, 1e-9)
            remaining = (len(records) - index) / max(rate, 1e-9)
            used_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(
                f"[{model_name}] {index}/{len(records)} images, {rate:.2f} img/s, "
                f"ETA {remaining/60:.1f} min, peak VRAM {used_mb:.0f} MB",
                flush=True,
            )
    runtime = perf_counter() - started
    return predictions, {
        "candidate": model_name,
        "family": "rfdetr",
        "model_path": checkpoint_path,
        "image_size": "",
        "confidence": threshold,
        "runtime_seconds": round(runtime, 3),
        "load_optimize_seconds": round(load_seconds, 3),
        "images": len(records),
        "detections": len(predictions),
        "failures": len(failures),
        "failure_examples": json.dumps(failures[:5]),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }


def write_markdown_scores(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    overall = [row for row in rows if row["subgroup"] == "all_images"]
    overall = sorted(overall, key=lambda row: (row["protocol"], -float(row["oapr_detector_score"])))
    lines = [
        f"# {title}",
        "",
        "| Protocol | Method | OAPR score | F1 | Precision | Recall | TP | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['protocol']} | {row['model']} | {float(row['oapr_detector_score']):.4f} | "
            f"{float(row['f1']):.4f} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{row['true_positives']} | {row['false_positives']} | {row['false_negatives']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_union(path: Path, rows: list[dict[str, Any]]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/02_face_detection/08_sliced_rfdetr_detector_experiment")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-rfdetr", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocols = [
        Protocol("01_baseline_500", PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/01_baseline_500/manifest.csv"),
        Protocol("02_egocentric_stress_500", PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"),
    ]
    records = load_records(protocols)
    if args.max_images is not None:
        records = records[: args.max_images]
    records_by_id = {record.scoped_id: record for record in records}

    predictions_by_model: dict[str, list[ScoredBox]] = {}
    runtime_rows: list[dict[str, Any]] = []

    base_candidates = [
        Candidate("yolo11s_widerface_1280", "yolo", "detector", "data/models/face_detection_candidates/yolo11s_widerface.pt", 1280),
        Candidate("scrfd_10g_current_640", "scrfd", "detector", "/home/arun-gmr/.insightface/models/buffalo_l/det_10g.onnx", 640),
        Candidate("yolo8s_widerface_repo_640", "yolo", "detector", "data/models/face_detection_candidates/yolov8s_widerface.pt", 640),
        Candidate("yolo11n_pose_widerface_640", "yolo", "detector", "data/models/face_detection_candidates/yolo11n-pose_widerface.pt", 640),
    ]
    for candidate in base_candidates:
        print(f"\n=== Running {candidate.name} ===", flush=True)
        predictions, runtime = run_candidate(candidate, records)
        predictions_by_model[candidate.name] = predictions
        runtime_rows.append(runtime)

    print("\n=== Running sliced_yolo11s_widerface_1280 ===", flush=True)
    sliced, sliced_runtime = run_sliced_yolo(
        records=records,
        model_path="data/models/face_detection_candidates/yolo11s_widerface.pt",
        model_name="sliced_yolo11s_widerface_1280",
    )
    predictions_by_model["sliced_yolo11s_widerface_1280"] = sliced
    runtime_rows.append(sliced_runtime)

    rfdetr_name = "rfdetr_medium_face_030"
    rfdetr_checkpoint = (
        PROJECT_ROOT
        / "data/models/face_detection_candidates/rfdetr_hf_cache/models--Herojayjay--RFDETR-Face-Detection/"
        / "snapshots/597fcce941997900080ce8127b53a5d24e330225/rfdetr_medium_face.pth"
    )
    rfdetr_available = False
    if not args.skip_rfdetr:
        print(f"\n=== Running {rfdetr_name} ===", flush=True)
        try:
            rfdetr_predictions, rfdetr_runtime = run_rfdetr(
                records=records,
                checkpoint_path=str(rfdetr_checkpoint),
                model_name=rfdetr_name,
            )
            predictions_by_model[rfdetr_name] = rfdetr_predictions
            runtime_rows.append(rfdetr_runtime)
            rfdetr_available = True
        except Exception as exc:  # pragma: no cover - evidence logging path
            runtime_rows.append(
                {
                    "candidate": rfdetr_name,
                    "family": "rfdetr",
                    "model_path": str(rfdetr_checkpoint),
                    "confidence": 0.30,
                    "runtime_seconds": "",
                    "images": len(records),
                    "detections": 0,
                    "failures": len(records),
                    "failure_examples": f"{type(exc).__name__}: {exc}",
                }
            )

    fusion_specs = [
        ("fixed_fusion_yolo11s1280_scrfd10g", ["yolo11s_widerface_1280", "scrfd_10g_current_640"], 0.50, 0.22, 0.08, 0.90),
        ("fusion_sliced_yolo11s_scrfd10g", ["sliced_yolo11s_widerface_1280", "scrfd_10g_current_640"], 0.50, 0.22, 0.08, 0.88),
        ("fusion_sliced_yolo11s_yolo11s_scrfd10g", ["sliced_yolo11s_widerface_1280", "yolo11s_widerface_1280", "scrfd_10g_current_640"], 0.50, 0.22, 0.06, 0.86),
    ]
    if rfdetr_available:
        fusion_specs.extend(
            [
                ("fusion_rfdetr_scrfd10g", [rfdetr_name, "scrfd_10g_current_640"], 0.50, 0.20, 0.08, 0.90),
                (
                    "fusion_rfdetr_yolo11s_scrfd10g",
                    [rfdetr_name, "yolo11s_widerface_1280", "scrfd_10g_current_640"],
                    0.50,
                    0.20,
                    0.06,
                    0.88,
                ),
                (
                    "fusion_rfdetr_sliced_yolo11s_scrfd10g",
                    [rfdetr_name, "sliced_yolo11s_widerface_1280", "scrfd_10g_current_640"],
                    0.50,
                    0.20,
                    0.06,
                    0.88,
                ),
            ]
        )
    for name, sources, iou, min_score, bonus, penalty in fusion_specs:
        predictions_by_model[name] = nms_fusion(predictions_by_model, sources, name, iou, min_score, bonus, penalty)
        runtime_rows.append(
            {
                "candidate": name,
                "family": "fusion",
                "model_path": "+".join(sources),
                "image_size": "",
                "confidence": min_score,
                "runtime_seconds": "",
                "images": len(records),
                "detections": len(predictions_by_model[name]),
                "failures": 0,
                "failure_examples": "",
            }
        )

    scene_predictions = load_scene_predictions(SCENE_PREDICTIONS)
    source_names_sliced = [
        "sliced_yolo11s_widerface_1280",
        "yolo11s_widerface_1280",
        "scrfd_10g_current_640",
        "yolo8s_widerface_repo_640",
        "yolo11n_pose_widerface_640",
    ]
    clusters_sliced = cluster_predictions(predictions_by_model, source_names_sliced, records_by_id)
    reranker_predictions, thresholds_sliced = cross_validated_reranker(
        clusters=clusters_sliced,
        records=records,
        source_names=source_names_sliced,
        scene_predictions=scene_predictions,
        use_oracle_conditions=False,
    )
    predictions_by_model["cv_box_reranker_with_sliced_yolo_predicted_conditions"] = reranker_predictions
    for row in thresholds_sliced:
        row["reranker"] = "cv_box_reranker_with_sliced_yolo_predicted_conditions"

    threshold_rows = thresholds_sliced
    cluster_summary_rows = [
        {
            "reranker": "cv_box_reranker_with_sliced_yolo_predicted_conditions",
            "clusters": len(clusters_sliced),
            "positive_clusters": sum(cluster.label for cluster in clusters_sliced),
            "positive_rate": round(sum(cluster.label for cluster in clusters_sliced) / max(1, len(clusters_sliced)), 6),
            "source_detectors": "|".join(source_names_sliced),
        }
    ]

    if rfdetr_available:
        source_names_rfdetr_clean = [
            rfdetr_name,
            "yolo11s_widerface_1280",
            "scrfd_10g_current_640",
            "yolo8s_widerface_repo_640",
            "yolo11n_pose_widerface_640",
        ]
        clusters_rfdetr_clean = cluster_predictions(predictions_by_model, source_names_rfdetr_clean, records_by_id)
        rfdetr_clean_predictions, thresholds_rfdetr_clean = cross_validated_reranker(
            clusters=clusters_rfdetr_clean,
            records=records,
            source_names=source_names_rfdetr_clean,
            scene_predictions=scene_predictions,
            use_oracle_conditions=False,
        )
        for row in thresholds_rfdetr_clean:
            row["reranker"] = "cv_box_reranker_with_rfdetr_predicted_conditions"
        threshold_rows.extend(thresholds_rfdetr_clean)
        cluster_summary_rows.append(
            {
                "reranker": "cv_box_reranker_with_rfdetr_predicted_conditions",
                "clusters": len(clusters_rfdetr_clean),
                "positive_clusters": sum(cluster.label for cluster in clusters_rfdetr_clean),
                "positive_rate": round(
                    sum(cluster.label for cluster in clusters_rfdetr_clean) / max(1, len(clusters_rfdetr_clean)), 6
                ),
                "source_detectors": "|".join(source_names_rfdetr_clean),
            }
        )
        predictions_by_model["cv_box_reranker_with_rfdetr_predicted_conditions"] = rfdetr_clean_predictions

        source_names_rfdetr = [rfdetr_name, *source_names_sliced]
        clusters_rfdetr = cluster_predictions(predictions_by_model, source_names_rfdetr, records_by_id)
        rfdetr_reranker_predictions, thresholds_rfdetr = cross_validated_reranker(
            clusters=clusters_rfdetr,
            records=records,
            source_names=source_names_rfdetr,
            scene_predictions=scene_predictions,
            use_oracle_conditions=False,
        )
        for row in thresholds_rfdetr:
            row["reranker"] = "cv_box_reranker_with_rfdetr_sliced_predicted_conditions"
        threshold_rows.extend(thresholds_rfdetr)
        cluster_summary_rows.append(
            {
                "reranker": "cv_box_reranker_with_rfdetr_sliced_predicted_conditions",
                "clusters": len(clusters_rfdetr),
                "positive_clusters": sum(cluster.label for cluster in clusters_rfdetr),
                "positive_rate": round(sum(cluster.label for cluster in clusters_rfdetr) / max(1, len(clusters_rfdetr)), 6),
                "source_detectors": "|".join(source_names_rfdetr),
            }
        )
        predictions_by_model["cv_box_reranker_with_rfdetr_sliced_predicted_conditions"] = rfdetr_reranker_predictions

    score_methods = [
        "fixed_fusion_yolo11s1280_scrfd10g",
        "sliced_yolo11s_widerface_1280",
        "fusion_sliced_yolo11s_scrfd10g",
        "fusion_sliced_yolo11s_yolo11s_scrfd10g",
        "cv_box_reranker_with_sliced_yolo_predicted_conditions",
    ]
    if rfdetr_available:
        score_methods.extend(
            [
                rfdetr_name,
                "fusion_rfdetr_scrfd10g",
                "fusion_rfdetr_yolo11s_scrfd10g",
                "fusion_rfdetr_sliced_yolo11s_scrfd10g",
                "cv_box_reranker_with_rfdetr_predicted_conditions",
                "cv_box_reranker_with_rfdetr_sliced_predicted_conditions",
            ]
        )
    score_rows: list[dict[str, Any]] = []
    for name in score_methods:
        score_rows.extend(score_model(name, predictions_by_model[name], records))

    write_csv(output_dir / "sliced_detector_policy_scores.csv", score_rows)
    write_csv_union(output_dir / "sliced_detector_runtime.csv", runtime_rows)
    write_csv_union(output_dir / "sliced_reranker_thresholds.csv", threshold_rows)
    write_csv(output_dir / "sliced_reranker_cluster_summary.csv", cluster_summary_rows)
    write_markdown_scores(output_dir / "sliced_detector_policy_scores.md", score_rows, "Sliced Detector Policy Scores")
    combined = [row for row in score_rows if row["protocol"] == "combined_1000" and row["subgroup"] == "all_images"]
    best = max(combined, key=lambda row: float(row["oapr_detector_score"]))
    sliced_row = next(row for row in combined if row["model"] == "sliced_yolo11s_widerface_1280")
    fixed_row = next(row for row in combined if row["model"] == "fixed_fusion_yolo11s1280_scrfd10g")
    rfdetr_row = next((row for row in combined if row["model"] == rfdetr_name), None)
    (output_dir / "sliced_detector_policy_summary.md").write_text(
        "\n".join(
            [
                "# Sliced Detector Policy Experiment",
                "",
                "This low-compute experiment tests whether overlapping 1280px YOLO11Face tiles and RF-DETR Medium face detection improve the detector policy on the two reviewed 500-image protocols.",
                "",
                f"RF-DETR status: {'run on CUDA and included in the score table' if rfdetr_available else 'not included because the CUDA/checkpoint smoke failed or --skip-rfdetr was used'}.",
                "",
                "Main result:",
                "",
                f"- Best combined 1,000-image method in this experiment: `{best['model']}`.",
                f"- Score: OAPR detector score `{float(best['oapr_detector_score']):.4f}`, F1 `{float(best['f1']):.4f}`, precision `{float(best['precision']):.4f}`, recall `{float(best['recall']):.4f}`, TP `{best['true_positives']}`, FP `{best['false_positives']}`, FN `{best['false_negatives']}`.",
                f"- Fixed YOLO11/SCRFD fusion comparator: score `{float(fixed_row['oapr_detector_score']):.4f}`, F1 `{float(fixed_row['f1']):.4f}`, precision `{float(fixed_row['precision']):.4f}`, recall `{float(fixed_row['recall']):.4f}`.",
                "",
                "Sliced inference result:",
                "",
                f"- Sliced YOLO11Face alone: score `{float(sliced_row['oapr_detector_score']):.4f}`, F1 `{float(sliced_row['f1']):.4f}`, precision `{float(sliced_row['precision']):.4f}`, recall `{float(sliced_row['recall']):.4f}`.",
                "- Sliced inference is feasible, but it is not promoted when false-positive control is part of the objective.",
                "",
                "RF-DETR result:",
                "",
                (
                    f"- RF-DETR alone: score `{float(rfdetr_row['oapr_detector_score']):.4f}`, F1 `{float(rfdetr_row['f1']):.4f}`, precision `{float(rfdetr_row['precision']):.4f}`, recall `{float(rfdetr_row['recall']):.4f}`."
                    if rfdetr_row
                    else "- RF-DETR was not scored in this run."
                ),
                "- RF-DETR is promoted only through the RF-DETR-aware box reranker, not as a standalone detector replacement.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote sliced detector outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
