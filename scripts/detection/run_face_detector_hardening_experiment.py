#!/usr/bin/env python3

"""Benchmark reproducible face-detector candidates and fusion variants.

This script is intentionally self-contained because the cleaned annotation
manifests live under ``outputs/01_protocol/annotations/face_detection`` rather than
the root-level CASTLE manifest layout expected by the generic dataset loader.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.scrfd_detector import SCRFDDetector
from src.detection.face_detection_package_detector import FaceDetectionPackageDetector
from src.detection.yunet_detector import YuNetDetector
from src.detection.yolo_detector import YOLODetector
from src.evaluation.detection_metrics import (
    GroundTruthBox,
    ScoredBox,
    compute_average_precision,
    compute_iou,
)


RAW_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"


@dataclass(frozen=True)
class Protocol:
    name: str
    manifest: Path


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    kind: str
    model_path: str
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.5
    note: str = ""


@dataclass(frozen=True)
class ImageRecord:
    protocol: str
    image_id: str
    relative_path: str
    path: Path
    gt_boxes: tuple[tuple[int, int, int, int], ...]
    attributes: dict[str, str]

    @property
    def scoped_id(self) -> str:
        return f"{self.protocol}::{self.image_id}"


def load_records(protocols: list[Protocol]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for protocol in protocols:
        with protocol.manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                relative_path = row["relative_path"]
                image_id = row.get("image_id") or relative_path
                boxes_payload = row.get("reviewed_face_boxes_json", "[]") or "[]"
                boxes = []
                for item in json.loads(boxes_payload):
                    boxes.append((int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])))
                records.append(
                    ImageRecord(
                        protocol=protocol.name,
                        image_id=image_id,
                        relative_path=relative_path,
                        path=RAW_ROOT / relative_path,
                        gt_boxes=tuple(boxes),
                        attributes=row,
                    )
                )
    return records


def build_subgroup_membership(records: list[ImageRecord]) -> dict[str, set[str]]:
    memberships: dict[str, set[str]] = {"all_images": {record.scoped_id for record in records}}
    checks = {
        "multi_face": lambda a: a.get("face_count_category") == "multi_face",
        "single_face": lambda a: a.get("face_count_category") == "single_face",
        "no_face": lambda a: a.get("face_count_category") == "no_face",
        "very_small_or_distant_face": lambda a: a.get("face_scale_category") == "very_small_or_distant",
        "small_face": lambda a: a.get("face_scale_category") == "small",
        "medium_face": lambda a: a.get("face_scale_category") == "medium",
        "large_face": lambda a: a.get("face_scale_category") == "large",
        "mixed_scale_face": lambda a: a.get("face_scale_category") == "mixed_scale",
        "edge_or_partial_face": lambda a: a.get("edge_partial_face") == "yes",
        "profile_or_occluded_face": lambda a: a.get("profile_occluded_face") == "yes",
        "downward_egocentric_view": lambda a: a.get("downward_egocentric_view") == "yes",
        "motion_blur_or_low_sharpness": lambda a: a.get("blur_low_sharpness") == "yes",
        "low_light_or_dim": lambda a: a.get("low_light_dim") == "yes",
        "high_clutter": lambda a: a.get("clutter_level") == "high",
        "outdoor_or_vehicle_scene": lambda a: a.get("outdoor_vehicle_scene") == "yes",
    }
    for subgroup, predicate in checks.items():
        memberships[subgroup] = {record.scoped_id for record in records if predicate(record.attributes)}
    return memberships


def build_yolo(candidate: Candidate) -> YOLODetector:
    return YOLODetector(
        model_path=candidate.model_path,
        confidence_threshold=candidate.confidence,
        iou_threshold=candidate.iou,
        device="0",
        image_size=candidate.image_size,
        allowed_class_ids=[0],
    )


def build_scrfd(candidate: Candidate) -> SCRFDDetector:
    return SCRFDDetector(
        model_path=candidate.model_path,
        confidence_threshold=candidate.confidence,
        input_size=(candidate.image_size, candidate.image_size),
    )


def run_candidate(candidate: Candidate, records: list[ImageRecord]) -> tuple[list[ScoredBox], dict[str, Any]]:
    if candidate.family == "yolo":
        detector = build_yolo(candidate)
    elif candidate.family == "scrfd":
        detector = build_scrfd(candidate)
    elif candidate.family == "face_detection_package":
        detector = FaceDetectionPackageDetector(
            backend_name=candidate.model_path,
            detector_name=candidate.name,
            confidence_threshold=candidate.confidence,
            nms_iou_threshold=candidate.iou,
            device="cuda",
            max_resolution=candidate.image_size,
            fp16_inference=True,
        )
    elif candidate.family == "yunet":
        detector = YuNetDetector(
            model_path=candidate.model_path,
            confidence_threshold=candidate.confidence,
            nms_threshold=candidate.iou,
            max_input_size=candidate.image_size,
        )
    else:
        raise ValueError(f"Unsupported candidate family: {candidate.family}")
    predictions: list[ScoredBox] = []
    started = perf_counter()
    failures: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        try:
            with Image.open(record.path) as loaded:
                image = loaded.convert("RGB")
                result = detector.detect(image)
        except Exception as exc:  # pragma: no cover - evidence logging path
            failures.append({"image_id": record.scoped_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        for det in result.detections:
            predictions.append(
                ScoredBox(
                    image_id=record.scoped_id,
                    box=det.box,
                    score=float(det.confidence),
                    metadata={"source_detector": candidate.name},
                )
            )
        if index % 50 == 0:
            elapsed = perf_counter() - started
            rate = index / max(elapsed, 1e-9)
            remaining = (len(records) - index) / max(rate, 1e-9)
            print(
                f"[{candidate.name}] {index}/{len(records)} images, "
                f"{rate:.2f} img/s, ETA {remaining/60:.1f} min",
                flush=True,
            )
    runtime = perf_counter() - started
    unload_detector(detector)
    return predictions, {
        "candidate": candidate.name,
        "family": candidate.family,
        "model_path": candidate.model_path,
        "image_size": candidate.image_size,
        "confidence": candidate.confidence,
        "runtime_seconds": round(runtime, 3),
        "images": len(records),
        "detections": len(predictions),
        "failures": len(failures),
        "failure_examples": json.dumps(failures[:5]),
    }


def unload_detector(detector: Any) -> None:
    del detector
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def nms_fusion(
    predictions_by_model: dict[str, list[ScoredBox]],
    sources: list[str],
    fused_name: str,
    iou_threshold: float,
    min_score: float,
    agreement_bonus: float,
    single_detector_penalty: float,
) -> list[ScoredBox]:
    by_image: dict[str, list[ScoredBox]] = {}
    for source in sources:
        for prediction in predictions_by_model[source]:
            by_image.setdefault(prediction.image_id, []).append(prediction)

    fused: list[ScoredBox] = []
    for image_id, boxes in by_image.items():
        pending = sorted(boxes, key=lambda item: item.score, reverse=True)
        clusters: list[list[ScoredBox]] = []
        while pending:
            seed = pending.pop(0)
            cluster = [seed]
            remaining = []
            for candidate in pending:
                if compute_iou(seed.box, candidate.box) >= iou_threshold:
                    cluster.append(candidate)
                else:
                    remaining.append(candidate)
            pending = remaining
            clusters.append(cluster)

        for cluster in clusters:
            detectors = {str(item.metadata.get("source_detector")) for item in cluster if item.metadata}
            weights = [max(item.score, 1e-6) for item in cluster]
            total = sum(weights)
            coords = [
                int(round(sum(item.box[idx] * weight for item, weight in zip(cluster, weights, strict=False)) / total))
                for idx in range(4)
            ]
            base_score = max(item.score for item in cluster)
            score = base_score + agreement_bonus * max(0, len(detectors) - 1)
            if len(detectors) == 1:
                score *= single_detector_penalty
            score = min(1.0, float(score))
            if score < min_score:
                continue
            fused.append(
                ScoredBox(
                    image_id=image_id,
                    box=(coords[0], coords[1], coords[2], coords[3]),
                    score=score,
                    metadata={"source_detector": fused_name, "source_count": len(detectors)},
                )
            )
    return fused


def gt_rows(records: list[ImageRecord]) -> list[GroundTruthBox]:
    rows: list[GroundTruthBox] = []
    for record in records:
        for box in record.gt_boxes:
            rows.append(GroundTruthBox(image_id=record.scoped_id, box=box))
    return rows


def oapr_score(metric: dict[str, Any], ground_truth_boxes: int, image_count: int) -> float:
    if ground_truth_boxes == 0:
        return float(metric.get("zero_face_specificity", 0.0))
    precision = float(metric["precision"])
    recall = float(metric["recall"])
    f1 = float(metric["f1"])
    return 0.65 * recall + 0.25 * f1 + 0.10 * precision


def score_predictions(
    model_name: str,
    predictions: list[ScoredBox],
    records: list[ImageRecord],
    memberships: dict[str, set[str]],
    protocol: str,
) -> list[dict[str, Any]]:
    gt = gt_rows(records)
    rows: list[dict[str, Any]] = []
    pred_by_image = {item.image_id for item in predictions}
    for subgroup, ids in memberships.items():
        subset_records = [record for record in records if record.scoped_id in ids]
        gt_subset = [item for item in gt if item.image_id in ids]
        pred_subset = [item for item in predictions if item.image_id in ids]
        metric = compute_average_precision(pred_subset, gt_subset, iou_threshold=0.5)
        false_positive_images = len(pred_by_image & ids) if not gt_subset else 0
        specificity = 1.0 - (false_positive_images / max(1, len(ids))) if not gt_subset else ""
        metric["zero_face_false_positive_images"] = false_positive_images
        metric["zero_face_specificity"] = specificity
        row = {
            "protocol": protocol,
            "model": model_name,
            "subgroup": subgroup,
            "image_count": len(subset_records),
            "ground_truth_boxes": len(gt_subset),
            "prediction_boxes": len(pred_subset),
            **metric,
        }
        row["oapr_detector_score"] = oapr_score(row, len(gt_subset), len(subset_records))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| Protocol | Subgroup | Best model | OAPR score | F1 | Precision | Recall | TP | FP | FN | Margin vs second |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['subgroup']} | {row['best_model']} | "
            f"{float(row['best_oapr_detector_score']):.4f} | {float(row['best_f1']):.4f} | "
            f"{float(row['best_precision']):.4f} | {float(row['best_recall']):.4f} | "
            f"{row['best_true_positives']} | {row['best_false_positives']} | {row['best_false_negatives']} | "
            f"{float(row['margin_vs_second']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in score_rows:
        grouped.setdefault((str(row["protocol"]), str(row["subgroup"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (protocol, subgroup), rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda item: (
                float(item["oapr_detector_score"]),
                float(item["recall"]),
                float(item["f1"]),
                float(item["precision"]),
            ),
            reverse=True,
        )
        best = ranked[0]
        second_score = float(ranked[1]["oapr_detector_score"]) if len(ranked) > 1 else float("nan")
        out.append(
            {
                "protocol": protocol,
                "subgroup": subgroup,
                "image_count": best["image_count"],
                "ground_truth_boxes": best["ground_truth_boxes"],
                "best_model": best["model"],
                "best_oapr_detector_score": best["oapr_detector_score"],
                "second_model": ranked[1]["model"] if len(ranked) > 1 else "",
                "second_oapr_detector_score": second_score if not math.isnan(second_score) else "",
                "margin_vs_second": float(best["oapr_detector_score"]) - second_score if not math.isnan(second_score) else "",
                "best_f1": best["f1"],
                "best_precision": best["precision"],
                "best_recall": best["recall"],
                "best_true_positives": best["true_positives"],
                "best_false_positives": best["false_positives"],
                "best_false_negatives": best["false_negatives"],
            }
        )
    return out


def candidate_availability() -> list[dict[str, str]]:
    return [
        {
            "candidate": "YOLO11Face yolo11s_widerface",
            "status": "RUNNABLE_5060",
            "source": "https://github.com/zjykzj/YOLO11Face/releases/tag/v1.0.0",
            "decision": "Tested as Ultralytics WIDERFace detector.",
        },
        {
            "candidate": "YOLO11Face yolo11n-pose_widerface",
            "status": "RUNNABLE_5060",
            "source": "https://github.com/zjykzj/YOLO11Face/releases/tag/v1.0.0",
            "decision": "Tested because pose-trained face boxes may improve partial/profile cases.",
        },
        {
            "candidate": "SCRFD current 10G InsightFace ONNX",
            "status": "RUNNABLE_5060",
            "source": "InsightFace buffalo_l/det_10g.onnx",
            "decision": "Tested as the canonical SCRFD implementation.",
        },
        {
            "candidate": "SCRFD HF ONNX 500M/1G/2.5G/34G mirrors",
            "status": "ATTEMPTED_NOT_PLUG_COMPATIBLE",
            "source": "https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX",
            "decision": "Attempted through existing InsightFace/SCRFD path; variants returned broadcast-shape errors and are excluded from ranking.",
        },
        {
            "candidate": "RF-DETR face checkpoint",
            "status": "A100_OR_SEPARATE_SETUP_RECOMMENDED",
            "source": "https://huggingface.co/Herojayjay/RFDETR-Face-Detection",
            "decision": "Checkpoint exists, but requires RF-DETR runtime and is medium transformer detector; not pulled into tonight's 5060 run unless final fusion still underperforms.",
        },
        {
            "candidate": "SFE-DETR",
            "status": "NOT_RUNNABLE_FROM_CLEAN_ASSET_SEARCH",
            "source": "repository/checkpoint not found in HF/GitHub search",
            "decision": "Keep as literature candidate only until reproducible weights are identified.",
        },
        {
            "candidate": "YOLOv12-face",
            "status": "NOT_RUNNABLE_FROM_CLEAN_ASSET_SEARCH",
            "source": "no clean WIDERFace face-specific weights found in HF/GitHub search",
            "decision": "Do not include without reproducible face-trained weights.",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/02_face_detection/06_detector_hardening_experiment")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    protocols = [
        Protocol("01_baseline_500", PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/01_baseline_500/manifest.csv"),
        Protocol("02_egocentric_stress_500", PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"),
    ]
    records = load_records(protocols)
    if args.max_images is not None:
        records = records[: args.max_images]

    candidates = [
        Candidate("yolo8s_lindevs_640", "yolo", "detector", "data/models/yolov8s-face-lindevs.pt", 640),
        Candidate("yolo8s_lindevs_1280", "yolo", "detector", "data/models/yolov8s-face-lindevs.pt", 1280),
        Candidate("yolo8s_widerface_repo_640", "yolo", "detector", "data/models/face_detection_candidates/yolov8s_widerface.pt", 640),
        Candidate("yolo11s_widerface_640", "yolo", "detector", "data/models/face_detection_candidates/yolo11s_widerface.pt", 640),
        Candidate("yolo11s_widerface_1280", "yolo", "detector", "data/models/face_detection_candidates/yolo11s_widerface.pt", 1280),
        Candidate("yolo11n_pose_widerface_640", "yolo", "detector", "data/models/face_detection_candidates/yolo11n-pose_widerface.pt", 640),
        Candidate("scrfd_500m_640", "scrfd", "detector", "data/models/face_detection_candidates/scrfd_500m.onnx", 640),
        Candidate("scrfd_1g_640", "scrfd", "detector", "data/models/face_detection_candidates/scrfd_1g.onnx", 640),
        Candidate("scrfd_2_5g_640", "scrfd", "detector", "data/models/face_detection_candidates/scrfd_2.5g_bnkps.onnx", 640),
        Candidate("scrfd_10g_current_640", "scrfd", "detector", "/home/arun-gmr/.insightface/models/buffalo_l/det_10g.onnx", 640),
        Candidate("scrfd_34g_640", "scrfd", "detector", "data/models/face_detection_candidates/scrfd_34g.onnx", 640),
    ]

    write_csv(output_dir / "candidate_availability.csv", candidate_availability())
    availability_lines = ["# Detector Candidate Availability", ""]
    for row in candidate_availability():
        availability_lines.append(f"- `{row['candidate']}`: `{row['status']}`. {row['decision']} Source: {row['source']}")
    (output_dir / "candidate_availability.md").write_text("\n".join(availability_lines) + "\n", encoding="utf-8")

    predictions_by_model: dict[str, list[ScoredBox]] = {}
    runtime_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        print(f"\n=== Running {candidate.name} ===", flush=True)
        preds, runtime = run_candidate(candidate, records)
        predictions_by_model[candidate.name] = preds
        runtime_rows.append(runtime)

    failed_models = {
        str(row["candidate"])
        for row in runtime_rows
        if int(row.get("failures") or 0) >= int(row.get("images") or 0)
    }

    fusion_specs = [
        ("fusion_yolo8s_scrfd10g_agreement", ["yolo8s_lindevs_640", "scrfd_10g_current_640"], 0.50, 0.22, 0.08, 0.90),
        ("fusion_yolo11s_scrfd10g_agreement", ["yolo11s_widerface_640", "scrfd_10g_current_640"], 0.50, 0.22, 0.08, 0.90),
        ("fusion_yolo8s_yolo11s_scrfd10g_agreement", ["yolo8s_lindevs_640", "yolo11s_widerface_640", "scrfd_10g_current_640"], 0.50, 0.22, 0.06, 0.88),
        ("fusion_yolo11s1280_scrfd10g_agreement", ["yolo11s_widerface_1280", "scrfd_10g_current_640"], 0.50, 0.22, 0.08, 0.90),
        ("fusion_yolo11s_scrfd34g_agreement", ["yolo11s_widerface_640", "scrfd_34g_640"], 0.50, 0.22, 0.08, 0.90),
    ]
    for name, sources, iou, min_score, bonus, penalty in fusion_specs:
        if any(source in failed_models for source in sources):
            runtime_rows.append(
                {
                    "candidate": name,
                    "family": "fusion",
                    "model_path": "+".join(sources),
                    "image_size": "",
                    "confidence": min_score,
                    "runtime_seconds": "",
                    "images": len(records),
                    "detections": 0,
                    "failures": len(records),
                    "failure_examples": f"Skipped because source detector failed: {sorted(set(sources) & failed_models)}",
                }
            )
            failed_models.add(name)
            continue
        print(f"\n=== Building {name} ===", flush=True)
        predictions_by_model[name] = nms_fusion(
            predictions_by_model=predictions_by_model,
            sources=sources,
            fused_name=name,
            iou_threshold=iou,
            min_score=min_score,
            agreement_bonus=bonus,
            single_detector_penalty=penalty,
        )
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

    all_score_rows: list[dict[str, Any]] = []
    for protocol in sorted({record.protocol for record in records}):
        protocol_records = [record for record in records if record.protocol == protocol]
        memberships = build_subgroup_membership(protocol_records)
        for model_name, predictions in predictions_by_model.items():
            if model_name in failed_models:
                continue
            protocol_predictions = [item for item in predictions if item.image_id.startswith(f"{protocol}::")]
            all_score_rows.extend(score_predictions(model_name, protocol_predictions, protocol_records, memberships, protocol))

    combined_memberships = build_subgroup_membership(records)
    for model_name, predictions in predictions_by_model.items():
        if model_name in failed_models:
            continue
        all_score_rows.extend(score_predictions(model_name, predictions, records, combined_memberships, "combined_1000"))

    best = best_rows(all_score_rows)
    write_csv(output_dir / "detector_hardening_subgroup_scores.csv", all_score_rows)
    write_csv(output_dir / "detector_hardening_best_by_subgroup.csv", best)
    write_csv(output_dir / "detector_hardening_runtime.csv", runtime_rows)
    write_markdown(output_dir / "detector_hardening_best_by_subgroup.md", best, "Face Detector Hardening Best by Subgroup")

    overall_rows = [row for row in all_score_rows if row["subgroup"] == "all_images"]
    overall_rows = sorted(
        overall_rows,
        key=lambda item: (str(item["protocol"]), -float(item["oapr_detector_score"])),
    )
    write_csv(output_dir / "detector_hardening_overall_scores.csv", overall_rows)
    lines = [
        "# Face Detector Hardening Overall Scores",
        "",
        "| Protocol | Model | OAPR score | F1 | Precision | Recall | TP | FP | FN | Predictions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['protocol']} | {row['model']} | {float(row['oapr_detector_score']):.4f} | "
            f"{float(row['f1']):.4f} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{row['true_positives']} | {row['false_positives']} | {row['false_negatives']} | {row['prediction_boxes']} |"
        )
    (output_dir / "detector_hardening_overall_scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hypothesis = [
        "# Face Detector Hardening Hypothesis",
        "",
        "The best path is not a single replacement detector. The tested hypothesis is that CASTLE face detection improves when face-specific YOLO and SCRFD candidates are combined with model-size/multiscale variants and agreement-aware fusion.",
        "",
        "The score used here is the project detector-routing score: no-face subgroups use specificity, while face-positive subgroups use `0.65*recall + 0.25*F1 + 0.10*precision`.",
        "",
        "Final interpretation should be based on `detector_hardening_best_by_subgroup.csv` and `detector_hardening_overall_scores.csv`.",
    ]
    (output_dir / "detector_hardening_hypothesis.md").write_text("\n".join(hypothesis) + "\n", encoding="utf-8")
    print(f"\nWrote final detector hardening outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
