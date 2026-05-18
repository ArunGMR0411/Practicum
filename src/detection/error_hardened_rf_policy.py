"""Adopted final face-detector policy: error_hardened_all_raw_rf_iou0_45.

Scientific score (fold-safe CV OOF on retained candidates):
  OAPR detector composite 0.9172, precision 0.9319, recall 0.9129, F1 0.9223
  Source: outputs/02_face_detection/12_detector_error_hardening/
  Decision: outputs/02_face_detection/14_final_detector_policy/02_detector_policy_decision.md

This module:
  - names the adopted policy consistently
  - clusters multi-detector candidates at IoU 0.45
  - applies a deployable RandomForest filter trained on geometry + source features
  - does NOT depend on offline scene-condition telemetry at runtime

The deploy model is intentionally a runtime artefact (geometry/source features only).
The thesis number 0.9172 remains the offline CV evaluation of the full feature recipe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ADOPTED_FACE_DETECTOR_POLICY_ID = "error_hardened_all_raw_rf_iou0_45"
RUNTIME_FACE_DETECTOR_POLICY_ID = "runtime_3_source_all_raw_rf_approximation"
ADOPTED_FACE_DETECTOR_EXPLORATORY_SCORE = 0.917165
ADOPTED_FACE_DETECTOR_PRECISION = 0.931872
ADOPTED_FACE_DETECTOR_RECALL = 0.912927
ADOPTED_FACE_DETECTOR_F1 = 0.922302
ADOPTED_FACE_DETECTOR_TP = 2380
ADOPTED_FACE_DETECTOR_FP = 174
ADOPTED_FACE_DETECTOR_FN = 227

# Candidate bank used by the offline hardening experiment (all_raw).
ALL_RAW_SOURCE_NAMES: tuple[str, ...] = (
    "yolo11s_widerface_1280",
    "scrfd_10g_current_640",
    "yolo8s_widerface_repo_640",
    "yolo11n_pose_widerface_640",
    "yolo11s_widerface_640",
    "sliced_yolo11s_widerface_1280",
    "rfdetr_medium_face_030",
)

# Sources the live App can generate without sliced inference / extra YOLO variants.
RUNTIME_SOURCE_NAMES: tuple[str, ...] = (
    "rfdetr_medium_face_030",
    "yolo11s_widerface_1280",
    "scrfd_10g_current_640",
)

CLUSTER_IOU = 0.45
DEFAULT_THRESHOLD = 0.37  # median fold threshold from offline CV
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs/02_face_detection/12_detector_error_hardening"
    / "deploy_error_hardened_all_raw_rf_iou0_45.joblib"
)
DEFAULT_META_PATH = DEFAULT_MODEL_PATH.with_suffix(".json")

Box = tuple[int, int, int, int]
ScoredBox = tuple[int, int, int, int, float]


@dataclass(frozen=True)
class DetectorCandidate:
    box: Box
    score: float
    source: str


def compute_iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1.0, float(area_a + area_b - inter))


def cluster_candidates(
    candidates: Sequence[DetectorCandidate],
    *,
    iou_threshold: float = CLUSTER_IOU,
    source_names: Sequence[str] = ALL_RAW_SOURCE_NAMES,
) -> list[dict[str, Any]]:
    """Greedy high-score-first clustering (matches offline hardening)."""
    pending = sorted(candidates, key=lambda item: item.score, reverse=True)
    clusters: list[dict[str, Any]] = []
    while pending:
        seed = pending.pop(0)
        cluster = [seed]
        remaining: list[DetectorCandidate] = []
        for item in pending:
            if compute_iou(seed.box, item.box) >= iou_threshold:
                cluster.append(item)
            else:
                remaining.append(item)
        pending = remaining
        weights = [max(item.score, 1e-6) for item in cluster]
        total = sum(weights)
        box = tuple(
            int(
                round(
                    sum(item.box[idx] * weight for item, weight in zip(cluster, weights, strict=False))
                    / total
                )
            )
            for idx in range(4)
        )
        source_scores = {
            name: max([item.score for item in cluster if item.source == name] or [0.0])
            for name in source_names
        }
        clusters.append(
            {
                "box": box,
                "score": max(item.score for item in cluster),
                "sources": {item.source for item in cluster},
                "source_scores": source_scores,
            }
        )
    return clusters


def cluster_feature_row(
    cluster: dict[str, Any],
    *,
    image_width: float,
    image_height: float,
    source_names: Sequence[str] = ALL_RAW_SOURCE_NAMES,
) -> list[float]:
    """Geometry + multi-source agreement features (runtime recipe)."""
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    x1, y1, x2, y2 = cluster["box"]
    bw = max(0.0, (x2 - x1) / width)
    bh = max(0.0, (y2 - y1) / height)
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    edge_distance = min(x1 / width, y1 / height, (width - x2) / width, (height - y2) / height)
    source_scores = np.asarray(
        [float(cluster["source_scores"].get(name, 0.0)) for name in source_names],
        dtype=float,
    )
    row: list[float] = [
        float(cluster["score"]),
        float(len(cluster["sources"])),
        bw,
        bh,
        bw * bh,
        cx,
        cy,
        edge_distance,
        bh / max(bw, 1e-9),
        float(source_scores.max()) if len(source_scores) else 0.0,
        float(source_scores.mean()) if len(source_scores) else 0.0,
        float(source_scores.std()) if len(source_scores) else 0.0,
        float(source_scores.max() - source_scores.min()) if len(source_scores) else 0.0,
        float((source_scores > 0).sum()),
    ]
    for name in source_names:
        row.extend(
            [
                float(name in cluster["sources"]),
                float(cluster["source_scores"].get(name, 0.0)),
            ]
        )
    return row


def adopted_policy_evidence_text() -> str:
    return (
        f"Adopted final detector `{ADOPTED_FACE_DETECTOR_POLICY_ID}`: "
        f"exploratory OAPR detector score {ADOPTED_FACE_DETECTOR_EXPLORATORY_SCORE:.4f}; "
        f"precision {ADOPTED_FACE_DETECTOR_PRECISION:.4f}; "
        f"recall {ADOPTED_FACE_DETECTOR_RECALL:.4f}; "
        f"F1 {ADOPTED_FACE_DETECTOR_F1:.4f} "
        f"(TP {ADOPTED_FACE_DETECTOR_TP}, FP {ADOPTED_FACE_DETECTOR_FP}, FN {ADOPTED_FACE_DETECTOR_FN}) "
        "on the combined 1,000-image reviewed protocol."
    )


def runtime_policy_evidence_text() -> str:
    return (
        f"Bounded App detector `{RUNTIME_FACE_DETECTOR_POLICY_ID}` uses RF-DETR, "
        "YOLO11Face-1280 and SCRFD candidates with the retained all_raw RF filter. "
        "The seven-source scientific score 0.917165 is not assigned to this runtime approximation."
    )


class ErrorHardenedRFFilter:
    """Apply the deployable RF cluster filter for the adopted detector policy."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        source_names: Sequence[str] = ALL_RAW_SOURCE_NAMES,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.source_names = tuple(source_names)
        self._model = None
        self.threshold = DEFAULT_THRESHOLD
        self.meta: dict[str, Any] = {}
        self._load()

    @property
    def available(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if not self.model_path.is_file():
            return
        try:
            import joblib
        except Exception:
            return
        payload = joblib.load(self.model_path)
        if isinstance(payload, dict):
            self._model = payload.get("model")
            self.threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))
            self.meta = {k: v for k, v in payload.items() if k not in {"model"}}
            names = payload.get("source_names")
            if names:
                self.source_names = tuple(names)
        else:
            self._model = payload
        meta_path = self.model_path.with_suffix(".json")
        if meta_path.is_file():
            try:
                self.meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                pass

    def filter_clusters(
        self,
        clusters: Sequence[dict[str, Any]],
        *,
        image_width: float,
        image_height: float,
    ) -> list[ScoredBox]:
        if not clusters:
            return []
        if self._model is None:
            # Keep high-score multi-source clusters.
            output: list[ScoredBox] = []
            for cluster in clusters:
                score = float(cluster["score"])
                if len(cluster["sources"]) == 1:
                    score *= 0.88
                if score >= 0.25:
                    box = cluster["box"]
                    output.append((*box, min(1.0, score)))
            return output

        rows = np.asarray(
            [
                cluster_feature_row(
                    cluster,
                    image_width=image_width,
                    image_height=image_height,
                    source_names=self.source_names,
                )
                for cluster in clusters
            ],
            dtype=float,
        )
        probabilities = self._model.predict_proba(rows)[:, 1]
        output = []
        for cluster, probability in zip(clusters, probabilities, strict=True):
            if float(probability) >= self.threshold:
                box = cluster["box"]
                output.append((*box, float(probability)))
        return output

    def apply(
        self,
        candidates: Sequence[DetectorCandidate],
        *,
        image_width: float,
        image_height: float,
        iou_threshold: float = CLUSTER_IOU,
    ) -> tuple[list[ScoredBox], dict[str, Any]]:
        clusters = cluster_candidates(
            candidates,
            iou_threshold=iou_threshold,
            source_names=self.source_names,
        )
        boxes = self.filter_clusters(
            clusters,
            image_width=image_width,
            image_height=image_height,
        )
        return boxes, {
            "policy_id": ADOPTED_FACE_DETECTOR_POLICY_ID,
            "cluster_count": len(clusters),
            "kept_count": len(boxes),
            "threshold": self.threshold,
            "model_available": self.available,
            "model_path": str(self.model_path),
            "source_names": list(self.source_names),
            "cluster_iou": iou_threshold,
            "offline_exploratory_score": ADOPTED_FACE_DETECTOR_EXPLORATORY_SCORE,
        }
