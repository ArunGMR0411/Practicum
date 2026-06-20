"""Deploy the face-detector policy selected by the thesis evidence.

App default (balanced/standard): ``runtime_3_source_all_raw_rf_approximation``.
It is a bounded deployment implementation and is not assigned the scientific
seven-source detector score.

Live path:
  1) generate multi-detector candidates (RF-DETR + YOLO11Face + SCRFD by default)
  2) cluster at IoU 0.45
  3) apply the deployable error-hardened RF filter when the model is present

Supporting tiers (fusion / YOLO-only) remain available for lower-compute fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from src.detection.error_hardened_rf_policy import (
    ADOPTED_FACE_DETECTOR_POLICY_ID,
    RUNTIME_FACE_DETECTOR_POLICY_ID,
    ALL_RAW_SOURCE_NAMES,
    RUNTIME_SOURCE_NAMES,
    CLUSTER_IOU,
    DetectorCandidate,
    ErrorHardenedRFFilter,
    adopted_policy_evidence_text,
    runtime_policy_evidence_text,
)


ROOT = Path(__file__).resolve().parents[3]
_RFDETR_CANDIDATES = (
    ROOT
    / "data/models/face_detection_candidates/rfdetr_hf_cache/"
    / "models--Herojayjay--RFDETR-Face-Detection/snapshots/"
    / "597fcce941997900080ce8127b53a5d24e330225/rfdetr_medium_face.pth",
    ROOT / "data/models/face_detection_candidates/rfdetr_medium_face.pth",
    ROOT / "data/models/face_detection_candidates/rfdetr_download/rfdetr_medium_face.pth",
)


def _resolve_existing(path: Path) -> Path | None:
    """Return path if it exists as a real file (broken symlinks count as missing)."""
    try:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path.resolve() if path.is_symlink() else path
    except OSError:
        return None
    return None


def resolve_rfdetr_checkpoint() -> Path:
    for cand in _RFDETR_CANDIDATES:
        found = _resolve_existing(cand)
        if found is not None:
            return found
    return _RFDETR_CANDIDATES[0]


RFDETR_CHECKPOINT = resolve_rfdetr_checkpoint()
YOLO11_FACE_CHECKPOINT = ROOT / "data/models/face_detection_candidates/yolo11s_widerface.pt"
SCRFD_CHECKPOINT = Path.home() / ".insightface/models/buffalo_l/det_10g.onnx"

# Preferred → portable when checkpoints/packages/CUDA are unavailable.
FACE_POLICY_FALLBACK_ORDER: tuple[str, ...] = (
    RUNTIME_FACE_DETECTOR_POLICY_ID,
    "fusion_rfdetr_yolo11s_scrfd10g",
    "fusion_rfdetr_scrfd10g",
    "fixed_fusion_yolo11s1280_scrfd10g",
    "yolo11s_face",
)

RFDETR_THRESHOLD = 0.30
YOLO_THRESHOLD = 0.25
YOLO_IMAGE_SIZE = 1280
SCRFD_THRESHOLD = 0.25
SCRFD_INPUT_SIZE = (640, 640)
FUSION_IOU = 0.50
FUSION_MIN_SCORE = 0.20
FUSION_AGREEMENT_BONUS = 0.06
FUSION_SINGLE_DETECTOR_PENALTY = 0.88

Box = tuple[int, int, int, int]
ScoredBox = tuple[int, int, int, int, float]


@dataclass(frozen=True)
class CandidateBox:
    box: Box
    score: float
    source: str


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1.0, float(area_a + area_b - intersection))


POLICY_COMPONENTS: dict[str, tuple[str, ...]] = {
    # App approximation: three live sources + retained all_raw RF filter.
    RUNTIME_FACE_DETECTOR_POLICY_ID: ("rfdetr", "yolo", "scrfd"),
    "fusion_rfdetr_yolo11s_scrfd10g": ("rfdetr", "yolo", "scrfd"),
    "fusion_rfdetr_scrfd10g": ("rfdetr", "scrfd"),
    "fixed_fusion_yolo11s1280_scrfd10g": ("yolo", "scrfd"),
    "yolo11s_face": ("yolo",),
}

HARDENED_RF_POLICIES: frozenset[str] = frozenset({RUNTIME_FACE_DETECTOR_POLICY_ID})


def _checkpoint_path_for(component: str) -> Path:
    if component == "rfdetr":
        return resolve_rfdetr_checkpoint()
    if component == "yolo":
        return YOLO11_FACE_CHECKPOINT
    if component == "scrfd":
        return SCRFD_CHECKPOINT
    raise KeyError(component)


def validate_thesis_face_detector(
    policy_id: str = RUNTIME_FACE_DETECTOR_POLICY_ID,
) -> dict[str, Any]:
    """Fail clearly when the selected detector tier cannot run as specified."""
    import importlib.util
    import torch

    if policy_id not in POLICY_COMPONENTS:
        raise ValueError(f"Unknown face-detector policy: {policy_id}")
    components = POLICY_COMPONENTS[policy_id]
    paths = {name: _checkpoint_path_for(name) for name in components}
    missing_files = [
        str(paths[name])
        for name in components
        if _resolve_existing(paths[name]) is None
    ]
    package_for = {
        "rfdetr": "rfdetr",
        "yolo": "ultralytics",
        "scrfd": "insightface",
    }
    missing_packages = [
        package_for[name]
        for name in components
        if importlib.util.find_spec(package_for[name]) is None
    ]
    if "scrfd" in components and importlib.util.find_spec("onnxruntime") is None:
        missing_packages.append("onnxruntime")
    if missing_files:
        raise RuntimeError("Required detector checkpoint(s) missing: " + ", ".join(missing_files))
    if missing_packages:
        raise RuntimeError("Required detector package(s) missing: " + ", ".join(sorted(set(missing_packages))))
    requires_cuda = any(name in components for name in ("rfdetr", "scrfd"))
    if requires_cuda and not torch.cuda.is_available():
        raise RuntimeError(f"The selected detector policy `{policy_id}` requires CUDA.")

    providers: list[str] = []
    if "scrfd" in components:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        # Prefer CUDA, but allow CPU ORT so preflight does not hard-fail on CPU-only.
        if not providers:
            raise RuntimeError("SCRFD requires ONNX Runtime with at least one execution provider.")
    rfdetr_path = _checkpoint_path_for("rfdetr") if "rfdetr" in components else None
    yolo_path = _checkpoint_path_for("yolo") if "yolo" in components else None
    scrfd_path = _checkpoint_path_for("scrfd") if "scrfd" in components else None
    hardened = policy_id in HARDENED_RF_POLICIES
    hardened_filter = ErrorHardenedRFFilter() if hardened else None
    return {
        "policy_id": policy_id,
        "components": list(components),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "rfdetr_checkpoint": str(rfdetr_path) if rfdetr_path is not None else None,
        "yolo_checkpoint": str(yolo_path) if yolo_path is not None else None,
        "scrfd_checkpoint": str(scrfd_path) if scrfd_path is not None else None,
        "onnx_providers": providers,
        "hardened_rf": hardened,
        "hardened_rf_model_available": bool(hardened_filter and hardened_filter.available),
        "scientific_primary": False,
        "runtime_approximation": policy_id == RUNTIME_FACE_DETECTOR_POLICY_ID,
        "candidate_sources": list(RUNTIME_SOURCE_NAMES) if policy_id == RUNTIME_FACE_DETECTOR_POLICY_ID else [],
        "scientific_score_assigned": False,
        "evidence": runtime_policy_evidence_text() if policy_id == RUNTIME_FACE_DETECTOR_POLICY_ID else "",
    }


def resolve_runnable_face_policy(
    preferred_policy_id: str = RUNTIME_FACE_DETECTOR_POLICY_ID,
) -> dict[str, Any]:
    """Validate preferred policy; otherwise step down until a runnable tier is found."""
    tried: list[str] = []
    errors: list[str] = []
    order: list[str] = []
    if preferred_policy_id in POLICY_COMPONENTS:
        order.append(preferred_policy_id)
    for pid in FACE_POLICY_FALLBACK_ORDER:
        if pid not in order:
            order.append(pid)
    for policy_id in order:
        tried.append(policy_id)
        try:
            runtime = validate_thesis_face_detector(policy_id)
            runtime["requested_policy_id"] = preferred_policy_id
            runtime["fallback_applied"] = policy_id != preferred_policy_id
            runtime["tried_policies"] = tried
            if runtime["fallback_applied"]:
                runtime["fallback_reason"] = (
                    f"Preferred `{preferred_policy_id}` unavailable; using `{policy_id}`."
                )
            return runtime
        except Exception as exc:  # noqa: BLE001 - collect and try next tier
            errors.append(f"{policy_id}: {exc}")
    raise RuntimeError(
        "No runnable face-detector policy. Tried: "
        + "; ".join(errors)
        + ". Install YOLO face weights under data/models/face_detection_candidates/ "
        "and optional RF-DETR/SCRFD for accelerated tiers."
    )


class ThesisFaceDetector:
    """Lazy runtime for a hardware-selected, evidence-backed detector policy."""

    def __init__(self, policy_id: str = RUNTIME_FACE_DETECTOR_POLICY_ID) -> None:
        self.policy_id = policy_id
        self.components = POLICY_COMPONENTS.get(policy_id, ())
        self._rfdetr = None
        self._yolo = None
        self._scrfd = None
        self.runtime = validate_thesis_face_detector(policy_id)
        self._device = str(self.runtime["device"])
        self._hardened_filter = (
            ErrorHardenedRFFilter() if policy_id in HARDENED_RF_POLICIES else None
        )

    def _get_rfdetr(self):
        if self._rfdetr is None:
            from rfdetr import RFDETRMedium

            self._rfdetr = RFDETRMedium(
                device="cuda",
                pretrain_weights=str(resolve_rfdetr_checkpoint()),
            )
        return self._rfdetr

    def _get_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO

            self._yolo = YOLO(str(YOLO11_FACE_CHECKPOINT))
        return self._yolo

    def _get_scrfd(self):
        if self._scrfd is None:
            from src.detection.scrfd_detector import SCRFDDetector

            self._scrfd = SCRFDDetector(
                model_path=str(SCRFD_CHECKPOINT),
                confidence_threshold=SCRFD_THRESHOLD,
                input_size=SCRFD_INPUT_SIZE,
                providers=["CUDAExecutionProvider"],
            )
        return self._scrfd

    def _rfdetr_boxes(self, image: Image.Image) -> list[CandidateBox]:
        detections = self._get_rfdetr().predict(
            image.convert("RGB"), threshold=RFDETR_THRESHOLD, include_source_image=False
        )
        output: list[CandidateBox] = []
        for box, score, class_id in zip(
            getattr(detections, "xyxy", []),
            getattr(detections, "confidence", []),
            getattr(detections, "class_id", []),
            strict=False,
        ):
            if int(class_id) != 0:
                continue
            coords = tuple(int(round(float(value))) for value in box)
            output.append(CandidateBox(coords, float(score), "rfdetr_medium_face_030"))
        return output

    def _yolo_boxes(self, image: Image.Image) -> list[CandidateBox]:
        bgr_image = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        result = self._get_yolo().predict(
            bgr_image,
            conf=YOLO_THRESHOLD,
            iou=0.50,
            imgsz=YOLO_IMAGE_SIZE,
            device="0" if self._device == "cuda" else "cpu",
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        output: list[CandidateBox] = []
        for box, score, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            strict=False,
        ):
            if int(class_id) != 0:
                continue
            coords = tuple(int(round(float(value))) for value in box)
            output.append(CandidateBox(coords, float(score), "yolo11s_widerface_1280"))
        return output

    def _scrfd_boxes(self, image: Image.Image) -> list[CandidateBox]:
        result = self._get_scrfd().detect(image.convert("RGB"))
        return [
            CandidateBox(det.box, float(det.confidence), "scrfd_10g_current_640")
            for det in result.detections
        ]

    def _release(self, attribute: str) -> None:
        """Release one detector before the next detector family is loaded."""
        setattr(self, attribute, None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def _fuse(candidates: list[CandidateBox]) -> tuple[list[ScoredBox], list[list[str]]]:
        pending = sorted(candidates, key=lambda item: item.score, reverse=True)
        fused: list[ScoredBox] = []
        provenance: list[list[str]] = []
        while pending:
            seed = pending.pop(0)
            cluster = [seed]
            remaining: list[CandidateBox] = []
            for candidate in pending:
                if _iou(seed.box, candidate.box) >= FUSION_IOU:
                    cluster.append(candidate)
                else:
                    remaining.append(candidate)
            pending = remaining

            sources = sorted({item.source for item in cluster})
            weights = [max(item.score, 1e-6) for item in cluster]
            total = sum(weights)
            coords = tuple(
                int(round(sum(item.box[index] * weight for item, weight in zip(cluster, weights, strict=False)) / total))
                for index in range(4)
            )
            score = max(item.score for item in cluster)
            score += FUSION_AGREEMENT_BONUS * max(0, len(sources) - 1)
            if len(sources) == 1:
                score *= FUSION_SINGLE_DETECTOR_PENALTY
            score = min(1.0, float(score))
            if score >= FUSION_MIN_SCORE:
                fused.append((*coords, score))
                provenance.append(sources)
        return fused, provenance

    def detect(self, image: Image.Image) -> tuple[list[ScoredBox], dict[str, Any]]:
        rgb = image.convert("RGB")
        candidates: list[CandidateBox] = []
        if "rfdetr" in self.components:
            candidates.extend(self._rfdetr_boxes(rgb))
            self._release("_rfdetr")
        if "yolo" in self.components:
            candidates.extend(self._yolo_boxes(rgb))
            self._release("_yolo")
        if "scrfd" in self.components:
            candidates.extend(self._scrfd_boxes(rgb))
            self._release("_scrfd")
        return self._result(candidates, image_size=rgb.size)

    def _result(
        self,
        candidates: list[CandidateBox],
        *,
        image_size: tuple[int, int] | None = None,
    ) -> tuple[list[ScoredBox], dict[str, Any]]:
        source_counts: dict[str, int] = {}
        for item in candidates:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1

        if self._hardened_filter is not None:
            width, height = image_size or (3840, 2160)
            detector_candidates = [
                DetectorCandidate(item.box, item.score, item.source) for item in candidates
            ]
            boxes, hardened_meta = self._hardened_filter.apply(
                detector_candidates,
                image_width=float(width),
                image_height=float(height),
                iou_threshold=CLUSTER_IOU,
            )
            return boxes, {
                "policy_id": self.policy_id,
                "candidate_counts": source_counts,
                "candidate_count": len(candidates),
                "fused_count": len(boxes),
                "fused_sources": [],
                "hardened_rf": hardened_meta,
                "evidence": runtime_policy_evidence_text(),
                "thresholds": {
                    "rfdetr": RFDETR_THRESHOLD,
                    "yolo": YOLO_THRESHOLD,
                    "scrfd": SCRFD_THRESHOLD,
                    "cluster_iou": CLUSTER_IOU,
                    "hardened_threshold": hardened_meta.get("threshold"),
                    "live_candidate_source_bank": list(RUNTIME_SOURCE_NAMES),
                    "filter_feature_source_layout": list(ALL_RAW_SOURCE_NAMES),
                    "scientific_score_assigned": False,
                },
            }

        boxes, provenance = self._fuse(candidates)
        return boxes, {
            "policy_id": self.policy_id,
            "candidate_counts": source_counts,
            "candidate_count": len(candidates),
            "fused_count": len(boxes),
            "fused_sources": provenance,
            "thresholds": {
                "rfdetr": RFDETR_THRESHOLD,
                "yolo": YOLO_THRESHOLD,
                "scrfd": SCRFD_THRESHOLD,
                "fusion_iou": FUSION_IOU,
                "fusion_min_score": FUSION_MIN_SCORE,
                "agreement_bonus": FUSION_AGREEMENT_BONUS,
                "single_detector_penalty": FUSION_SINGLE_DETECTOR_PENALTY,
            },
        }

    def detect_paths(
        self,
        items: list[tuple[str, Path]],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, tuple[list[ScoredBox], dict[str, Any]]]:
        """Run each detector over the batch, unloading it before the next stage.

        Optional ``progress_callback(fraction, message)`` reports 0.0–1.0 progress
        while each component processes images (RF-DETR / YOLO / SCRFD).
        """

        def emit(fraction: float, message: str) -> None:
            if progress_callback is not None:
                progress_callback(max(0.0, min(1.0, fraction)), message)

        candidates: dict[str, list[CandidateBox]] = {image_id: [] for image_id, _ in items}
        active_components = [name for name in ("rfdetr", "yolo", "scrfd") if name in self.components]
        n_items = max(len(items), 1)
        n_stages = max(len(active_components), 1)
        total_units = max(n_stages * n_items, 1)
        completed_units = 0

        def report(stage_label: str, image_index: int, image_id: str) -> None:
            nonlocal completed_units
            completed_units += 1
            emit(
                completed_units / total_units,
                f"Face detection ({stage_label}): image {image_index} of {len(items)} · {image_id}",
            )

        if "rfdetr" in self.components:
            emit(completed_units / total_units, "Face detection (RF-DETR): loading model")
            rfdetr = self._get_rfdetr()
            for image_index, (image_id, path) in enumerate(items, start=1):
                with Image.open(path) as loaded:
                    image = loaded.convert("RGB")
                detections = rfdetr.predict(
                    image, threshold=RFDETR_THRESHOLD, include_source_image=False
                )
                for box, score, class_id in zip(
                    getattr(detections, "xyxy", []),
                    getattr(detections, "confidence", []),
                    getattr(detections, "class_id", []),
                    strict=False,
                ):
                    if int(class_id) == 0:
                        coords = tuple(int(round(float(value))) for value in box)
                        candidates[image_id].append(
                            CandidateBox(coords, float(score), "rfdetr_medium_face_030")
                        )
                report("RF-DETR", image_index, image_id)
            self._release("_rfdetr")

        if "yolo" in self.components:
            emit(completed_units / total_units, "Face detection (YOLO): loading model")
            yolo = self._get_yolo()
            for image_index, (image_id, path) in enumerate(items, start=1):
                with Image.open(path) as loaded:
                    image = loaded.convert("RGB")
                bgr_image = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
                result = yolo.predict(
                    bgr_image,
                    conf=YOLO_THRESHOLD,
                    iou=0.50,
                    imgsz=YOLO_IMAGE_SIZE,
                    device="0" if self._device == "cuda" else "cpu",
                    verbose=False,
                )[0]
                if result.boxes is not None:
                    for box, score, class_id in zip(
                        result.boxes.xyxy.cpu().tolist(),
                        result.boxes.conf.cpu().tolist(),
                        result.boxes.cls.cpu().tolist(),
                        strict=False,
                    ):
                        if int(class_id) == 0:
                            coords = tuple(int(round(float(value))) for value in box)
                            candidates[image_id].append(
                                CandidateBox(coords, float(score), "yolo11s_widerface_1280")
                            )
                report("YOLO", image_index, image_id)
            self._release("_yolo")

        if "scrfd" in self.components:
            emit(completed_units / total_units, "Face detection (SCRFD): loading model")
            scrfd = self._get_scrfd()
            for image_index, (image_id, path) in enumerate(items, start=1):
                with Image.open(path) as loaded:
                    image = loaded.convert("RGB")
                result = scrfd.detect(image)
                candidates[image_id].extend(
                    CandidateBox(det.box, float(det.confidence), "scrfd_10g_current_640")
                    for det in result.detections
                )
                report("SCRFD", image_index, image_id)
            self._release("_scrfd")

        emit(1.0, "Face detection complete · fusing detector outputs")
        results: dict[str, tuple[list[ScoredBox], dict[str, Any]]] = {}
        for image_id, path in items:
            with Image.open(path) as loaded:
                size = loaded.size
            results[image_id] = self._result(candidates[image_id], image_size=size)
        return results
