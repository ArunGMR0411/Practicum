"""Heuristic quality-signal assessor for adaptive routing experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.detection.base_detector import BoundingBox


@dataclass(frozen=True)
class QualitySignals:
    """Numeric quality signals used by the routing calibration stage."""

    blur_score: float
    face_size_px: float
    occlusion_ratio: float
    webp_artifact_score: float


@dataclass(frozen=True)
class QualityAssessment:
    """One quality-assessment result with signals and supporting metadata."""

    signals: QualitySignals
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self.signals)
        payload.update(self.metadata)
        return payload


class QualityAssessor:
    """Compute lightweight routing features from an image and optional face boxes.

    The current assessor is intentionally heuristic and cheap:
    - blur is inverse Laplacian-variance sharpness on grayscale image content,
    - face size uses the dominant face box equivalent square size,
    - occlusion ratio estimates visible-skin coverage inside the dominant face crop,
    - WebP artefact score is a simple 8x8 boundary blockiness measure.
    """

    def __init__(self, block_size: int = 8) -> None:
        self.block_size = int(block_size)

    def assess(
        self,
        image: Image.Image,
        face_boxes: list[BoundingBox] | None = None,
    ) -> QualityAssessment:
        rgb_image = image.convert("RGB")
        image_np = np.array(rgb_image)
        dominant_box = self._select_dominant_box(rgb_image.size, face_boxes or [])

        blur_score = self._compute_blur_score(image_np)
        face_size_px = self._compute_face_size_px(dominant_box)
        occlusion_ratio = self._compute_occlusion_ratio(image_np, dominant_box)
        webp_artifact_score = self._compute_blockiness_score(image_np)

        metadata = {
            "image_width": int(rgb_image.width),
            "image_height": int(rgb_image.height),
            "face_box_count": len(face_boxes or []),
            "dominant_face_box": list(dominant_box) if dominant_box is not None else None,
        }
        return QualityAssessment(
            signals=QualitySignals(
                blur_score=float(blur_score),
                face_size_px=float(face_size_px),
                occlusion_ratio=float(occlusion_ratio),
                webp_artifact_score=float(webp_artifact_score),
            ),
            metadata=metadata,
        )

    def _select_dominant_box(
        self,
        image_size: tuple[int, int],
        face_boxes: list[BoundingBox],
    ) -> BoundingBox | None:
        if not face_boxes:
            return None
        width, height = image_size
        best_box: BoundingBox | None = None
        best_area = -1
        for x1, y1, x2, y2 in face_boxes:
            left = max(0, min(int(x1), width))
            top = max(0, min(int(y1), height))
            right = max(left + 1, min(int(x2), width))
            bottom = max(top + 1, min(int(y2), height))
            area = (right - left) * (bottom - top)
            if area > best_area:
                best_area = area
                best_box = (left, top, right, bottom)
        return best_box

    def _compute_blur_score(self, image_np: np.ndarray) -> float:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        return float(1.0 / max(sharpness, 1e-6))

    def _compute_face_size_px(self, dominant_box: BoundingBox | None) -> float:
        if dominant_box is None:
            return 0.0
        x1, y1, x2, y2 = dominant_box
        area = max(1, (x2 - x1) * (y2 - y1))
        return float(np.sqrt(area))

    def _compute_occlusion_ratio(
        self,
        image_np: np.ndarray,
        dominant_box: BoundingBox | None,
    ) -> float:
        if dominant_box is None:
            return 1.0

        x1, y1, x2, y2 = dominant_box
        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0:
            return 1.0

        # Skin-mask heuristic in YCrCb space.
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower, upper)
        skin_ratio = float(np.count_nonzero(skin_mask)) / float(skin_mask.size)

        # Penalise boxes clipped by image borders because they often indicate partial faces.
        image_height, image_width = image_np.shape[:2]
        touches_border = (
            x1 <= 0 or y1 <= 0 or x2 >= image_width or y2 >= image_height
        )
        border_penalty = 0.15 if touches_border else 0.0
        occlusion_ratio = 1.0 - skin_ratio + border_penalty
        return float(np.clip(occlusion_ratio, 0.0, 1.0))

    def _compute_blockiness_score(self, image_np: np.ndarray) -> float:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        block = max(2, self.block_size)

        vertical_boundary = self._boundary_difference(gray, axis=1, step=block)
        vertical_internal = self._internal_difference(gray, axis=1, step=block)
        horizontal_boundary = self._boundary_difference(gray, axis=0, step=block)
        horizontal_internal = self._internal_difference(gray, axis=0, step=block)

        score = (vertical_boundary + horizontal_boundary) - (vertical_internal + horizontal_internal)
        return float(max(0.0, score))

    def _boundary_difference(self, gray: np.ndarray, axis: int, step: int) -> float:
        size = gray.shape[axis]
        offsets = range(step, size, step)
        diffs: list[float] = []
        for idx in offsets:
            if axis == 1:
                diffs.append(float(np.mean(np.abs(gray[:, idx] - gray[:, idx - 1]))))
            else:
                diffs.append(float(np.mean(np.abs(gray[idx, :] - gray[idx - 1, :]))))
        return float(np.mean(diffs)) if diffs else 0.0

    def _internal_difference(self, gray: np.ndarray, axis: int, step: int) -> float:
        size = gray.shape[axis]
        diffs: list[float] = []
        for start in range(0, size, step):
            for offset in (1, max(1, step // 2)):
                idx = start + offset
                if idx <= 0 or idx >= size:
                    continue
                if axis == 1:
                    diffs.append(float(np.mean(np.abs(gray[:, idx] - gray[:, idx - 1]))))
                else:
                    diffs.append(float(np.mean(np.abs(gray[idx, :] - gray[idx - 1, :]))))
        return float(np.mean(diffs)) if diffs else 0.0
