"""Precision-gated screen and text localisation for the application pipeline."""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from PIL import Image

from src.detection.multimodal_stack import MultimodalDetections


ROOT = Path(__file__).resolve().parents[2]
SCREEN_MODEL = ROOT / "app" / "models" / "multimodal_screen_yolo11s.pt"
SCREEN_CONFIDENCE = 0.43

ScoredBox = tuple[int, int, int, int, float]


class MultimodalPrecisionStack:
    """Localise screens precisely and retain only OCR-recognised text regions."""

    def __init__(
        self,
        *,
        image_size: int = 1280,
        text_canvas_size: int = 2560,
        text_confidence: float = 0.30,
        text_use_gpu: bool = True,
    ) -> None:
        self.image_size = int(image_size)
        self.text_canvas_size = int(text_canvas_size)
        self.text_confidence = float(text_confidence)
        self.text_use_gpu = bool(text_use_gpu and self._cuda_available())
        self.policy_id = f"precision_screen_{self.image_size}_recognised_text_{self.text_canvas_size}"
        self._screen_model: Any | None = None
        self._text_reader: Any | None = None
        self.warnings: list[str] = []
        self.runtime: dict[str, Any] = {
            "policy_id": self.policy_id,
            "screen_seconds": 0.0,
            "text_seconds": 0.0,
            "images": 0,
            "text_gpu": self.text_use_gpu,
        }

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _get_screen_model(self):
        if self._screen_model is not None:
            return self._screen_model
        if not SCREEN_MODEL.exists():
            raise FileNotFoundError(f"Screen model is missing: {SCREEN_MODEL}")
        from ultralytics import YOLO

        self._screen_model = YOLO(str(SCREEN_MODEL))
        return self._screen_model

    def _get_text_reader(self):
        if self._text_reader is not None:
            return self._text_reader
        import easyocr

        self._text_reader = easyocr.Reader(["en"], gpu=self.text_use_gpu, verbose=False)
        return self._text_reader

    @staticmethod
    def _iou(a: ScoredBox, b: ScoredBox) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection <= 0:
            return 0.0
        area_a = max(1, a[2] - a[0]) * max(1, a[3] - a[1])
        area_b = max(1, b[2] - b[0]) * max(1, b[3] - b[1])
        return intersection / float(area_a + area_b - intersection)

    @classmethod
    def _nms(cls, boxes: list[ScoredBox], threshold: float = 0.45) -> list[ScoredBox]:
        pending = sorted(boxes, key=lambda box: box[4], reverse=True)
        kept: list[ScoredBox] = []
        while pending:
            best = pending.pop(0)
            kept.append(best)
            pending = [box for box in pending if cls._iou(best, box) < threshold]
        return kept

    @staticmethod
    def _intersects_screen(text: ScoredBox, screen: ScoredBox) -> bool:
        corners = (
            (text[0], text[1]),
            (text[2], text[1]),
            (text[0], text[3]),
            (text[2], text[3]),
        )
        return any(screen[0] <= x <= screen[2] and screen[1] <= y <= screen[3] for x, y in corners)

    def _detect_screens(self, image: Image.Image) -> list[ScoredBox]:
        model = self._get_screen_model()
        device: str | int = 0 if self._cuda_available() else "cpu"
        # Ultralytics treats NumPy image inputs as BGR. PIL supplies RGB, so an
        # explicit conversion is required to match file-based evaluation.
        bgr_image = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        results = model.predict(
            bgr_image,
            conf=SCREEN_CONFIDENCE,
            iou=0.60,
            imgsz=self.image_size,
            device=device,
            half=device != "cpu",
            verbose=False,
        )
        boxes: list[ScoredBox] = []
        if results and results[0].boxes is not None:
            for prediction in results[0].boxes:
                # The reviewed-data model is screen-only. Text is handled by OCR
                # recognition because proposal-only text boxes were too imprecise.
                if int(prediction.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = map(int, prediction.xyxy[0].tolist())
                confidence = float(prediction.conf[0])
                boxes.append((x1, y1, x2, y2, confidence))
        return self._nms(boxes)

    def _detect_text(self, image: Image.Image) -> list[ScoredBox]:
        reader = self._get_text_reader()
        rows = reader.readtext(
            np.asarray(image.convert("RGB")),
            detail=1,
            paragraph=False,
            min_size=8,
            text_threshold=0.60,
            low_text=0.30,
            link_threshold=0.30,
            canvas_size=self.text_canvas_size,
            mag_ratio=1.0,
            batch_size=8 if self.text_use_gpu else 1,
            workers=0,
        )
        width, height = image.size
        image_area = max(1, width * height)
        boxes: list[ScoredBox] = []
        for polygon, recognised_text, confidence in rows:
            clean_text = re.sub(r"[^A-Za-z0-9]", "", str(recognised_text))
            score = float(confidence)
            if len(clean_text) < 3 or score < self.text_confidence:
                continue
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
            x2, y2 = min(width, int(max(xs))), min(height, int(max(ys)))
            area_fraction = max(0, x2 - x1) * max(0, y2 - y1) / image_area
            if x2 - x1 < 8 or y2 - y1 < 6 or area_fraction > 0.08:
                continue
            boxes.append((x1, y1, x2, y2, score))
        return self._nms(boxes, threshold=0.35)

    def detect(self, image: Image.Image) -> MultimodalDetections:
        warnings_before = len(self.warnings)
        screen_start = time.perf_counter()
        screens = self._detect_screens(image)
        self.runtime["screen_seconds"] += time.perf_counter() - screen_start

        text_start = time.perf_counter()
        texts = self._detect_text(image)
        self.runtime["text_seconds"] += time.perf_counter() - text_start
        texts = [
            text
            for text in texts
            if not any(self._intersects_screen(text, screen) for screen in screens)
        ]
        self.runtime["images"] += 1
        return MultimodalDetections(
            screens=screens,
            texts=texts,
            screen_sources=["reviewed_screen_detector"] * len(screens),
            screen_policy=f"reviewed_screen_detector_{self.image_size}_conf043",
            text_policy=f"easyocr_recognised_text_{self.text_canvas_size}_conf{self.text_confidence:.2f}",
            warnings=self.warnings[warnings_before:],
        )
