"""Scene-text detector with pluggable backends."""

from __future__ import annotations

from typing import Any
import math

import cv2
import numpy as np
from PIL import Image

from src.detection.base_detector import BaseDetector, DetectionResult


class TextDetector(BaseDetector):
    """Detect text regions using a configurable backend."""

    detector_name = "text_detector"

    def __init__(
        self,
        backend: str = "easyocr",
        languages: list[str] | None = None,
        device: str = "cpu",
        easyocr_multiscale_scales: list[float] | None = None,
        min_size: int = 10,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        east_model_path: str = "data/models/frozen_east_text_detection.pb",
        east_input_size: int = 1280,
        east_score_threshold: float = 0.5,
        east_nms_threshold: float = 0.4,
        doctr_det_arch: str = "db_resnet50",
        doctr_reco_arch: str = "crnn_vgg16_bn",
    ) -> None:
        if backend not in {"easyocr", "east", "doctr"}:
            raise ValueError("backend must be 'easyocr', 'east', or 'doctr'")
        self.backend = backend
        self.languages = languages or ["en"]
        self.device = device
        self.easyocr_multiscale_scales = easyocr_multiscale_scales or [1.0]
        self.min_size = min_size
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.canvas_size = int(canvas_size)
        self.mag_ratio = float(mag_ratio)
        self.east_model_path = east_model_path
        self.east_input_size = east_input_size
        self.east_score_threshold = east_score_threshold
        self.east_nms_threshold = east_nms_threshold
        self.doctr_det_arch = doctr_det_arch
        self.doctr_reco_arch = doctr_reco_arch

        if backend == "easyocr":
            try:
                import easyocr
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "TextDetector with backend='easyocr' requires the 'easyocr' package. "
                    "Install it with `pip install easyocr`."
                ) from exc

            gpu_enabled = device == "cuda"
            self._reader = easyocr.Reader(self.languages, gpu=gpu_enabled)
            self.detector_name = "craft_easyocr"
        elif backend == "east":
            model_path = str(east_model_path)
            self._east_net = cv2.dnn.readNet(model_path)
            self.detector_name = "east_opencv"
        else:
            try:
                from doctr.models import ocr_predictor
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "TextDetector with backend='doctr' requires the 'python-doctr' package. "
                    "Install it with `pip install python-doctr`."
                ) from exc

            self._doctr_predictor = ocr_predictor(
                det_arch=self.doctr_det_arch,
                reco_arch=self.doctr_reco_arch,
                pretrained=True,
            )
            if self.device == "cuda":
                self._doctr_predictor = self._doctr_predictor.to("cuda")
            self.detector_name = "doctr_dbnet"

    @staticmethod
    def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter_area / float(area_a + area_b - inter_area)

    def _merge_boxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        metadata_rows: list[dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> tuple[list[tuple[int, int, int, int]], list[dict[str, Any]]]:
        merged_boxes: list[tuple[int, int, int, int]] = []
        merged_metadata: list[dict[str, Any]] = []
        for box, metadata in zip(boxes, metadata_rows, strict=True):
            duplicate = False
            for existing in merged_boxes:
                if self._box_iou(box, existing) >= iou_threshold:
                    duplicate = True
                    break
            if duplicate:
                continue
            merged_boxes.append(box)
            merged_metadata.append(metadata)
        return merged_boxes, merged_metadata

    def _detect_easyocr(self, image: Image.Image) -> DetectionResult:
        boxes: list[tuple[int, int, int, int]] = []
        metadata_rows: list[dict[str, Any]] = []
        original_width, original_height = image.size

        for scale in self.easyocr_multiscale_scales:
            if scale <= 0:
                continue
            if scale == 1.0:
                scaled_image = image.convert("RGB")
            else:
                scaled_image = image.convert("RGB").resize(
                    (
                        max(1, int(round(original_width * scale))),
                        max(1, int(round(original_height * scale))),
                    ),
                    Image.Resampling.LANCZOS,
                )
            image_np = np.array(scaled_image)
            horizontal_list, free_list = self._reader.detect(
                image_np,
                min_size=self.min_size,
                text_threshold=self.text_threshold,
                low_text=self.low_text,
                link_threshold=self.link_threshold,
                canvas_size=self.canvas_size,
                mag_ratio=self.mag_ratio,
            )

            for box in horizontal_list[0]:
                x_min, x_max, y_min, y_max = box
                boxes.append(
                    (
                        int(round(x_min / scale)),
                        int(round(y_min / scale)),
                        int(round(x_max / scale)),
                        int(round(y_max / scale)),
                    )
                )
                metadata_rows.append({"geometry": "horizontal", "scale": scale})

            for poly in free_list[0]:
                xs = [point[0] for point in poly]
                ys = [point[1] for point in poly]
                scaled_poly = [[float(point[0]) / scale, float(point[1]) / scale] for point in poly]
                boxes.append(
                    (
                        int(round(min(xs) / scale)),
                        int(round(min(ys) / scale)),
                        int(round(max(xs) / scale)),
                        int(round(max(ys) / scale)),
                    )
                )
                metadata_rows.append({"geometry": "polygon", "polygon": scaled_poly, "scale": scale})

        boxes, metadata_rows = self._merge_boxes(boxes, metadata_rows)

        detections = self.normalise_detections(
            image=image,
            boxes=boxes,
            per_detection_metadata=metadata_rows,
        )
        return DetectionResult(
            detections=detections,
            metadata={
                "detector_name": self.detector_name,
                "backend": self.backend,
                "languages": self.languages,
                "device": self.device,
                "easyocr_multiscale_scales": self.easyocr_multiscale_scales,
                "canvas_size": self.canvas_size,
                "mag_ratio": self.mag_ratio,
            },
        )

    def _decode_east(
        self,
        scores: np.ndarray,
        geometry: np.ndarray,
        score_threshold: float,
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        height, width = scores.shape[2:4]

        for y in range(height):
            scores_data = scores[0, 0, y]
            x0_data = geometry[0, 0, y]
            x1_data = geometry[0, 1, y]
            x2_data = geometry[0, 2, y]
            x3_data = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            for x in range(width):
                score = scores_data[x]
                if score < score_threshold:
                    continue

                offset_x = x * 4.0
                offset_y = y * 4.0
                angle = angles_data[x]
                cos = math.cos(angle)
                sin = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]
                end_x = int(offset_x + (cos * x1_data[x]) + (sin * x2_data[x]))
                end_y = int(offset_y - (sin * x1_data[x]) + (cos * x2_data[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                boxes.append((start_x, start_y, end_x, end_y))
                confidences.append(float(score))

        return boxes, confidences

    def _detect_east(self, image: Image.Image) -> DetectionResult:
        rgb = image.convert("RGB")
        orig_w, orig_h = rgb.size
        new_w = self.east_input_size
        new_h = self.east_input_size
        r_w = orig_w / float(new_w)
        r_h = orig_h / float(new_h)

        resized = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
        image_np = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(
            image_np,
            1.0,
            (new_w, new_h),
            (123.68, 116.78, 103.94),
            swapRB=False,
            crop=False,
        )
        self._east_net.setInput(blob)
        scores, geometry = self._east_net.forward(
            [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3",
            ]
        )
        boxes, confidences = self._decode_east(
            scores=scores,
            geometry=geometry,
            score_threshold=self.east_score_threshold,
        )
        rects = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
        indices = cv2.dnn.NMSBoxes(
            bboxes=rects,
            scores=confidences,
            score_threshold=self.east_score_threshold,
            nms_threshold=self.east_nms_threshold,
        )

        final_boxes: list[tuple[int, int, int, int]] = []
        final_scores: list[float] = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                x1, y1, x2, y2 = boxes[int(idx)]
                final_boxes.append(
                    (
                        int(x1 * r_w),
                        int(y1 * r_h),
                        int(x2 * r_w),
                        int(y2 * r_h),
                    )
                )
                final_scores.append(confidences[int(idx)])

        detections = self.normalise_detections(
            image=image,
            boxes=final_boxes,
            confidences=final_scores,
            per_detection_metadata=[{"geometry": "east"} for _ in final_boxes],
        )
        return DetectionResult(
            detections=detections,
            metadata={
                "detector_name": self.detector_name,
                "backend": self.backend,
                "east_model_path": self.east_model_path,
                "device": self.device,
            },
        )

    def _detect_doctr(self, image: Image.Image) -> DetectionResult:
        image_np = np.array(image.convert("RGB"))
        document = self._doctr_predictor([image_np])

        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        metadata_rows: list[dict[str, Any]] = []
        image_width, image_height = image.size

        for block in document.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    (x_min, y_min), (x_max, y_max) = word.geometry
                    boxes.append(
                        (
                            int(float(x_min) * image_width),
                            int(float(y_min) * image_height),
                            int(float(x_max) * image_width),
                            int(float(y_max) * image_height),
                        )
                    )
                    confidences.append(float(word.confidence))
                    metadata_rows.append(
                        {
                            "geometry": "doctr_word",
                            "text": word.value,
                        }
                    )

        detections = self.normalise_detections(
            image=image,
            boxes=boxes,
            confidences=confidences,
            per_detection_metadata=metadata_rows,
        )
        return DetectionResult(
            detections=detections,
            metadata={
                "detector_name": self.detector_name,
                "backend": self.backend,
                "device": self.device,
                "doctr_det_arch": self.doctr_det_arch,
                "doctr_reco_arch": self.doctr_reco_arch,
            },
        )

    def detect(self, image: Image.Image) -> DetectionResult:
        if self.backend == "easyocr":
            return self._detect_easyocr(image)
        if self.backend == "east":
            return self._detect_east(image)
        return self._detect_doctr(image)
