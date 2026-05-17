#!/usr/bin/env python3
"""Evidence-selected multimodal screen/text stack with screen-first sequential text.

Thesis evidence (outputs/04_multimodal_privacy/01_multimodal_250_evidence):
  - Screen base: YOLO11n COCO 1280/conf=0.10 (tv/laptop/cell phone)
  - Completions: text-cluster screen hyp + strict edge-phone residual
  - Text: CRAFT 4K recall (EasyOCR detect API), after screens are handled
  - Screen-priority: strip text boxes inside screens

Operational order (research-handover practical path):
  1. Detect screens on the *original* image (YOLO + residual completions).
  2. Redact screens on a working copy (fill for moderate boxes, strong blur for huge ones).
  3. Detect text on the *screen-redacted* image (CRAFT 4K) so on-screen UI text
     does not generate a second wave of boxes.
  4. Filter text proposals (size / density) and strip any that still sit in screens.

Why not "text only after screens" without any original-image text?
  Text-cluster screen completion needs CRAFT proposals on the *original* when YOLO
  is empty (dense UI text on phones). That probe is used only to *propose screens*,
  not to redact those text boxes. Final text redaction uses post-screen CRAFT only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]

Box = tuple[int, int, int, int]
ScoredBox = tuple[int, int, int, int, float]

# CRAFT 4K recall (selected text method)
CRAFT_RECALL_4K = {
    "backend": "easyocr",
    "min_size": 6,
    "text_threshold": 0.55,
    "low_text": 0.25,
    "link_threshold": 0.25,
    "canvas_size": 3840,
    "mag_ratio": 1.0,
}

# YOLO11 screen union confidences (selected screen base)
SCREEN_CONF = 0.10
SCREEN_IOU = 0.70
SCREEN_SIZES = (1280,)

# Area gates (fraction of image). Huge uncertain screens get blur not fill.
SCREEN_FILL_MAX_AREA_FRAC = 0.22
SCREEN_MAX_AREA_FRAC = 0.55
SCREEN_MIN_AREA_FRAC = 0.0008
SCREEN_MIN_SCORE = 0.20

# Text gates after sequential CRAFT (reduce environmental FPs)
TEXT_MIN_AREA_FRAC = 0.00015
TEXT_MAX_AREA_FRAC = 0.12
TEXT_MIN_SIDE = 8
TEXT_MAX_BOXES_PER_IMAGE = 40


@dataclass
class MultimodalDetections:
    screens: list[ScoredBox] = field(default_factory=list)
    texts: list[ScoredBox] = field(default_factory=list)
    screen_sources: list[str] = field(default_factory=list)
    text_policy: str = "craft_recall_4k_after_screen_redact"
    screen_policy: str = "yolo11n_1280_conf010+text_cluster_hyp+strict_edge_phone"
    aux_text_for_screen_hyp: int = 0
    warnings: list[str] = field(default_factory=list)


class MultimodalStack:
    """Lazy-loaded evidence multimodal detector with screen-first text pass."""

    def __init__(
        self,
        device: str | None = None,
        *,
        screen_conf: float = SCREEN_CONF,
        screen_sizes: tuple[int, ...] = SCREEN_SIZES,
    ) -> None:
        self.device = device or self._default_device()
        self.screen_conf = float(screen_conf)
        self.screen_sizes = tuple(int(size) for size in screen_sizes)
        self._text_detector = None
        self._screen_model = None
        self._screen_model_path: Path | None = None
        self.warnings: list[str] = []

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _resolve_yolo11(self) -> Path | None:
        for name in ("yolo11n.pt", "yolov8n.pt"):
            path = ROOT / "data" / "models" / name
            if path.exists():
                return path
        return None

    def _get_screen_model(self):
        if self._screen_model is not None:
            return self._screen_model
        path = self._resolve_yolo11()
        if path is None:
            self.warnings.append("No YOLO11/YOLOv8 screen weights under data/models/")
            return None
        try:
            from ultralytics import YOLO

            self._screen_model = YOLO(str(path))
            self._screen_model_path = path
            return self._screen_model
        except Exception as exc:
            self.warnings.append(f"Screen YOLO load failed: {type(exc).__name__}: {exc}")
            return None

    def _get_text_detector(self):
        if self._text_detector is not None:
            return self._text_detector
        try:
            from src.detection.text_detector import TextDetector

            kwargs = dict(CRAFT_RECALL_4K)
            # Run CRAFT 4K on CPU to leave GPU memory for face and screen models.
            # Batch jobs may select a GPU by constructing TextDetector directly.
            kwargs["device"] = "cpu"
            self._text_detector = TextDetector(**kwargs)
            return self._text_detector
        except Exception as exc:
            self.warnings.append(f"CRAFT/EasyOCR text detector unavailable: {type(exc).__name__}: {exc}")
            return None

    # ------------------------------------------------------------------ boxes
    @staticmethod
    def _area(box: Box) -> float:
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    @staticmethod
    def _centre(box: Box) -> tuple[float, float]:
        return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        union = MultimodalStack._area(a) + MultimodalStack._area(b) - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _nms(cls, boxes: list[ScoredBox], iou_thr: float = 0.5) -> list[ScoredBox]:
        ordered = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep: list[ScoredBox] = []
        while ordered:
            best = ordered.pop(0)
            keep.append(best)
            ordered = [b for b in ordered if cls._iou(best[:4], b[:4]) < iou_thr]
        return keep

    @classmethod
    def _union_scored(cls, *groups: list[ScoredBox]) -> list[ScoredBox]:
        merged: list[ScoredBox] = []
        for g in groups:
            merged.extend(g)
        return cls._nms(merged, iou_thr=0.5)

    @staticmethod
    def _any_corner_inside(inner: Box, outer: Box) -> bool:
        return any(
            outer[0] <= x <= outer[2] and outer[1] <= y <= outer[3]
            for x, y in (
                (inner[0], inner[1]),
                (inner[2], inner[1]),
                (inner[0], inner[3]),
                (inner[2], inner[3]),
            )
        )

    @classmethod
    def strip_text_in_screens(cls, texts: list[ScoredBox], screens: list[ScoredBox]) -> list[ScoredBox]:
        if not screens:
            return texts
        out: list[ScoredBox] = []
        for t in texts:
            tb = t[:4]
            cx, cy = cls._centre(tb)
            if any(
                cls._any_corner_inside(tb, s[:4]) or (s[0] <= cx <= s[2] and s[1] <= cy <= s[3])
                for s in screens
            ):
                continue
            out.append(t)
        return out

    # -------------------------------------------------------------- screen det
    def detect_screens_yolo(self, image: Image.Image) -> list[ScoredBox]:
        model = self._get_screen_model()
        if model is None:
            return []
        rgb = image.convert("RGB")
        arr = np.array(rgb)
        device = "0" if self.device in {"cuda", "0"} else "cpu"
        boxes: list[ScoredBox] = []
        for imgsz in self.screen_sizes:
            try:
                results = model.predict(
                    arr,
                    conf=self.screen_conf,
                    iou=SCREEN_IOU,
                    imgsz=imgsz,
                    classes=[62, 63, 67],
                    device=device,
                    verbose=False,
                    half=(device != "cpu"),
                )
            except Exception as exc:
                self.warnings.append(f"screen YOLO imgsz={imgsz} failed: {type(exc).__name__}")
                continue
            if not results or results[0].boxes is None:
                continue
            for b in results[0].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                boxes.append((x1, y1, x2, y2, conf))
        return self._nms(boxes, iou_thr=0.5)

    def detect_text_craft(self, image: Image.Image) -> list[ScoredBox]:
        detector = self._get_text_detector()
        if detector is None:
            return []
        try:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            result = detector.detect(image.convert("RGB"))
        except Exception as exc:
            # One retry on CPU if GPU OOM
            msg = f"{type(exc).__name__}: {exc}"
            if "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower():
                self.warnings.append(f"CRAFT GPU OOM - rebuilding on CPU: {msg[:120]}")
                try:
                    from src.detection.text_detector import TextDetector

                    kwargs = dict(CRAFT_RECALL_4K)
                    kwargs["device"] = "cpu"
                    self._text_detector = TextDetector(**kwargs)
                    result = self._text_detector.detect(image.convert("RGB"))
                except Exception as exc2:
                    self.warnings.append(f"CRAFT detect failed: {type(exc2).__name__}: {exc2}")
                    return []
            else:
                self.warnings.append(f"CRAFT detect failed: {msg}")
                return []
        boxes: list[ScoredBox] = []
        for det in result.detections:
            x1, y1, x2, y2 = map(int, det.box)
            conf = float(det.confidence) if det.confidence is not None else 0.5
            boxes.append((x1, y1, x2, y2, conf))
        return boxes

    def hypothesize_screens_from_text(
        self,
        text_boxes: list[ScoredBox],
        width: int,
        height: int,
        *,
        min_count: int = 5,
        margin_ratio: float = 0.18,
        link_frac: float = 0.09,
        min_area_frac: float = 0.006,
        max_area_frac: float = 0.45,
    ) -> list[ScoredBox]:
        plain = [b[:4] for b in text_boxes]
        link = link_frac * max(width, height)
        remaining = list(plain)
        clusters: list[list[Box]] = []
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            changed = True
            while changed:
                changed = False
                keep: list[Box] = []
                for box in remaining:
                    cx, cy = self._centre(box)
                    if any(
                        abs(cx - self._centre(m)[0]) <= link and abs(cy - self._centre(m)[1]) <= link
                        for m in cluster
                    ):
                        cluster.append(box)
                        changed = True
                    else:
                        keep.append(box)
                remaining = keep
            clusters.append(cluster)

        hyps: list[ScoredBox] = []
        for cluster in clusters:
            if len(cluster) < min_count:
                continue
            x1 = min(b[0] for b in cluster)
            y1 = min(b[1] for b in cluster)
            x2 = max(b[2] for b in cluster)
            y2 = max(b[3] for b in cluster)
            mx = int((x2 - x1) * margin_ratio)
            my = int((y2 - y1) * margin_ratio)
            box = (
                max(0, x1 - mx),
                max(0, y1 - my),
                min(width, x2 + mx),
                min(height, y2 + my),
            )
            frac = self._area(box) / float(width * height)
            if frac < min_area_frac or frac > max_area_frac:
                continue
            hyps.append((box[0], box[1], box[2], box[3], 0.35))
        return hyps

    def edge_phone_proposals(self, image: Image.Image, *, top_k: int = 1) -> list[ScoredBox]:
        """Strict edge-phone residual (promoted in residual hard-miss campaign)."""
        try:
            import cv2
        except Exception:
            return []
        rgb = image.convert("RGB")
        w, h = rgb.size
        arr = np.array(rgb)
        y0_frac = 0.55
        min_area_frac, max_area_frac = 0.015, 0.05
        min_edge_density = 0.08
        min_center_y = 0.82
        aspect_range = (1.3, 1.9)
        min_score = 0.15
        min_intensity_std = 60.0

        y0 = int(y0_frac * h)
        crop = arr[y0:, :, :]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        full_gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        scored: list[tuple[float, Box]] = []
        ar_lo, ar_hi = aspect_range
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 40 or bh < 40:
                continue
            box = (x, y + y0, x + bw, y + y0 + bh)
            frac = (bw * bh) / float(w * h)
            if frac < min_area_frac or frac > max_area_frac:
                continue
            aspect = bw / max(1, bh)
            if not (ar_lo <= aspect <= ar_hi):
                continue
            cy = (box[1] + box[3]) / 2.0 / h
            if cy < min_center_y:
                continue
            region = edges[y : y + bh, x : x + bw]
            dens = float(np.mean(region > 0)) if region.size else 0.0
            if dens < min_edge_density:
                continue
            patch = full_gray[box[1] : box[3], box[0] : box[2]]
            if patch.size == 0 or float(np.std(patch)) < min_intensity_std:
                continue
            score = dens * (1.0 - abs(aspect - 1.55) / 1.55) * cy
            if score < min_score:
                continue
            scored.append((score, box))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(b[0], b[1], b[2], b[3], float(s)) for s, b in scored[:top_k]]

    def filter_screens(self, screens: list[ScoredBox], width: int, height: int) -> list[ScoredBox]:
        area_img = float(width * height)
        kept: list[ScoredBox] = []
        for b in screens:
            frac = self._area(b[:4]) / area_img
            if frac < SCREEN_MIN_AREA_FRAC or frac > SCREEN_MAX_AREA_FRAC:
                continue
            if b[4] < SCREEN_MIN_SCORE and frac > 0.02:
                # keep low-score residual hyps only if compact
                if b[4] < 0.12:
                    continue
            kept.append(b)
        return self._nms(kept, iou_thr=0.5)

    def filter_texts(self, texts: list[ScoredBox], width: int, height: int) -> list[ScoredBox]:
        area_img = float(width * height)
        kept: list[ScoredBox] = []
        for b in texts:
            x1, y1, x2, y2, score = b
            bw, bh = x2 - x1, y2 - y1
            if bw < TEXT_MIN_SIDE or bh < TEXT_MIN_SIDE:
                continue
            frac = self._area(b[:4]) / area_img
            if frac < TEXT_MIN_AREA_FRAC or frac > TEXT_MAX_AREA_FRAC:
                continue
            # very large "text" boxes are almost always environmental FPs
            if max(bw, bh) > 0.55 * max(width, height) and frac > 0.04:
                continue
            kept.append(b)
        kept = self._nms(kept, iou_thr=0.4)
        # keep highest-score boxes if still too many
        kept = sorted(kept, key=lambda t: t[4], reverse=True)[:TEXT_MAX_BOXES_PER_IMAGE]
        return kept

    def redact_screens_for_text_pass(self, image: Image.Image, screens: list[ScoredBox]) -> Image.Image:
        """Working copy used only to suppress on-screen text before CRAFT."""
        out = image.convert("RGB").copy()
        draw = ImageDraw.Draw(out)
        for b in screens:
            draw.rectangle(b[:4], fill=(0, 0, 0))
        return out

    def detect(self, image: Image.Image) -> MultimodalDetections:
        """Full sequential multimodal detection on one RGB image."""
        self.warnings = []
        rgb = image.convert("RGB")
        w, h = rgb.size
        sources: list[str] = []

        screens = self.detect_screens_yolo(rgb)
        if screens:
            sources.append(f"yolo11_union({len(screens)})")

        aux_n = 0
        if not screens:
            # Use CRAFT regions to form screen hypotheses.
            aux = self.detect_text_craft(rgb)
            aux_n = len(aux)
            hyps = self.hypothesize_screens_from_text(aux, w, h)
            if hyps:
                screens = self._union_scored(screens, hyps)
                sources.append(f"text_cluster_hyp({len(hyps)})")

        if not screens:
            edge = self.edge_phone_proposals(rgb, top_k=1)
            if edge:
                screens = self._union_scored(screens, edge)
                sources.append(f"strict_edge_phone({len(edge)})")

        screens = self.filter_screens(screens, w, h)

        # Screen-first: redact screens, then CRAFT for residual off-screen text.
        work = self.redact_screens_for_text_pass(rgb, screens) if screens else rgb
        texts = self.detect_text_craft(work)
        texts = self.filter_texts(texts, w, h)
        texts = self.strip_text_in_screens(texts, screens)

        return MultimodalDetections(
            screens=screens,
            texts=texts,
            screen_sources=sources or ["none"],
            aux_text_for_screen_hyp=aux_n,
            warnings=list(self.warnings),
        )


def apply_screen_operator(
    image: Image.Image,
    screens: list[ScoredBox],
    *,
    mode: str = "fill",
    fill_max_area_frac: float = SCREEN_FILL_MAX_AREA_FRAC,
    blur_radius: float = 22.0,
    pixel_size: int = 18,
) -> tuple[Image.Image, list[str]]:
    """Apply the evidence-selected fill, blur, pixelate, or adaptive operator."""
    out = image.convert("RGB").copy()
    w, h = out.size
    area_img = float(w * h)
    ops: list[str] = []
    for b in screens:
        x1, y1, x2, y2, score = b
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        frac = ((x2 - x1) * (y2 - y1)) / area_img
        selected = mode
        if mode == "adaptive":
            selected = "fill" if frac <= fill_max_area_frac else "blur"
        if selected == "fill":
            fill = Image.new("RGB", (x2 - x1, y2 - y1), (0, 0, 0))
            out.paste(fill, (x1, y1))
            ops.append(f"screen_fill@{score:.2f}/a{frac:.3f}")
        elif selected == "pixelate":
            region = out.crop((x1, y1, x2, y2))
            rw, rh = region.size
            small = region.resize(
                (max(1, rw // pixel_size), max(1, rh // pixel_size)),
                Image.Resampling.BILINEAR,
            )
            out.paste(small.resize((rw, rh), Image.Resampling.NEAREST), (x1, y1))
            ops.append(f"screen_pixelate@{score:.2f}/a{frac:.3f}")
        elif selected == "blur":
            region = out.crop((x1, y1, x2, y2))
            out.paste(region.filter(ImageFilter.GaussianBlur(radius=blur_radius)), (x1, y1))
            ops.append(f"screen_blur@{score:.2f}/a{frac:.3f}")
        else:
            raise ValueError(f"Unsupported screen operator: {mode}")
    return out, ops


def apply_text_operator(
    image: Image.Image,
    texts: list[ScoredBox],
    *,
    mode: str = "blur",
    blur_radius: float = 14.0,
    pixel_size: int = 12,
) -> tuple[Image.Image, list[str]]:
    """Text operator: blur (balanced), fill (privacy), or pixelate (utility)."""
    out = image.convert("RGB").copy()
    w, h = out.size
    ops: list[str] = []
    for b in texts:
        x1, y1, x2, y2, score = b
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if mode == "fill":
            fill = Image.new("RGB", (x2 - x1, y2 - y1), (0, 0, 0))
            out.paste(fill, (x1, y1))
            ops.append(f"text_fill@{score:.2f}")
        elif mode == "pixelate":
            region = out.crop((x1, y1, x2, y2))
            rw, rh = region.size
            small = region.resize(
                (max(1, rw // pixel_size), max(1, rh // pixel_size)),
                Image.Resampling.BILINEAR,
            )
            out.paste(small.resize((rw, rh), Image.Resampling.NEAREST), (x1, y1))
            ops.append(f"text_pixelate@{score:.2f}")
        else:
            region = out.crop((x1, y1, x2, y2))
            out.paste(region.filter(ImageFilter.GaussianBlur(radius=blur_radius)), (x1, y1))
            ops.append(f"text_blur@{score:.2f}")
    return out, ops
