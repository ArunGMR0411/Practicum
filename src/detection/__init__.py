"""Detection package exports with lazy backend imports."""

from __future__ import annotations

from src.detection.base_detector import BaseDetector, BoundingBox, Detection, DetectionResult

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "MTCNNDetector",
    "RetinaFaceDetector",
    "YOLODetector",
    "YOLOSCRFDFallbackDetector",
    "YOLOSCRFDRetinaFaceSelectiveDetector",
    "TextDetector",
    "ScreenDetector",
]


def __getattr__(name: str):
    if name == "MTCNNDetector":
        from src.detection.mtcnn_detector import MTCNNDetector

        return MTCNNDetector
    if name == "RetinaFaceDetector":
        from src.detection.retinaface_detector import RetinaFaceDetector

        return RetinaFaceDetector
    if name == "YOLODetector":
        from src.detection.yolo_detector import YOLODetector

        return YOLODetector
    if name == "YOLOSCRFDFallbackDetector":
        from src.detection.yolo_scrfd_fallback_detector import YOLOSCRFDFallbackDetector

        return YOLOSCRFDFallbackDetector
    if name == "YOLOSCRFDRetinaFaceSelectiveDetector":
        from src.detection.yolo_scrfd_retinaface_selective_detector import YOLOSCRFDRetinaFaceSelectiveDetector

        return YOLOSCRFDRetinaFaceSelectiveDetector
    if name == "TextDetector":
        from src.detection.text_detector import TextDetector

        return TextDetector
    if name == "ScreenDetector":
        from src.detection.screen_detector import ScreenDetector

        return ScreenDetector
    raise AttributeError(f"module 'src.detection' has no attribute {name!r}")
