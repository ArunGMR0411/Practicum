"""End-to-end image anonymisation pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from src.anonymisation.base_anonymiser import AnonymiserResult, BaseAnonymiser
from src.anonymisation.registry import build_anonymiser_registry
from src.anonymisation.screen_redactor import ScreenRedactor
from src.anonymisation.text_redactor import TextRedactor
from src.detection import (
    MTCNNDetector,
    ScreenDetector,
    TextDetector,
    YOLOSCRFDFallbackDetector,
    YOLOSCRFDRetinaFaceSelectiveDetector,
)
from src.detection.base_detector import DetectionResult
from src.evaluation.perceptual_metrics import LPIPSEvaluator, compute_ssim
from src.routing import QualityAssessor, RuleBasedRouter
from src.utils.logger import get_logger
from src.utils.system_config import resolve_torch_device
from src.utils.timer import StageTimer


class RoutingProtocol(Protocol):
    def decide(self, assessment: Any) -> Any:
        """Return a routing decision with method_name and metadata."""


class OCREvaluatorProtocol(Protocol):
    def recognise_regions(self, image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> list[Any]:
        """Recognise text inside regions."""

    @staticmethod
    def suppression_rate(
        original_results: list[Any],
        redacted_results: list[Any],
        similarity_threshold: float = 0.5,
    ) -> float:
        """Compute OCR suppression rate."""


class ReIDEvaluatorProtocol(Protocol):
    def extract_embeddings_adaface(self, crops: list[Image.Image], batch_size: int = 64) -> Any:
        """Extract embeddings for crops."""

    def compute_reid_metrics(self, gallery_embeddings: Any, query_embeddings: Any) -> dict[str, float]:
        """Compute re-identification metrics."""


@dataclass(frozen=True)
class PipelineConfig:
    face_detector_backend: str = "yolo_scrfd_fallback"
    detector_operating_mode: str = "operational"
    text_detector_backend: str = "easyocr"
    screen_model_path: str = "data/models/yolov8n.pt"
    face_detector_confidence: float = 0.75
    text_similarity_threshold: float = 0.5
    preferred_face_method: str = "rule_based"
    timer_output_path: str = "outputs/timing.json"
    log_dir: str = "logs"
    perceptual_eval_scale: float = 0.25


class CASTLEPipeline:
    """Master pipeline for one image through detection, routing, anonymisation, and evaluation."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        face_detector: Any | None = None,
        text_detector: Any | None = None,
        screen_detector: Any | None = None,
        router: RoutingProtocol | None = None,
        quality_assessor: QualityAssessor | None = None,
        anonymiser_registry: dict[str, BaseAnonymiser] | None = None,
        text_redactor: BaseAnonymiser | None = None,
        screen_redactor: BaseAnonymiser | None = None,
        lpips_evaluator: Any | None = None,
        ocr_evaluator: OCREvaluatorProtocol | None = None,
        reid_evaluator: ReIDEvaluatorProtocol | None = None,
        timer: StageTimer | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.device = resolve_torch_device()
        self.logger = get_logger("pipeline", log_dir=self.config.log_dir)
        self.timer = timer or StageTimer(self.config.timer_output_path)

        self.face_detector = face_detector or self._build_face_detector()
        self._use_injected_multimodal_detectors = (
            text_detector is not None or screen_detector is not None
        )
        self.text_detector = text_detector or TextDetector(
            backend=self.config.text_detector_backend,
            device=self.device,
        )
        self.screen_detector = screen_detector or ScreenDetector(
            model_path=self.config.screen_model_path,
            device="0" if self.device == "cuda" else self.device,
        )
        self.router = router or RuleBasedRouter()
        self.quality_assessor = quality_assessor or QualityAssessor()
        self.anonymiser_registry = anonymiser_registry or build_anonymiser_registry()
        # Evidence adaptive path: text blur + screen fill (area-aware stack handles huge screens).
        self.text_redactor = text_redactor or TextRedactor(mode="blur")
        self.screen_redactor = screen_redactor or ScreenRedactor(mode="fill")
        self.lpips_evaluator = lpips_evaluator
        self.ocr_evaluator = ocr_evaluator
        self.reid_evaluator = reid_evaluator

    def _build_face_detector(self) -> Any:
        if self.config.face_detector_backend == "mtcnn":
            return MTCNNDetector(
                confidence_threshold=self.config.face_detector_confidence,
                device=self.device,
            )
        if self.config.face_detector_backend == "yolo_scrfd_fallback":
            return YOLOSCRFDFallbackDetector(
                yolo_confidence_threshold=self.config.face_detector_confidence,
                yolo_device=self.device,
            )
        if self.config.face_detector_backend == "yolo_scrfd_retinaface_selective":
            base_detector = YOLOSCRFDFallbackDetector(
                yolo_confidence_threshold=self.config.face_detector_confidence,
                yolo_device=self.device,
            )
            return YOLOSCRFDRetinaFaceSelectiveDetector(
                base_detector=base_detector,
                trigger_prediction_count_threshold=8,
            )
        raise ValueError(f"Unsupported face_detector_backend: {self.config.face_detector_backend}")

    def _get_anonymiser(self, method_name: str) -> BaseAnonymiser:
        anonymiser = self.anonymiser_registry.get(method_name)
        if anonymiser is None:
            raise KeyError(f"Anonymiser not registered: {method_name}")
        return anonymiser

    @staticmethod
    def _extract_boxes(result: DetectionResult) -> list[tuple[int, int, int, int]]:
        return [detection.box for detection in result.detections]

    @staticmethod
    def _crop_boxes(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> list[Image.Image]:
        return [image.crop(box) for box in boxes]

    def _compute_reid_confidence(
        self,
        original: Image.Image,
        anonymised: Image.Image,
        face_boxes: list[tuple[int, int, int, int]],
    ) -> float | None:
        if self.reid_evaluator is None or not face_boxes:
            return None
        original_crops = self._crop_boxes(original, face_boxes)
        anonymised_crops = self._crop_boxes(anonymised, face_boxes)
        gallery = self.reid_evaluator.extract_embeddings_adaface(original_crops)
        query = self.reid_evaluator.extract_embeddings_adaface(anonymised_crops)
        metrics = self.reid_evaluator.compute_reid_metrics(gallery, query)
        return float(metrics.get("cosine_similarity", 0.0))

    def _prepare_perceptual_images(
        self,
        original: Image.Image,
        anonymised: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        scale = float(self.config.perceptual_eval_scale)
        if scale <= 0.0 or scale == 1.0:
            return original, anonymised
        width = max(1, int(round(original.width * scale)))
        height = max(1, int(round(original.height * scale)))
        return (
            original.resize((width, height), Image.Resampling.LANCZOS),
            anonymised.resize((width, height), Image.Resampling.LANCZOS),
        )

    def process_image(self, image_path: str | Path) -> dict[str, Any]:
        image_path = Path(image_path)
        self.timer.records = {}
        with self.timer.track("load_image"):
            original_image = Image.open(image_path).convert("RGB")

        with self.timer.track("detect_faces"):
            face_result = self.face_detector.detect(original_image)
            face_boxes = self._extract_boxes(face_result)

        # Screen-first sequential multimodal path (evidence stack):
        # detect screens on original → redact screens on a working copy → detect text.
        # Prefer MultimodalStack when available; fall back to independent detectors.
        text_boxes: list[tuple[int, int, int, int]] = []
        screen_boxes: list[tuple[int, int, int, int]] = []
        multimodal_meta: dict[str, Any] = {}
        with self.timer.track("detect_screens_then_text"):
            if self._use_injected_multimodal_detectors:
                screen_result = self.screen_detector.detect(original_image)
                screen_boxes = self._extract_boxes(screen_result)
                from src.anonymisation.screen_redactor import ScreenRedactor as _SR

                work = _SR(mode="fill").anonymise(original_image, screen_boxes).image
                text_result = self.text_detector.detect(work)
                text_boxes = self._extract_boxes(text_result)
                multimodal_meta = {"policy": "injected_detectors_screen_first"}
            else:
                try:
                    from src.detection.multimodal_stack import MultimodalStack

                    stack = getattr(self, "_multimodal_stack", None)
                    if stack is None:
                        stack = MultimodalStack(device=self.device)
                        self._multimodal_stack = stack
                    mm = stack.detect(original_image)
                    screen_boxes = [b[:4] for b in mm.screens]
                    text_boxes = [b[:4] for b in mm.texts]
                    multimodal_meta = {
                        "screen_policy": mm.screen_policy,
                        "text_policy": mm.text_policy,
                        "screen_sources": mm.screen_sources,
                        "aux_text_for_screen_hyp": mm.aux_text_for_screen_hyp,
                    }
                except Exception as exc:
                    self.logger.warning("MultimodalStack unavailable (%s); using independent detectors.", exc)
                    screen_result = self.screen_detector.detect(original_image)
                    screen_boxes = self._extract_boxes(screen_result)
                    # Still do screen-first for text: redact screens then detect text.
                    from src.anonymisation.screen_redactor import ScreenRedactor as _SR

                    work = _SR(mode="fill").anonymise(original_image, screen_boxes).image
                    text_result = self.text_detector.detect(work)
                    text_boxes = self._extract_boxes(text_result)
                    multimodal_meta = {"fallback": "independent_detectors_screen_first", "error": str(exc)}

        with self.timer.track("route_faces"):
            assessment = self.quality_assessor.assess(original_image, face_boxes=face_boxes)
            decision = self.router.decide(assessment)

        with self.timer.track("anonymise_faces"):
            anonymiser = self._get_anonymiser(decision.method_name)
            face_result_image: AnonymiserResult = anonymiser.anonymise(original_image, face_boxes)

        with self.timer.track("redact_screens"):
            fully_screen = self.screen_redactor.anonymise(face_result_image.image, screen_boxes)

        with self.timer.track("redact_text"):
            fully_redacted = self.text_redactor.anonymise(fully_screen.image, text_boxes)

        final_image = fully_redacted.image

        with self.timer.track("evaluate_perceptual"):
            original_eval, final_eval = self._prepare_perceptual_images(original_image, final_image)
            ssim_score = compute_ssim(original_eval, final_eval)
            lpips_score = (
                float(self.lpips_evaluator.compute_lpips(original_eval, final_eval))
                if self.lpips_evaluator is not None
                else None
            )

        with self.timer.track("evaluate_reid"):
            reid_confidence = self._compute_reid_confidence(original_image, final_image, face_boxes)

        with self.timer.track("evaluate_text"):
            text_redaction_rate = None
            if self.ocr_evaluator is not None and text_boxes:
                original_ocr = self.ocr_evaluator.recognise_regions(original_image, text_boxes)
                redacted_ocr = self.ocr_evaluator.recognise_regions(final_image, text_boxes)
                text_redaction_rate = float(
                    self.ocr_evaluator.suppression_rate(
                        original_ocr,
                        redacted_ocr,
                        similarity_threshold=self.config.text_similarity_threshold,
                    )
                )

        screen_redaction_rate = float(bool(screen_boxes)) if screen_boxes else 0.0
        self.timer.save()

        per_face_methods = [
            {
                "box": list(box),
                "method": decision.method_name,
            }
            for box in face_boxes
        ]

        result = {
            "image_path": str(image_path),
            "anonymised_image": final_image,
            "face_detection_count": len(face_boxes),
            "text_detection_count": len(text_boxes),
            "screen_detection_count": len(screen_boxes),
            "per_face_method_selected": per_face_methods,
            "face_routing_method": decision.method_name,
            "face_routing_metadata": dict(decision.metadata),
            "ssim": float(ssim_score),
            "lpips": lpips_score,
            "reid_confidence": reid_confidence,
            "text_redaction_rate": text_redaction_rate,
            "screen_redaction_rate": screen_redaction_rate,
            "multimodal_meta": multimodal_meta,
            "timings": dict(self.timer.records),
            "metadata": {
                "face_detector": getattr(self.face_detector, "detector_name", type(self.face_detector).__name__),
                "text_detector": getattr(self.text_detector, "detector_name", type(self.text_detector).__name__),
                "screen_detector": getattr(self.screen_detector, "detector_name", type(self.screen_detector).__name__),
                "detector_operating_mode": self.config.detector_operating_mode,
            },
        }
        self.logger.info(
            "Processed %s with %d faces, %d text regions, %d screens via %s",
            image_path,
            len(face_boxes),
            len(text_boxes),
            len(screen_boxes),
            decision.method_name,
        )
        return result
