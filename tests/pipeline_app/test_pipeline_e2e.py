"""E2E tests for the master CASTLE pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from src.anonymisation.base_anonymiser import AnonymiserResult, BaseAnonymiser
from src.detection.base_detector import Detection, DetectionResult
from src.pipeline import CASTLEPipeline, PipelineConfig
from src.routing.quality_assessor import QualityAssessment, QualitySignals
from src.routing.router import RouterDecision
from src.utils.timer import StageTimer


TEST_OUTPUT_DIR = Path("/tmp/practicum_work/pipeline_tests")


class StubDetector:
    def __init__(self, detections: list[tuple[int, int, int, int]], detector_name: str) -> None:
        self._detections = detections
        self.detector_name = detector_name

    def detect(self, image: Image.Image) -> DetectionResult:
        return DetectionResult(
            detections=[Detection(box=box, confidence=1.0) for box in self._detections],
            metadata={"detector_name": self.detector_name},
        )


class StubRouter:
    def __init__(self, method_name: str = "blur") -> None:
        self.method_name = method_name

    def decide(self, assessment: QualityAssessment) -> RouterDecision:
        return RouterDecision(
            method_name=self.method_name,
            metadata={
                "route_reason": "test",
                "face_size_px": assessment.signals.face_size_px,
                "blur_score": assessment.signals.blur_score,
            },
        )


class StubAnonymiser(BaseAnonymiser):
    method_name = "stub"

    def __init__(self, fill_color: tuple[int, int, int]) -> None:
        self.fill_color = fill_color

    def anonymise(self, image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> AnonymiserResult:
        output = image.copy()
        for left, top, right, bottom in self.validate_boxes(output, boxes):
            patch = Image.new("RGB", (right - left, bottom - top), self.fill_color)
            output.paste(patch, (left, top))
        return AnonymiserResult(
            image=output,
            metadata={"method": self.method_name, "boxes_processed": len(boxes)},
        )


class StubQualityAssessor:
    def assess(
        self,
        image: Image.Image,
        face_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> QualityAssessment:
        return QualityAssessment(
            signals=QualitySignals(
                blur_score=0.01,
                face_size_px=144.0 if face_boxes else 0.0,
                occlusion_ratio=0.1,
                webp_artifact_score=0.0,
            ),
            metadata={"face_box_count": len(face_boxes or [])},
        )


class StubLPIPS:
    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        return 0.123


class StubOCR:
    def recognise_regions(self, image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> list[str]:
        return ["visible" for _ in boxes]

    @staticmethod
    def suppression_rate(
        original_results: list[str],
        redacted_results: list[str],
        similarity_threshold: float = 0.5,
    ) -> float:
        return 1.0 if original_results else 0.0


class StubReID:
    def extract_embeddings_adaface(self, crops: list[Image.Image], batch_size: int = 64) -> list[int]:
        return [1 for _ in crops]

    def compute_reid_metrics(self, gallery_embeddings: list[int], query_embeddings: list[int]) -> dict[str, float]:
        return {"cosine_similarity": 0.25, "reid_rate": 0.0}


def _make_test_image(path: Path, with_face: bool = True, with_text: bool = True) -> None:
    image = Image.new("RGB", (96, 96), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    if with_face:
        draw.rectangle((10, 10, 40, 40), fill=(180, 120, 120))
    if with_text:
        draw.rectangle((50, 10, 90, 25), fill=(20, 20, 20))
    draw.rectangle((50, 40, 90, 80), outline=(0, 0, 255), width=2)
    image.save(path)


class TestPipelineE2E(unittest.TestCase):
    def _make_pipeline(
        self,
        *,
        face_boxes: list[tuple[int, int, int, int]],
        text_boxes: list[tuple[int, int, int, int]],
        screen_boxes: list[tuple[int, int, int, int]],
    ) -> CASTLEPipeline:
        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timer_path = TEST_OUTPUT_DIR / "timing.json"
        return CASTLEPipeline(
            config=PipelineConfig(
                timer_output_path=str(timer_path),
                log_dir=str(TEST_OUTPUT_DIR),
            ),
            face_detector=StubDetector(face_boxes, "stub_face"),
            text_detector=StubDetector(text_boxes, "stub_text"),
            screen_detector=StubDetector(screen_boxes, "stub_screen"),
            router=StubRouter("blur"),
            quality_assessor=StubQualityAssessor(),
            anonymiser_registry={"blur": StubAnonymiser((0, 0, 0))},
            text_redactor=StubAnonymiser((255, 0, 0)),
            screen_redactor=StubAnonymiser((0, 255, 0)),
            lpips_evaluator=StubLPIPS(),
            ocr_evaluator=StubOCR(),
            reid_evaluator=StubReID(),
            timer=StageTimer(timer_path),
        )

    def test_pipeline_handles_webp_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.webp"
            _make_test_image(image_path)
            pipeline = self._make_pipeline(
                face_boxes=[(10, 10, 40, 40)],
                text_boxes=[(50, 10, 90, 25)],
                screen_boxes=[(50, 40, 90, 80)],
            )
            result = pipeline.process_image(image_path)

        self.assertEqual(result["face_detection_count"], 1)
        self.assertEqual(result["text_detection_count"], 1)
        self.assertEqual(result["screen_detection_count"], 1)
        self.assertEqual(result["face_routing_method"], "blur")
        self.assertEqual(len(result["per_face_method_selected"]), 1)
        self.assertIsInstance(result["anonymised_image"], Image.Image)
        self.assertIsInstance(result["ssim"], float)
        self.assertEqual(result["lpips"], 0.123)
        self.assertEqual(result["reid_confidence"], 0.25)
        self.assertEqual(result["text_redaction_rate"], 1.0)
        self.assertEqual(result["screen_redaction_rate"], 1.0)
        self.assertIn("detect_faces", result["timings"])

    def test_pipeline_handles_png_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            _make_test_image(image_path)
            pipeline = self._make_pipeline(
                face_boxes=[(10, 10, 40, 40)],
                text_boxes=[],
                screen_boxes=[],
            )
            result = pipeline.process_image(image_path)

        self.assertEqual(result["face_detection_count"], 1)
        self.assertEqual(result["text_detection_count"], 0)
        self.assertEqual(result["screen_detection_count"], 0)

    def test_pipeline_handles_small_synthetic_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "tiny.png"
            Image.new("RGB", (8, 8), (128, 128, 128)).save(image_path)
            pipeline = self._make_pipeline(face_boxes=[], text_boxes=[], screen_boxes=[])
            result = pipeline.process_image(image_path)

        self.assertEqual(result["face_detection_count"], 0)
        self.assertIsNone(result["reid_confidence"])

    def test_pipeline_handles_image_with_no_face(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "noface.webp"
            _make_test_image(image_path, with_face=False)
            pipeline = self._make_pipeline(
                face_boxes=[],
                text_boxes=[(50, 10, 90, 25)],
                screen_boxes=[],
            )
            result = pipeline.process_image(image_path)

        self.assertEqual(result["face_detection_count"], 0)
        self.assertEqual(result["per_face_method_selected"], [])
        self.assertIsNone(result["reid_confidence"])

    def test_pipeline_handles_image_with_no_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "notext.webp"
            _make_test_image(image_path, with_text=False)
            pipeline = self._make_pipeline(
                face_boxes=[(10, 10, 40, 40)],
                text_boxes=[],
                screen_boxes=[(50, 40, 90, 80)],
            )
            result = pipeline.process_image(image_path)

        self.assertEqual(result["text_detection_count"], 0)
        self.assertIsNone(result["text_redaction_rate"])
        self.assertEqual(result["screen_redaction_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
