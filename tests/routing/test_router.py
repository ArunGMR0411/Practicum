"""Unit tests for the adaptive routing layer.

Covers:
- QualityAssessor: correct return types and signal bounds.
- RuleBasedRouter: valid method for every quality profile, calibration load.
- LearnedRouter: loads and predicts without error, returns valid method names.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from src.routing.quality_assessor import QualityAssessment, QualityAssessor, QualitySignals
from src.routing.router import RouterDecision, RuleBasedRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_image(width: int = 64, height: int = 64, seed: int = 42) -> Image.Image:
    """Return a deterministic RGB PIL image for testing."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _blank_box(size: int = 50) -> tuple[int, int, int, int]:
    return (10, 10, 10 + size, 10 + size)


def _minimal_calibration_json(
    face_size_threshold_px: float = 300.0,
    blur_threshold: float = 0.05,
) -> dict:
    return {
        "best_thresholds": {
            "face_size_threshold_px": face_size_threshold_px,
            "blur_threshold": blur_threshold,
        }
    }


# ---------------------------------------------------------------------------
# QualityAssessor tests
# ---------------------------------------------------------------------------

class TestQualityAssessorReturnTypes(unittest.TestCase):
    """QualityAssessor must return correctly typed objects."""

    def setUp(self) -> None:
        self.assessor = QualityAssessor()
        self.image = _synthetic_image()

    def test_returns_quality_assessment_instance(self) -> None:
        result = self.assessor.assess(self.image)
        self.assertIsInstance(result, QualityAssessment)

    def test_signals_are_quality_signals_instance(self) -> None:
        result = self.assessor.assess(self.image)
        self.assertIsInstance(result.signals, QualitySignals)

    def test_all_signals_are_float(self) -> None:
        result = self.assessor.assess(self.image)
        s = result.signals
        self.assertIsInstance(s.blur_score, float)
        self.assertIsInstance(s.face_size_px, float)
        self.assertIsInstance(s.occlusion_ratio, float)
        self.assertIsInstance(s.webp_artifact_score, float)

    def test_blur_score_positive(self) -> None:
        result = self.assessor.assess(self.image)
        self.assertGreater(result.signals.blur_score, 0.0)

    def test_face_size_zero_when_no_boxes(self) -> None:
        result = self.assessor.assess(self.image, face_boxes=None)
        self.assertEqual(result.signals.face_size_px, 0.0)

    def test_face_size_positive_when_box_provided(self) -> None:
        result = self.assessor.assess(self.image, face_boxes=[_blank_box()])
        self.assertGreater(result.signals.face_size_px, 0.0)

    def test_occlusion_ratio_bounded(self) -> None:
        result = self.assessor.assess(self.image, face_boxes=[_blank_box()])
        self.assertGreaterEqual(result.signals.occlusion_ratio, 0.0)
        self.assertLessEqual(result.signals.occlusion_ratio, 1.0)

    def test_occlusion_ratio_is_one_when_no_face(self) -> None:
        """No face box → occlusion defaults to 1.0 (fully occluded / no info)."""
        result = self.assessor.assess(self.image, face_boxes=[])
        self.assertEqual(result.signals.occlusion_ratio, 1.0)

    def test_webp_artifact_score_non_negative(self) -> None:
        result = self.assessor.assess(self.image)
        self.assertGreaterEqual(result.signals.webp_artifact_score, 0.0)

    def test_metadata_contains_expected_keys(self) -> None:
        result = self.assessor.assess(self.image, face_boxes=[_blank_box()])
        for key in ("image_width", "image_height", "face_box_count", "dominant_face_box"):
            self.assertIn(key, result.metadata)

    def test_face_box_count_matches_input(self) -> None:
        boxes = [_blank_box(30), _blank_box(20)]
        result = self.assessor.assess(self.image, face_boxes=boxes)
        self.assertEqual(result.metadata["face_box_count"], 2)

    def test_to_dict_returns_dict(self) -> None:
        result = self.assessor.assess(self.image)
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("blur_score", d)

    def test_very_small_image_does_not_crash(self) -> None:
        tiny = _synthetic_image(width=4, height=4)
        result = self.assessor.assess(tiny)
        self.assertIsInstance(result, QualityAssessment)

    def test_uniform_image_blur_score_high(self) -> None:
        """A perfectly uniform image has zero Laplacian variance → very high blur_score."""
        uniform = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8), mode="RGB")
        result = self.assessor.assess(uniform)
        self.assertGreater(result.signals.blur_score, 1.0)


# ---------------------------------------------------------------------------
# RuleBasedRouter tests
# ---------------------------------------------------------------------------

class TestRuleBasedRouterDecisions(unittest.TestCase):
    """RuleBasedRouter must assign a valid method for every quality profile."""

    def _make_router(self, face_size_threshold_px: float = 300.0, blur_threshold: float = 0.05):
        calibration = _minimal_calibration_json(face_size_threshold_px, blur_threshold)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(calibration, fh)
            tmp_path = Path(fh.name)
        registry_patch = patch(
            "src.routing.router.build_anonymiser_registry",
            return_value={"blur": object(), "pixelate": object()},
        )
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        return RuleBasedRouter(routing_calibration_path=tmp_path), tmp_path

    def _make_assessment(
        self,
        blur_score: float = 0.01,
        face_size_px: float = 100.0,
        occlusion_ratio: float = 0.5,
        webp_artifact_score: float = 0.0,
        face_box_count: int = 1,
    ) -> QualityAssessment:
        return QualityAssessment(
            signals=QualitySignals(
                blur_score=blur_score,
                face_size_px=face_size_px,
                occlusion_ratio=occlusion_ratio,
                webp_artifact_score=webp_artifact_score,
            ),
            metadata={"face_box_count": face_box_count},
        )

    def test_returns_router_decision_instance(self) -> None:
        router, tmp = self._make_router()
        assessment = self._make_assessment()
        decision = router.decide(assessment)
        self.assertIsInstance(decision, RouterDecision)
        tmp.unlink()

    def test_decision_method_name_is_string(self) -> None:
        router, tmp = self._make_router()
        decision = router.decide(self._make_assessment())
        self.assertIsInstance(decision.method_name, str)
        tmp.unlink()

    def test_small_face_routes_to_blur(self) -> None:
        """face_size_px below threshold → blur regardless of blur score."""
        router, tmp = self._make_router(face_size_threshold_px=300.0, blur_threshold=0.05)
        decision = router.decide(self._make_assessment(face_size_px=50.0, blur_score=0.001))
        self.assertEqual(decision.method_name, "blur")
        self.assertEqual(decision.metadata["route_reason"], "small_or_medium_face")
        tmp.unlink()

    def test_high_blur_routes_to_blur(self) -> None:
        """Large face but blurry frame → blur."""
        router, tmp = self._make_router(face_size_threshold_px=300.0, blur_threshold=0.05)
        decision = router.decide(self._make_assessment(face_size_px=500.0, blur_score=0.9))
        self.assertEqual(decision.method_name, "blur")
        self.assertEqual(decision.metadata["route_reason"], "high_blur")
        tmp.unlink()

    def test_sharp_large_face_routes_to_pixelate(self) -> None:
        """Large sharp face → pixelate (when available)."""
        router, tmp = self._make_router(face_size_threshold_px=300.0, blur_threshold=0.05)
        decision = router.decide(self._make_assessment(face_size_px=500.0, blur_score=0.001))
        self.assertEqual(decision.method_name, "pixelate")
        self.assertEqual(decision.metadata["route_reason"], "sharper_large_face")
        tmp.unlink()

    def test_no_face_routes_to_blur(self) -> None:
        """No detected face → face_size_px = 0 → below any positive threshold → blur."""
        router, tmp = self._make_router(face_size_threshold_px=300.0)
        decision = router.decide(self._make_assessment(face_size_px=0.0, blur_score=0.001))
        self.assertEqual(decision.method_name, "blur")
        tmp.unlink()

    def test_metadata_contains_thresholds(self) -> None:
        router, tmp = self._make_router(face_size_threshold_px=300.0, blur_threshold=0.05)
        decision = router.decide(self._make_assessment())
        self.assertIn("face_size_threshold_px", decision.metadata)
        self.assertIn("blur_threshold", decision.metadata)
        tmp.unlink()

    def test_missing_calibration_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            RuleBasedRouter(routing_calibration_path="/nonexistent/calibration.json")

    def test_policy_version_in_metadata(self) -> None:
        router, tmp = self._make_router()
        decision = router.decide(self._make_assessment())
        self.assertIn("policy_version", decision.metadata)
        tmp.unlink()

    def test_all_quality_signals_in_metadata(self) -> None:
        router, tmp = self._make_router()
        decision = router.decide(self._make_assessment(blur_score=0.03, face_size_px=150.0))
        for key in ("blur_score", "face_size_px", "occlusion_ratio", "webp_artifact_score"):
            self.assertIn(key, decision.metadata)
        tmp.unlink()


# ---------------------------------------------------------------------------
# LearnedRouter tests
# ---------------------------------------------------------------------------

class TestLearnedRouter(unittest.TestCase):
    """LearnedRouter must load and predict without error and return valid method names."""

    def _make_mock_model(self, classes=("blur", "pixelate")) -> MagicMock:
        model = MagicMock()
        model.predict.return_value = [classes[0]]
        # Return a numpy array so that [0].tolist() works, matching the real sklearn API.
        model.predict_proba.return_value = np.array([[0.7, 0.3]])
        return model

    def _make_mock_payload(self, classes=("blur", "pixelate")) -> dict:
        return {
            "model": self._make_mock_model(classes),
            "feature_names": ["blur_score", "face_size_px", "occlusion_ratio", "webp_artifact_score", "face_box_count"],
            "classes": list(classes),
            "model_version": "learned_router",
        }

    def _make_assessment(self) -> QualityAssessment:
        return QualityAssessment(
            signals=QualitySignals(
                blur_score=0.02,
                face_size_px=200.0,
                occlusion_ratio=0.4,
                webp_artifact_score=0.1,
            ),
            metadata={"face_box_count": 1},
        )

    def test_decide_returns_router_decision(self) -> None:
        from src.routing.learned_router import LearnedRouter
        payload = self._make_mock_payload()
        with patch("joblib.load", return_value=payload):
            router = LearnedRouter(model_path="/fake/model.joblib")
        decision = router.decide(self._make_assessment())
        self.assertIsInstance(decision, RouterDecision)

    def test_decide_method_name_is_in_classes(self) -> None:
        from src.routing.learned_router import LearnedRouter
        classes = ("blur", "pixelate")
        payload = self._make_mock_payload(classes)
        with patch("joblib.load", return_value=payload):
            router = LearnedRouter(model_path="/fake/model.joblib")
        decision = router.decide(self._make_assessment())
        self.assertIn(decision.method_name, classes)

    def test_decide_metadata_contains_class_probabilities(self) -> None:
        from src.routing.learned_router import LearnedRouter
        payload = self._make_mock_payload()
        with patch("joblib.load", return_value=payload):
            router = LearnedRouter(model_path="/fake/model.joblib")
        decision = router.decide(self._make_assessment())
        self.assertIn("class_probabilities", decision.metadata)
        self.assertIsInstance(decision.metadata["class_probabilities"], dict)

    def test_decide_metadata_contains_feature_names(self) -> None:
        from src.routing.learned_router import LearnedRouter
        payload = self._make_mock_payload()
        with patch("joblib.load", return_value=payload):
            router = LearnedRouter(model_path="/fake/model.joblib")
        decision = router.decide(self._make_assessment())
        self.assertIn("feature_names", decision.metadata)

    def test_decide_all_signal_values_in_metadata(self) -> None:
        from src.routing.learned_router import LearnedRouter
        payload = self._make_mock_payload()
        with patch("joblib.load", return_value=payload):
            router = LearnedRouter(model_path="/fake/model.joblib")
        decision = router.decide(self._make_assessment())
        for key in ("blur_score", "face_size_px", "occlusion_ratio", "webp_artifact_score", "face_box_count"):
            self.assertIn(key, decision.metadata)

    def test_missing_model_file_raises(self) -> None:
        from src.routing.learned_router import LearnedRouter
        with self.assertRaises(Exception):
            LearnedRouter(model_path="/nonexistent/model.joblib")

    def test_decide_with_three_classes(self) -> None:
        """Router must handle 3-class models (blur/pixelate/diffusion) gracefully."""
        from src.routing.learned_router import LearnedRouter
        classes = ("blur", "diffusion", "pixelate")
        payload = self._make_mock_payload(classes)
        payload["model"].predict.return_value = ["diffusion"]
        payload["model"].predict_proba.return_value = np.array([[0.3, 0.5, 0.2]])
        with patch("joblib.load", return_value=payload):
            router = LearnedRouter(model_path="/fake/model.joblib")
        decision = router.decide(self._make_assessment())
        self.assertEqual(decision.method_name, "diffusion")


if __name__ == "__main__":
    unittest.main()
