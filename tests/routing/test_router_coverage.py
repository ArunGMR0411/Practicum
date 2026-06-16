"""Coverage tests for evidence-bearing routing branches."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.routing.learned_router import LearnedRouter
from src.routing.quality_assessor import QualityAssessment, QualitySignals
from src.routing.router import RuleBasedRouter


def _make_assessment(
    *,
    blur_score: float,
    face_size_px: float,
    occlusion_ratio: float = 0.35,
    webp_artifact_score: float = 0.10,
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


class TestRuleRouterCoverage(unittest.TestCase):
    """Every active rule-based branch should be reachable by a realistic profile."""

    def _make_router(self) -> tuple[RuleBasedRouter, Path]:
        calibration = {
            "best_thresholds": {
                "face_size_threshold_px": 300.0,
                "blur_threshold": 0.05,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(calibration, handle)
            path = Path(handle.name)
        return RuleBasedRouter(routing_calibration_path=path), path

    def test_blur_branch_reachable_from_small_face_profile(self) -> None:
        router, path = self._make_router()
        assessment = _make_assessment(
            blur_score=0.01,
            face_size_px=128.0,
            occlusion_ratio=0.30,
            webp_artifact_score=0.08,
        )
        decision = router.decide(assessment)
        self.assertEqual(decision.method_name, "blur")
        self.assertEqual(decision.metadata["route_reason"], "small_or_medium_face")
        path.unlink()

    def test_blur_branch_reachable_from_high_blur_large_face_profile(self) -> None:
        router, path = self._make_router()
        assessment = _make_assessment(
            blur_score=0.22,
            face_size_px=540.0,
            occlusion_ratio=0.18,
            webp_artifact_score=0.05,
        )
        decision = router.decide(assessment)
        self.assertEqual(decision.method_name, "blur")
        self.assertEqual(decision.metadata["route_reason"], "high_blur")
        path.unlink()

    def test_pixelate_branch_reachable_from_sharp_large_face_profile(self) -> None:
        router, path = self._make_router()
        assessment = _make_assessment(
            blur_score=0.01,
            face_size_px=620.0,
            occlusion_ratio=0.12,
            webp_artifact_score=0.04,
        )
        decision = router.decide(assessment)
        self.assertEqual(decision.method_name, "pixelate")
        self.assertEqual(decision.metadata["route_reason"], "sharper_large_face")
        path.unlink()


class TestExploratoryDiffusionCoverage(unittest.TestCase):
    """The exploratory diffusion branch should remain reachable in 3-class routing mocks."""

    def _payload(self, predicted_label: str) -> dict:
        model = unittest.mock.MagicMock()
        model.predict.return_value = [predicted_label]
        model.predict_proba.return_value = np.array([[0.15, 0.70, 0.15]])
        return {
            "model": model,
            "feature_names": [
                "blur_score",
                "face_size_px",
                "occlusion_ratio",
                "webp_artifact_score",
                "face_box_count",
            ],
            "classes": ["blur", "diffusion", "pixelate"],
            "model_version": "learned_router",
        }

    def test_diffusion_branch_reachable_under_three_class_router(self) -> None:
        assessment = _make_assessment(
            blur_score=0.16,
            face_size_px=420.0,
            occlusion_ratio=0.22,
            webp_artifact_score=0.11,
        )
        with patch("joblib.load", return_value=self._payload("diffusion")):
            router = LearnedRouter(model_path="/fake/learned_router.joblib")
        decision = router.decide(assessment)
        self.assertEqual(decision.method_name, "diffusion")
        self.assertIn("diffusion", decision.metadata["class_probabilities"])


if __name__ == "__main__":
    unittest.main()
