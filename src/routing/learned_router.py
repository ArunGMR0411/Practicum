"""Learned routing baseline backed by a scikit-learn classifier."""

from __future__ import annotations

import joblib
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.routing.quality_assessor import QualityAssessment
from src.routing.router import RouterDecision

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "learned_router.joblib"


@dataclass(frozen=True)
class LearnedRouterMetadata:
    """Metadata stored alongside the learned routing model."""

    feature_names: list[str]
    classes: list[str]
    model_version: str


class LearnedRouter:
    """Predict the preferred anonymisation method from quality features."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        payload = joblib.load(Path(model_path))
        self.model = payload["model"]
        self.metadata = LearnedRouterMetadata(
            feature_names=list(payload["feature_names"]),
            classes=list(payload["classes"]),
            model_version=str(payload.get("model_version", "learned_router")),
        )

    def decide(self, assessment: QualityAssessment) -> RouterDecision:
        feature_row = pd.DataFrame(
            [
                {
                    "blur_score": float(assessment.signals.blur_score),
                    "face_size_px": float(assessment.signals.face_size_px),
                    "occlusion_ratio": float(assessment.signals.occlusion_ratio),
                    "webp_artifact_score": float(assessment.signals.webp_artifact_score),
                    "face_box_count": float(assessment.metadata.get("face_box_count", 0)),
                }
            ]
        )
        method_name = str(self.model.predict(feature_row)[0])
        probabilities = self.model.predict_proba(feature_row)[0].tolist()
        class_probabilities = {
            str(label): float(probability)
            for label, probability in zip(self.metadata.classes, probabilities, strict=False)
        }
        return RouterDecision(
            method_name=method_name,
            metadata={
                "policy_version": self.metadata.model_version,
                "feature_names": self.metadata.feature_names,
                "class_probabilities": class_probabilities,
                "blur_score": float(assessment.signals.blur_score),
                "face_size_px": float(assessment.signals.face_size_px),
                "occlusion_ratio": float(assessment.signals.occlusion_ratio),
                "webp_artifact_score": float(assessment.signals.webp_artifact_score),
                "face_box_count": int(assessment.metadata.get("face_box_count", 0)),
            },
        )
