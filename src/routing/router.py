"""Rule-based routing helpers for adaptive anonymisation experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.anonymisation.registry import build_anonymiser_registry
from src.anonymisation.unavailable_anonymiser import UnavailableAnonymiser
from src.routing.quality_assessor import QualityAssessment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTING_CALIBRATION = PROJECT_ROOT / "outputs" / "routing_calibration.json"


@dataclass(frozen=True)
class RouterDecision:
    """One routing decision with method name and supporting metadata."""

    method_name: str
    metadata: dict[str, Any]


def load_routing_calibration(path: str | Path = DEFAULT_ROUTING_CALIBRATION) -> dict[str, Any]:
    calibration_path = Path(path)
    with calibration_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class RuleBasedRouter:
    """Choose an anonymisation method from calibrated quality signals.

    Current policy status:
    - primary validated thresholds: `face_size_px`, `blur_score`
    - current robust methods: `blur`, `pixelate`
    - `occlusion_ratio` and generative branches remain extension points until validated
    """

    def __init__(
        self,
        routing_calibration_path: str | Path = DEFAULT_ROUTING_CALIBRATION,
    ) -> None:
        calibration = load_routing_calibration(routing_calibration_path)
        best = calibration["best_thresholds"]
        self.face_size_threshold_px = float(best["face_size_threshold_px"])
        self.blur_threshold = float(best["blur_threshold"])
        self.registry = build_anonymiser_registry()

    def _is_available(self, method_name: str) -> bool:
        anonymiser = self.registry.get(method_name)
        return anonymiser is not None and not isinstance(anonymiser, UnavailableAnonymiser)

    def decide(self, assessment: QualityAssessment) -> RouterDecision:
        signals = assessment.signals
        metadata = {
            "face_size_px": float(signals.face_size_px),
            "blur_score": float(signals.blur_score),
            "occlusion_ratio": float(signals.occlusion_ratio),
            "webp_artifact_score": float(signals.webp_artifact_score),
            "face_size_threshold_px": self.face_size_threshold_px,
            "blur_threshold": self.blur_threshold,
            "policy_version": "rule_based",
        }

        # Current evidence favors blur for small faces and blurrier frames.
        if float(signals.face_size_px) <= self.face_size_threshold_px:
            return RouterDecision("blur", {**metadata, "route_reason": "small_or_medium_face"})

        if float(signals.blur_score) >= self.blur_threshold:
            return RouterDecision("blur", {**metadata, "route_reason": "high_blur"})

        if self._is_available("pixelate"):
            return RouterDecision("pixelate", {**metadata, "route_reason": "sharper_large_face"})

        return RouterDecision("blur", {**metadata, "route_reason": "fallback_blur"})
