#!/usr/bin/env python3
"""Evidence-backed method plans for Privacy / Balanced / Utility.

Stage method IDs come from the authoritative ``configs/policy_registry.json``
via ``src.policy.registry``. Narrative evidence strings remain here for UI copy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

from src.policy.registry import get_app_policy_semantics, get_profile


ROOT = Path(__file__).resolve().parents[3]
APP_RUNS = ROOT / "app" / "outputs" / "runs"


@dataclass(frozen=True)
class StagePlan:
    """One pipeline stage under a chosen objective."""

    stage: str
    method_id: str
    display_name: str
    why: str
    evidence: str
    eta_seconds_per_image: float


@dataclass(frozen=True)
class ObjectivePlan:
    focus: str  # privacy | balanced | utility
    objective_id: str  # privacy_first | utility_under_privacy_floor | utility_priority
    title: str
    summary: str
    face_detection: StagePlan
    multimodal_detection: StagePlan
    face_anonymisation: StagePlan
    screen_operator: StagePlan
    text_operator: StagePlan

    def all_stages(self) -> list[StagePlan]:
        return [
            self.face_detection,
            self.multimodal_detection,
            self.face_anonymisation,
            self.screen_operator,
            self.text_operator,
        ]

    def to_dict(self) -> dict[str, Any]:
        semantics = get_app_policy_semantics()
        return {
            "focus": self.focus,
            "objective_id": self.objective_id,
            "title": self.title,
            "summary": self.summary,
            "stages": [asdict(s) for s in self.all_stages()],
            "app_policy_id": semantics["app_policy_id"],
            "scientific_policy_id": semantics["scientific_policy_id"],
            "simplification": semantics["simplification"],
        }


# UI narrative keyed by method_id (not authoritative for which method is chosen).
_FACE_DET_META = {
    "runtime_3_source_all_raw_rf_approximation": (
        "Three-source hardened RF runtime approximation",
        "Bounded App runtime tier using RF-DETR, YOLO11Face and SCRFD candidates.",
        "No scientific detector score is assigned; the seven-source RQ1 result remains separate.",
        1.5,
    ),
    "fixed_fusion_yolo11s1280_scrfd10g": (
        "YOLO11Face + SCRFD",
        "Lower-memory face detector tier for utility-first / compact hardware profiles.",
        "Supporting compact fusion tier; scientific and App detector evidence remain separate.",
        1.0,
    ),
    "fusion_rfdetr_yolo11s_scrfd10g": (
        "RF-DETR + YOLO11Face + SCRFD (supporting)",
        "Supporting high-recall multi-detector fusion.",
        "Supporting evidence only; not the adopted primary.",
        1.4,
    ),
}

_MM_DET_META = {
    "reviewed_screen_yolo11s_1280": (
        "Screen YOLO 1280 + text OCR",
        "Reviewed screen detector scale used for full/accelerated App tiers.",
        "App multimodal path; scientific locked stack is separate (YOLO union + CRAFT 4K).",
        2.0,
    ),
    "reviewed_screen_yolo11s_960": (
        "Screen YOLO 960 + text OCR",
        "Compact reviewed screen scale for utility / lower-memory tiers.",
        "App multimodal compact tier.",
        1.6,
    ),
    "precision_screen_recognised_text": (
        "Precision screen detector + recognised text OCR",
        "Suppresses low-confidence region proposals before redaction.",
        "Compatibility alias for the multimodal option.",
        2.0,
    ),
}

_FACE_ANON_META = {
    "solid_mask": (
        "Solid mask (black)",
        "Strongest deployable Re-ID suppression among visual-safe methods.",
        "Deployment: solid_mask_black = privacy-first terminal action.",
        0.08,
    ),
    "layered": (
        "Layered (blur + downscale + noise)",
        "Balanced visual-safe default among eligible deterministic methods.",
        "Scientific visual-safe OAPR uses layered heavily (286/500 routes).",
        0.12,
    ),
    "blur": (
        "Blur",
        "Practical utility-oriented visual-safe face operator.",
        "Eligible deterministic fallback.",
        0.08,
    ),
}

_OP_META = {
    "fill": ("Solid fill", "Strong content wipe.", "Multimodal fill operator.", 0.02),
    "blur": ("Blur", "Moderate multimodal redaction.", "Multimodal blur operator.", 0.03),
    "pixelate": ("Pixelate", "Light multimodal redaction.", "Multimodal pixelate operator.", 0.02),
}


def _stage(stage: str, method_id: str, meta_map: dict[str, tuple[str, str, str, float]]) -> StagePlan:
    display, why, evidence, eta = meta_map.get(
        method_id,
        (method_id, "Selected from policy registry.", "configs/policy_registry.json", 0.1),
    )
    return StagePlan(
        stage=stage,
        method_id=method_id,
        display_name=display,
        why=why,
        evidence=evidence,
        eta_seconds_per_image=eta,
    )


def _build_plan(focus: str) -> ObjectivePlan:
    profile = get_profile(focus)
    return ObjectivePlan(
        focus=str(profile["focus"]),
        objective_id=str(profile["objective_id"]),
        title=str(profile["title"]),
        summary=(
            f"{profile['title']} objective profile (App policy id=`objective_profile`). "
            "Not the scientific condition-aware OAPR 286/81/133 materialisation."
        ),
        face_detection=_stage("Face detection", str(profile["face_detection"]), _FACE_DET_META),
        multimodal_detection=_stage(
            "Screen / text detection",
            str(profile["multimodal_detection"]),
            _MM_DET_META,
        ),
        face_anonymisation=_stage(
            "Face anonymisation",
            str(profile["face_anonymisation"]),
            _FACE_ANON_META,
        ),
        screen_operator=_stage("Screen redaction", str(profile["screen_operator"]), _OP_META),
        text_operator=_stage("Text redaction", str(profile["text_operator"]), _OP_META),
    )


def resolve_plan(focus: str) -> ObjectivePlan:
    key = (focus or "balanced").strip().lower()
    if key in {"privacy", "privacy-focused", "privacy_first"}:
        return _build_plan("privacy")
    if key in {"utility", "utility-focused", "utility_priority"}:
        return _build_plan("utility")
    return _build_plan("balanced")


# Back-compat names used in tests
_PRIVACY = resolve_plan("privacy")
_BALANCED = resolve_plan("balanced")
_UTILITY = resolve_plan("utility")
PLANS: dict[str, ObjectivePlan] = {
    "privacy": _PRIVACY,
    "balanced": _BALANCED,
    "utility": _UTILITY,
    "Privacy": _PRIVACY,
    "Balanced": _BALANCED,
    "Utility": _UTILITY,
}


def plan_cards_html(plan: ObjectivePlan) -> str:
    """Optional HTML cards for preflight (kept for UI helpers)."""
    blocks = []
    for stage in plan.all_stages():
        blocks.append(
            "<div class='plan-card'>"
            f"<strong>{escape(stage.stage)}</strong><br/>"
            f"{escape(stage.display_name)}<br/>"
            f"<span class='muted'>{escape(stage.why)}</span>"
            "</div>"
        )
    return "\n".join(blocks)
