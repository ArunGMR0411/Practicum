"""Single policy registry must drive App defaults consistently."""

from __future__ import annotations

from privacy_pipeline_app.method_catalog import PROFILE_DEFAULTS, resolve_defaults_for_profile
from privacy_pipeline_app.objective_policy import resolve_plan
from privacy_pipeline_app.runtime_policy import select_runtime_policy
from src.policy.registry import (
    get_app_policy_semantics,
    get_profile_defaults,
    load_policy_registry,
)


def test_registry_file_loads() -> None:
    reg = load_policy_registry()
    assert "app_policy" in reg and "runtime_tiers" in reg
    assert set(reg["app_policy"]["profiles"]) == {"privacy", "balanced", "utility"}


def test_profile_defaults_match_objective_plan_and_catalog() -> None:
    for focus in ("privacy", "balanced", "utility"):
        reg_defaults = get_profile_defaults(focus)
        plan = resolve_plan(focus)
        assert plan.face_detection.method_id == reg_defaults["face_detection"]
        assert plan.face_anonymisation.method_id == reg_defaults["face_anonymisation"]
        assert plan.screen_operator.method_id == reg_defaults["screen_operator"]
        assert plan.text_operator.method_id == reg_defaults["text_operator"]
        assert PROFILE_DEFAULTS[focus] == reg_defaults


def test_a100_runtime_tier_uses_registry_primary_detector() -> None:
    env = {"cuda_available": True, "vram_total_mb": 40 * 1024}
    policy = select_runtime_policy(env)
    assert policy.policy_id == "accelerated_full"
    assert policy.face_policy_id == "runtime_3_source_all_raw_rf_approximation"


def test_app_policy_semantics_are_honest() -> None:
    sem = get_app_policy_semantics()
    assert sem["app_policy_id"] == "objective_profile"
    assert sem["scientific_policy_id"] == "oapr_visual_safe_balanced_500"
    assert any("286" in s or "condition-aware" in s for s in sem["simplification"])


def test_resolve_defaults_for_profile_balanced_uses_registry() -> None:
    env = {"cuda_available": True, "vram_total_mb": 40 * 1024}
    defaults = resolve_defaults_for_profile("balanced", env)
    assert defaults["face_anonymisation"] == "layered"
    assert defaults["face_detection"] == "runtime_3_source_all_raw_rf_approximation"
