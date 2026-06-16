"""Evidence-mapped Privacy / Balanced / Utility objective plans."""

from __future__ import annotations

from privacy_pipeline_app.objective_policy import PLANS, resolve_plan


def test_resolve_plan_aliases() -> None:
    assert resolve_plan("privacy").objective_id == "privacy_first"
    assert resolve_plan("Privacy").objective_id == "privacy_first"
    assert resolve_plan("balanced").objective_id == "utility_under_privacy_floor"
    assert resolve_plan("utility").objective_id == "utility_priority"
    assert resolve_plan("unknown").objective_id == "utility_under_privacy_floor"


def test_privacy_plan_uses_solid_mask_and_fill_operators() -> None:
    plan = resolve_plan("privacy")
    assert plan.face_anonymisation.method_id == "solid_mask"
    assert plan.screen_operator.method_id == "fill"
    assert plan.text_operator.method_id == "fill"


def test_balanced_plan_default_face_and_multimodal() -> None:
    plan = resolve_plan("balanced")
    assert plan.objective_id == "utility_under_privacy_floor"
    # Authoritative: configs/policy_registry.json
    assert plan.face_anonymisation.method_id == "layered"
    assert plan.screen_operator.method_id == "fill"
    assert plan.text_operator.method_id == "blur"


def test_utility_plan_prefers_blur_and_pixelate_text() -> None:
    plan = resolve_plan("utility")
    assert plan.face_anonymisation.method_id == "blur"
    assert plan.screen_operator.method_id == "blur"
    assert plan.text_operator.method_id == "pixelate"


def test_plan_to_dict_records_policy_semantics() -> None:
    payload = resolve_plan("balanced").to_dict()
    assert payload["app_policy_id"] == "objective_profile"
    assert payload["scientific_policy_id"] == "oapr_visual_safe_balanced_500"
    assert payload["simplification"]


def test_all_plans_share_promoted_detector_ids() -> None:
    for key in ("privacy", "balanced", "utility"):
        plan = resolve_plan(key)
        mid = plan.face_detection.method_id
        assert (
            "runtime_3_source" in mid
            or "error_hardened" in mid
            or "fusion" in mid
            or "yolo" in mid
        ), mid
        assert plan.multimodal_detection.method_id
        assert len(plan.all_stages()) == 5


def test_plan_to_dict_includes_stage_evidence() -> None:
    payload = resolve_plan("balanced").to_dict()
    assert payload["focus"] == "balanced"
    assert payload["stages"]
    assert all("evidence" in stage for stage in payload["stages"])
    assert "privacy" in PLANS and "balanced" in PLANS and "utility" in PLANS
