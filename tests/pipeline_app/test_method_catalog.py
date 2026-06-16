"""Preflight method catalog: compute-only badges and profile step-down."""

from __future__ import annotations

from privacy_pipeline_app.method_catalog import (
    STAGE_KEYS,
    any_not_recommended,
    apply_selections_to_plan,
    compute_badge,
    fits_compute,
    get_option,
    method_id_from_display_name,
    normalize_focus,
    resolve_defaults_for_profile,
)


def test_normalize_focus() -> None:
    assert normalize_focus("Privacy-first") == "privacy"
    assert normalize_focus("utility_priority") == "utility"
    assert normalize_focus("Balanced") == "balanced"


def test_recommended_badge_is_compute_only() -> None:
    env_small = {"cuda_available": True, "vram_total_mb": 4000}
    full = get_option("face_detection", "runtime_3_source_all_raw_rf_approximation")
    yolo = get_option("face_detection", "yolo11s_face")
    assert full is not None and yolo is not None
    assert fits_compute(full, env_small) is False
    assert compute_badge(full, env_small) == "Not recommended"
    assert fits_compute(yolo, env_small) is True
    assert compute_badge(yolo, env_small) == "Recommended"


def test_cpu_only_rejects_cuda_detectors() -> None:
    env = {"cuda_available": False, "vram_total_mb": 0}
    opt = get_option("face_detection", "fusion_rfdetr_scrfd10g")
    assert opt is not None
    assert fits_compute(opt, env) is False
    assert compute_badge(get_option("face_detection", "yolo11s_face"), env) == "Recommended"


def test_profile_defaults_step_down_when_compute_is_weak() -> None:
    env = {"cuda_available": False, "vram_total_mb": 0}
    defaults = resolve_defaults_for_profile("privacy", env)
    assert set(defaults) == set(STAGE_KEYS)
    assert defaults["face_detection"] == "yolo11s_face"
    assert "640" in defaults["multimodal_detection"] or defaults["multimodal_detection"].endswith("640")
    # Operators stay light
    assert defaults["face_anonymisation"] in {"solid_mask", "layered", "blur", "pixelate"}


def test_profile_defaults_keep_heavy_when_vram_allows() -> None:
    env = {"cuda_available": True, "vram_total_mb": 16 * 1024}
    defaults = resolve_defaults_for_profile("privacy", env)
    assert defaults["face_detection"] == "runtime_3_source_all_raw_rf_approximation"
    assert defaults["multimodal_detection"] == "reviewed_screen_yolo11s_1280"


def test_any_not_recommended_lists_stretch_methods() -> None:
    env = {"cuda_available": True, "vram_total_mb": 4096}
    selections = {
        "face_detection": "runtime_3_source_all_raw_rf_approximation",
        "multimodal_detection": "reviewed_screen_yolo11s_640",
        "face_anonymisation": "solid_mask",
        "screen_operator": "fill",
        "text_operator": "fill",
    }
    risky = any_not_recommended(selections, env)
    assert risky
    assert any("Face detection" in item for item in risky)


def test_apply_selections_to_plan_records_user_choices() -> None:
    env = {"cuda_available": True, "vram_total_mb": 8192}
    plan = {
        "focus": "balanced",
        "title": "Balanced",
        "stages": [],
        "runtime_policy": {"policy_id": "accelerated_efficient"},
    }
    selections = resolve_defaults_for_profile("balanced", env)
    out = apply_selections_to_plan(plan, selections, env)
    assert out["user_method_selections"] == selections
    assert len(out["stages"]) == 5
    assert all(stage.get("user_selected") for stage in out["stages"])
    assert out["runtime_policy"]["face_policy_id"] == selections["face_detection"]


def test_method_id_from_display_name_roundtrip() -> None:
    mid = method_id_from_display_name("face_anonymisation", "Solid mask")
    assert mid == "solid_mask"
    assert method_id_from_display_name("face_anonymisation", "does-not-exist") is None
