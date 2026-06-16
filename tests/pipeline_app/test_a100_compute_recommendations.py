"""A100 40GB: compute power correctly identified; Recommended is compute-only.

Recommendation must NOT use visual-quality gates. A method is Recommended iff
fits_compute(CUDA + VRAM) says so.
"""

from __future__ import annotations

from privacy_pipeline_app.method_catalog import (
    FACE_ANON_OPTIONS,
    FACE_DETECTION_OPTIONS,
    MULTIMODAL_DETECTION_OPTIONS,
    RESEARCH_FACE_METHOD_IDS,
    compute_badge,
    fits_compute,
    get_option,
    method_choice_options,
    resolve_defaults_for_profile,
)
from privacy_pipeline_app.runtime_policy import select_runtime_policy


def _env_a100_40gb() -> dict:
    return {
        "cuda_available": True,
        "device": "cuda",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "vram_total_mb": 40 * 1024,  # 40960 MB
        "cpu_count": 48,
    }


def test_a100_40gb_runtime_policy_is_accelerated_full() -> None:
    policy = select_runtime_policy(_env_a100_40gb())
    assert policy.policy_id == "accelerated_full"
    assert policy.tier == "accelerated"
    assert policy.face_policy_id == "runtime_3_source_all_raw_rf_approximation"
    assert policy.multimodal_image_size == 1280


def test_a100_40gb_recommends_full_face_fusion() -> None:
    env = _env_a100_40gb()
    full = get_option("face_detection", "runtime_3_source_all_raw_rf_approximation")
    assert full is not None
    assert fits_compute(full, env) is True
    assert compute_badge(full, env) == "Recommended"


def test_a100_40gb_recommends_riddle_and_other_research_methods() -> None:
    """RiDDLE must NOT be Not recommended on every system - only when VRAM insufficient."""
    env = _env_a100_40gb()
    riddle = get_option("face_anonymisation", "riddle")
    assert riddle is not None
    assert riddle.min_vram_mb == 12 * 1024
    assert 40 * 1024 >= riddle.min_vram_mb
    assert compute_badge(riddle, env) == "Recommended"

    for mid in sorted(RESEARCH_FACE_METHOD_IDS):
        opt = get_option("face_anonymisation", mid)
        assert opt is not None
        assert fits_compute(opt, env) is True, f"{mid} should fit A100 40GB"
        assert compute_badge(opt, env) == "Recommended", (
            f"{mid} must be Recommended on A100 40GB (compute-only rule)"
        )


def test_a100_40gb_picker_labels_mark_riddle_recommended_not_visual_gate() -> None:
    env = _env_a100_40gb()
    choices = method_choice_options("face_anonymisation", env)
    by_name = {display: label for label, display in choices}
    assert "RiDDLE" in by_name
    assert "Recommended" in by_name["RiDDLE"]
    assert "Not recommended" not in by_name["RiDDLE"]
    # Labels must not encode visual-gate wording as the badge
    for label, _display in choices:
        assert "visual gate" not in label.lower()
        assert "research-only for defaults" not in label  # badge line is compute only


def test_recommendation_ignores_visual_quality_notes() -> None:
    """Even methods marked research-only in visual_note stay Recommended if VRAM fits."""
    env = _env_a100_40gb()
    for opt in FACE_ANON_OPTIONS:
        if "visual" in opt.visual_note.lower() or "research" in opt.visual_note.lower():
            if opt.needs_cuda and opt.min_vram_mb <= 40 * 1024:
                assert compute_badge(opt, env) == "Recommended"
                # Badge function only uses fits_compute
                assert compute_badge(opt, env) == (
                    "Recommended" if fits_compute(opt, env) else "Not recommended"
                )


def test_a100_40gb_not_identical_to_weak_gpu_for_riddle() -> None:
    weak = {"cuda_available": True, "vram_total_mb": 6 * 1024, "gpu_name": "small"}
    a100 = _env_a100_40gb()
    riddle = get_option("face_anonymisation", "riddle")
    assert compute_badge(riddle, weak) == "Not recommended"
    assert compute_badge(riddle, a100) == "Recommended"


def test_a100_40gb_all_detector_and_mm_options_that_fit_are_recommended() -> None:
    env = _env_a100_40gb()
    for opt in FACE_DETECTION_OPTIONS + MULTIMODAL_DETECTION_OPTIONS:
        if fits_compute(opt, env):
            assert compute_badge(opt, env) == "Recommended"
        else:
            assert compute_badge(opt, env) == "Not recommended"


def test_a100_40gb_auto_defaults_still_visual_safe_for_face_ops() -> None:
    """Auto profile defaults stay visual-safe; badges for heavy methods remain Recommended."""
    env = _env_a100_40gb()
    defaults = resolve_defaults_for_profile("balanced", env)
    assert defaults["face_detection"] == "runtime_3_source_all_raw_rf_approximation"
    assert defaults["face_anonymisation"] in {"solid_mask", "layered", "blur", "pixelate"}
    assert compute_badge(get_option("face_anonymisation", "riddle"), env) == "Recommended"
