"""Selection + Recommended badges for research face methods and hardware profiles."""

from __future__ import annotations

from privacy_pipeline_app.method_catalog import (
    FACE_ANON_OPTIONS,
    RESEARCH_FACE_METHOD_IDS,
    STAGE_KEYS,
    STAGE_OPTIONS,
    VISUAL_SAFE_FACE_METHOD_IDS,
    compute_badge,
    fits_compute,
    get_option,
    method_id_from_display_name,
    resolve_defaults_for_profile,
)
from privacy_pipeline_app.runtime_policy import select_runtime_policy
from privacy_pipeline_app.wizard_workflow import _apply_method, _select_method
from PIL import Image


def _env_h100() -> dict:
    return {
        "cuda_available": True,
        "device": "cuda",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "vram_total_mb": 80 * 1024,
        "cpu_count": 64,
    }


def _env_a100() -> dict:
    return {
        "cuda_available": True,
        "device": "cuda",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "vram_total_mb": 40 * 1024,
        "cpu_count": 48,
    }


def _env_cpu_only() -> dict:
    return {
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": "not_available",
        "vram_total_mb": 0,
        "cpu_count": 8,
    }


def test_research_face_methods_are_in_face_anonymisation_catalog() -> None:
    ids = {opt.method_id for opt in FACE_ANON_OPTIONS}
    for mid in (
        "nullface",
        "diffusion",
        "riddle",
        "falco",
        "fams",
        "reverse_personalization",
        "stylegan",
    ):
        assert mid in ids, f"missing face method {mid}"
        assert mid in RESEARCH_FACE_METHOD_IDS


def test_display_name_roundtrip_for_riddle_and_falco() -> None:
    assert method_id_from_display_name("face_anonymisation", "RiDDLE") == "riddle"
    assert method_id_from_display_name("face_anonymisation", "FALCO") == "falco"
    assert method_id_from_display_name("face_anonymisation", "Diffusion (low-step)") == "diffusion"


def test_h100_recommends_full_detector_and_research_face_methods() -> None:
    env = _env_h100()
    full = get_option("face_detection", "fusion_rfdetr_yolo11s_scrfd10g")
    riddle = get_option("face_anonymisation", "riddle")
    falco = get_option("face_anonymisation", "falco")
    assert full is not None and riddle is not None and falco is not None
    assert compute_badge(full, env) == "Recommended"
    assert compute_badge(riddle, env) == "Recommended"
    assert compute_badge(falco, env) == "Recommended"
    assert compute_badge(get_option("face_anonymisation", "nullface"), env) == "Recommended"


def test_a100_recommends_heavy_stack() -> None:
    env = _env_a100()
    assert compute_badge(get_option("face_detection", "fusion_rfdetr_yolo11s_scrfd10g"), env) == "Recommended"
    assert compute_badge(get_option("multimodal_detection", "reviewed_screen_yolo11s_1280"), env) == "Recommended"
    assert compute_badge(get_option("face_anonymisation", "riddle"), env) == "Recommended"
    assert compute_badge(get_option("face_anonymisation", "diffusion"), env) == "Recommended"


def test_h100_auto_defaults_pick_strongest_detectors_not_generative_defaults() -> None:
    """Powerful GPU: profile defaults keep visual-safe face ops; detectors at profile max."""
    env = _env_h100()
    expected_face_det = {
        "privacy": "runtime_3_source_all_raw_rf_approximation",
        "balanced": "runtime_3_source_all_raw_rf_approximation",
        "utility": "fixed_fusion_yolo11s1280_scrfd10g",
    }
    expected_mm = {
        "privacy": "reviewed_screen_yolo11s_1280",
        "balanced": "reviewed_screen_yolo11s_1280",
        "utility": "reviewed_screen_yolo11s_960",
    }
    for focus in ("privacy", "balanced", "utility"):
        defaults = resolve_defaults_for_profile(focus, env)
        assert defaults["face_detection"] == expected_face_det[focus]
        assert defaults["multimodal_detection"] == expected_mm[focus]
        # Generative methods Available+Recommended on H100 but not auto profile defaults
        assert defaults["face_anonymisation"] in VISUAL_SAFE_FACE_METHOD_IDS
        assert defaults["face_anonymisation"] not in RESEARCH_FACE_METHOD_IDS
        assert defaults["screen_operator"] in {"fill", "blur", "pixelate"}
        assert defaults["text_operator"] in {"fill", "blur", "pixelate"}
        # Full fusion remains selectable on H100.
        assert (
            compute_badge(get_option("face_detection", "fusion_rfdetr_yolo11s_scrfd10g"), env)
            == "Recommended"
        )


def test_h100_runtime_policy_matches_accelerated_full() -> None:
    policy = select_runtime_policy(_env_h100())
    assert policy.policy_id == "accelerated_full"
    assert policy.face_policy_id == "runtime_3_source_all_raw_rf_approximation"


def test_cpu_only_recommends_light_paths_and_not_research() -> None:
    env = _env_cpu_only()
    assert compute_badge(get_option("face_detection", "yolo11s_face"), env) == "Recommended"
    assert compute_badge(get_option("face_detection", "fusion_rfdetr_yolo11s_scrfd10g"), env) == "Not recommended"
    assert compute_badge(get_option("multimodal_detection", "reviewed_screen_yolo11s_640"), env) == "Recommended"
    assert compute_badge(get_option("multimodal_detection", "reviewed_screen_yolo11s_1280"), env) == "Not recommended"
    for mid in RESEARCH_FACE_METHOD_IDS:
        opt = get_option("face_anonymisation", mid)
        assert opt is not None
        assert compute_badge(opt, env) == "Not recommended"
    for mid in ("solid_mask", "layered", "blur", "pixelate"):
        assert compute_badge(get_option("face_anonymisation", mid), env) == "Recommended"


def test_cpu_only_auto_defaults_are_portable_and_light() -> None:
    env = _env_cpu_only()
    for focus in ("privacy", "balanced", "utility"):
        defaults = resolve_defaults_for_profile(focus, env)
        assert set(defaults) == set(STAGE_KEYS)
        assert defaults["face_detection"] == "yolo11s_face"
        assert "640" in defaults["multimodal_detection"]
        assert defaults["face_anonymisation"] in VISUAL_SAFE_FACE_METHOD_IDS
        assert defaults["face_anonymisation"] not in RESEARCH_FACE_METHOD_IDS
        # Operators are all light
        for key in ("screen_operator", "text_operator"):
            opt = get_option(key, defaults[key])
            assert opt is not None
            assert opt.needs_cuda is False
            assert fits_compute(opt, env)


def test_cpu_only_runtime_policy_is_portable() -> None:
    policy = select_runtime_policy(_env_cpu_only())
    assert policy.policy_id == "portable_cpu"
    assert policy.face_policy_id == "yolo11s_face"


def test_user_can_select_riddle_and_plan_records_it() -> None:
    plan = {
        "title": "Balanced",
        "stages": [
            {
                "stage": "Face anonymisation",
                "method_id": "riddle",
                "display_name": "RiDDLE",
                "why": "user stretch",
                "recommendation": "Recommended",
            }
        ],
    }
    method, reason = _select_method(
        "objective_profile",
        "layered",
        "utility_under_privacy_floor",
        face_count=1,
        text_count=0,
        screen_count=0,
        plan=plan,
    )
    assert method == "riddle"
    assert "RiDDLE" in reason


def test_apply_riddle_without_assets_is_honest_fallback() -> None:
    """Without RiDDLE assets, App must not silently claim generative success."""
    image = Image.new("RGB", (48, 48), (9, 9, 9))
    result = _apply_method(image, [(4, 4, 24, 24)], "riddle", fallback_method="solid_mask")
    assert result.selected_method == "riddle"
    # Installed assets → ok; missing assets/source → fallback solid_mask
    if result.status == "ok":
        assert result.applied_method == "riddle"
    else:
        assert result.status == "fallback"
        assert result.applied_method == "solid_mask"
        assert result.error


def test_all_catalog_face_methods_resolvable_by_id() -> None:
    for opt in STAGE_OPTIONS["face_anonymisation"]:
        assert get_option("face_anonymisation", opt.method_id) is opt
