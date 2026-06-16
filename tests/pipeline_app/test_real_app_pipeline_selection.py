"""Real App selection contracts without multi-model GPU loads.

Heavy live RiDDLE/FALCO GPU runs are intentionally NOT in this file (OOM risk).
See ``test_model_availability_and_compute_profiles.py`` for availability +
compute-profile coverage, and the spy-based invocation test for registry routing.
"""

from __future__ import annotations

from privacy_pipeline_app.method_catalog import (
    compute_badge,
    get_option,
    resolve_defaults_for_profile,
)
from privacy_pipeline_app.runtime_env import configure_app_runtime
from privacy_pipeline_app.runtime_policy import select_runtime_policy
from privacy_pipeline_app.method_catalog import apply_selections_to_plan
from privacy_pipeline_app.wizard_workflow import _select_method


def test_runtime_env_sets_default_backend_paths() -> None:
    snap = configure_app_runtime(force=True)
    assert snap["PROJECT_ROOT"]
    # Paths set automatically when trees exist (no shell source step)
    if snap["RIDDLE_SOURCE_ROOT"]:
        assert "riddle" in snap["RIDDLE_SOURCE_ROOT"]
    if snap["FALCO_SOURCE_ROOT"]:
        assert "falco" in snap["FALCO_SOURCE_ROOT"]


def test_fake_a100_real_catalog_recommends_riddle() -> None:
    env = {
        "cuda_available": True,
        "vram_total_mb": 40 * 1024,
        "gpu_name": "NVIDIA A100-SXM4-40GB",
    }
    assert select_runtime_policy(env).policy_id == "accelerated_full"
    assert compute_badge(get_option("face_anonymisation", "riddle"), env) == "Recommended"
    assert compute_badge(get_option("face_anonymisation", "falco"), env) == "Recommended"
    assert resolve_defaults_for_profile("privacy", env)["face_detection"] == (
        "runtime_3_source_all_raw_rf_approximation"
    )


def test_plan_selection_records_user_riddle_in_real_wizard_selector() -> None:
    env = {"cuda_available": True, "vram_total_mb": 40 * 1024, "gpu_name": "A100"}
    plan = apply_selections_to_plan(
        {"title": "Balanced", "focus": "balanced", "stages": [], "runtime_policy": {}},
        {
            **resolve_defaults_for_profile("balanced", env),
            "face_anonymisation": "riddle",
        },
        env,
    )
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
    assert "RiDDLE" in reason or "riddle" in reason.lower()
