"""Extended App contracts: setup/plan consistency, risk gate, ETA, review, done, mid-tier VRAM."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from privacy_pipeline_app import production_app
from privacy_pipeline_app import wizard_workflow as ww
from privacy_pipeline_app.method_catalog import (
    any_not_recommended,
    compute_badge,
    get_option,
    resolve_defaults_for_profile,
)
from privacy_pipeline_app.runtime_policy import estimate_runtime, select_runtime_policy


def test_mid_tier_8gb_uses_efficient_not_full_fusion() -> None:
    env = {"cuda_available": True, "vram_total_mb": 8 * 1024, "gpu_name": "RTX 4060"}
    policy = select_runtime_policy(env)
    assert policy.policy_id == "accelerated_efficient"
    assert policy.face_policy_id == "fusion_rfdetr_scrfd10g"
    assert compute_badge(get_option("face_detection", "fusion_rfdetr_yolo11s_scrfd10g"), env) == "Not recommended"
    assert compute_badge(get_option("face_detection", "fusion_rfdetr_scrfd10g"), env) == "Recommended"


def test_create_run_setup_face_method_matches_plan_face_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: setup.json face_method must match objective plan face stage."""
    source = tmp_path / "in"
    source.mkdir()
    Image.new("RGB", (16, 16), (1, 1, 1)).save(source / "a.jpg")
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(ww, "APP_RUNS", runs)
    monkeypatch.setattr(
        ww,
        "probe_environment",
        lambda: {
            "cuda_available": True,
            "vram_total_mb": 8192,
            "gpu_name": "Test",
            "device": "cuda",
            "cpu_count": 4,
        },
    )
    state = ww.create_run(str(source), False, "objective_profile", "layered", "balanced", True)
    setup = json.loads((Path(state.run_dir) / "metadata" / "setup.json").read_text())
    plan = json.loads((Path(state.run_dir) / "metadata" / "objective_plan.json").read_text())
    face_stages = [s for s in plan.get("stages", []) if s.get("stage") == "Face anonymisation"]
    # Setup and plan use the same face method.
    if face_stages:
        assert setup.get("face_method") == face_stages[0].get("method_id")
    assert setup.get("face_method") == state.fixed_method


def test_risk_gate_blocks_when_not_recommended_without_confirm(monkeypatch) -> None:
    state = SimpleNamespace(
        focus="balanced",
        n_images=10,
        source_dir="app/inputs",
        plan={
            "user_method_selections": {
                "face_detection": "runtime_3_source_all_raw_rf_approximation",
                "multimodal_detection": "reviewed_screen_yolo11s_1280",
                "face_anonymisation": "riddle",
                "screen_operator": "fill",
                "text_operator": "blur",
            }
        },
    )
    monkeypatch.setattr(production_app, "load_state", lambda _: state)
    monkeypatch.setattr(
        production_app,
        "probe_environment",
        lambda: {"cuda_available": True, "vram_total_mb": 4096},
    )
    monkeypatch.setattr(production_app, "_stats_html", lambda _: "stats")
    monkeypatch.setattr(production_app, "_dashboard_for_state", lambda *_a, **_k: "dash")
    result = production_app.ui_proceed("run", risk_ok=False)
    assert result[1] == "preflight"
    assert any_not_recommended(state.plan["user_method_selections"], {"cuda_available": True, "vram_total_mb": 4096})


def test_eta_prefers_measured_over_tier_estimate(tmp_path: Path) -> None:
    policy = select_runtime_policy({"cuda_available": True, "vram_total_mb": 24 * 1024})
    run = tmp_path / "r1"
    (run / "metadata").mkdir(parents=True)
    (run / "state.json").write_text(
        json.dumps(
            {
                "plan": {"runtime_policy_id": policy.policy_id},
                "detect_summary": {"n_images": 20, "runtime_seconds": 100, "n_errors": 0},
            }
        ),
        encoding="utf-8",
    )
    (run / "metadata" / "system_profile.json").write_text(
        json.dumps({"gpu_name": "A100"}), encoding="utf-8"
    )
    est = estimate_runtime(40, policy, {"gpu_name": "A100"}, tmp_path)
    assert est.seconds_per_image == 5.0
    assert est.total_seconds == 200
    assert "measured" in est.source


def test_accept_preflight_required_before_detect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(ww, "APP_RUNS", runs)
    source = tmp_path / "in"
    source.mkdir()
    Image.new("RGB", (8, 8)).save(source / "x.jpg")
    monkeypatch.setattr(
        ww,
        "probe_environment",
        lambda: {"cuda_available": False, "vram_total_mb": 0, "device": "cpu", "gpu_name": "n", "cpu_count": 2},
    )
    state = ww.create_run(str(source), False, "objective_profile", "layered", "utility", True)
    state = ww.step_scan(state.run_dir)
    with pytest.raises(RuntimeError, match="preflight"):
        ww.step_detect(state.run_dir)


def test_zero_faces_selects_copy() -> None:
    method, reason = ww._select_method("objective_profile", "layered", "privacy_first", 0, 0, 0, None)
    assert method == "copy"
    assert "No face" in reason


def test_unknown_method_fallback_logs_selected_vs_applied() -> None:
    image = Image.new("RGB", (20, 20), (0, 0, 0))
    result = ww._apply_method(image, [(1, 1, 10, 10)], "totally_unknown", fallback_method="solid_mask")
    assert result.status == "fallback"
    assert result.selected_method == "totally_unknown"
    assert result.applied_method == "solid_mask"


def test_objective_switch_remaps_operators() -> None:
    from privacy_pipeline_app.objective_policy import resolve_plan

    p = resolve_plan("privacy")
    u = resolve_plan("utility")
    assert p.face_anonymisation.method_id == "solid_mask"
    assert u.face_anonymisation.method_id == "blur"
    assert p.screen_operator.method_id == "fill"
    assert u.screen_operator.method_id == "blur"
    assert u.text_operator.method_id == "pixelate"


def test_done_gallery_allow_preview_false_in_source() -> None:
    import inspect
    from privacy_pipeline_app import production_app as mod

    assert "allow_preview=False" in inspect.getsource(mod.build_app)


def test_refresh_state_after_detection_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(ww, "APP_RUNS", runs)
    source = tmp_path / "in"
    source.mkdir()
    Image.new("RGB", (12, 12)).save(source / "a.jpg")
    monkeypatch.setattr(
        ww,
        "probe_environment",
        lambda: {"cuda_available": False, "vram_total_mb": 0, "device": "cpu", "gpu_name": "n", "cpu_count": 2},
    )
    state = ww.create_run(str(source), False, "objective_profile", "blur", "utility", True)
    state = ww.step_scan(state.run_dir)
    # Manual detection with one face
    rows = list(csv.DictReader((Path(state.run_dir) / "input_manifest.csv").open()))
    rec = {
        "image_id": rows[0]["image_id"],
        "local_path": rows[0]["local_path"],
        "detector": "manual",
        "faces": [{"x1": 1, "y1": 1, "x2": 5, "y2": 5, "score": 1.0}],
        "screens": [],
        "texts": [],
        "screen_sources": [],
        "text_policy": "none",
    }
    ww.write_detection_artifacts(Path(state.run_dir) / "detections", [rec])
    state.stages_done["detect"] = True
    state.detect_summary = {"n_images": 1}
    ww.save_state(state)
    state = ww.refresh_state_from_detections(state.run_dir)
    assert state.n_faces == 1
    assert state.detect_summary.get("reviewed") is True
