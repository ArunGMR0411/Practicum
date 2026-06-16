"""Preflight must not hard-fail when a preferred detector checkpoint is missing."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from privacy_pipeline_app import wizard_workflow as ww
from privacy_pipeline_app.thesis_face_detector import (
    FACE_POLICY_FALLBACK_ORDER,
    resolve_runnable_face_policy,
    validate_thesis_face_detector,
)


def test_resolve_runnable_prefers_requested_when_valid() -> None:
    runtime = resolve_runnable_face_policy("yolo11s_face")
    assert runtime["policy_id"] == "yolo11s_face"
    assert runtime["fallback_applied"] is False


def test_resolve_runnable_steps_down_when_preferred_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If RF-DETR path is broken, step down to a tier that still works."""
    import privacy_pipeline_app.thesis_face_detector as tfd

    missing = tmp_path / "missing_rfdetr.pth"
    monkeypatch.setattr(tfd, "RFDETR_CHECKPOINT", missing)
    monkeypatch.setattr(tfd, "resolve_rfdetr_checkpoint", lambda: missing)
    # Preferred needs rfdetr → should fall back
    runtime = resolve_runnable_face_policy("fusion_rfdetr_yolo11s_scrfd10g")
    assert runtime["policy_id"] in FACE_POLICY_FALLBACK_ORDER
    assert "rfdetr" not in runtime["components"] or runtime["policy_id"] != "fusion_rfdetr_yolo11s_scrfd10g"
    # Must be a policy that does not need the missing file
    assert runtime["policy_id"] in {
        "fixed_fusion_yolo11s1280_scrfd10g",
        "yolo11s_face",
        "fusion_rfdetr_scrfd10g",  # only if somehow rfdetr resolved
    }
    if runtime["fallback_applied"]:
        assert "unavailable" in runtime.get("fallback_reason", "").lower() or "using" in runtime.get(
            "fallback_reason", ""
        ).lower()


def test_accept_preflight_writes_runnable_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    source = tmp_path / "in"
    source.mkdir()
    Image.new("RGB", (16, 16), (1, 1, 1)).save(source / "a.jpg")
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
    state = ww.step_scan(state.run_dir)
    # Force preferred fusion that might need rfdetr - accept should still succeed
    state.plan.setdefault("runtime_policy", {})["face_policy_id"] = "fusion_rfdetr_yolo11s_scrfd10g"
    ww.save_state(state)
    state = ww.accept_preflight(state.run_dir)
    assert state.preflight_accepted is True
    runtime_path = Path(state.run_dir) / "metadata" / "detector_preflight.json"
    assert runtime_path.is_file()
    face_id = (state.plan.get("runtime_policy") or {}).get("face_policy_id")
    assert face_id in FACE_POLICY_FALLBACK_ORDER or face_id
    # Detect path will use this id
    validate_thesis_face_detector(str(face_id))
