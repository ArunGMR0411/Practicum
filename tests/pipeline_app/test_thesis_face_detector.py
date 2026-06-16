"""Thesis face-detector policy tables and geometry helpers (no GPU load)."""

from __future__ import annotations

import pytest

from privacy_pipeline_app.thesis_face_detector import (
    POLICY_COMPONENTS,
    FUSION_IOU,
    RFDETR_THRESHOLD,
    YOLO_THRESHOLD,
    SCRFD_THRESHOLD,
    _iou,
    validate_thesis_face_detector,
)


def test_policy_component_sets() -> None:
    assert POLICY_COMPONENTS["runtime_3_source_all_raw_rf_approximation"] == ("rfdetr", "yolo", "scrfd")
    assert POLICY_COMPONENTS["fusion_rfdetr_yolo11s_scrfd10g"] == ("rfdetr", "yolo", "scrfd")
    assert POLICY_COMPONENTS["fusion_rfdetr_scrfd10g"] == ("rfdetr", "scrfd")
    assert POLICY_COMPONENTS["fixed_fusion_yolo11s1280_scrfd10g"] == ("yolo", "scrfd")
    assert POLICY_COMPONENTS["yolo11s_face"] == ("yolo",)


def test_fusion_thresholds_match_evidence_stack() -> None:
    assert RFDETR_THRESHOLD == 0.30
    assert YOLO_THRESHOLD == 0.25
    assert SCRFD_THRESHOLD == 0.25
    assert FUSION_IOU == 0.50


def test_iou_geometry() -> None:
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.0 < _iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1.0


def test_validate_unknown_policy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown face-detector policy"):
        validate_thesis_face_detector("not_a_real_policy")


def test_validate_yolo_only_policy_when_checkpoint_present() -> None:
    """Portable policy only needs YOLO weights + ultralytics; skip if missing."""
    from privacy_pipeline_app.thesis_face_detector import YOLO11_FACE_CHECKPOINT

    if not YOLO11_FACE_CHECKPOINT.exists():
        pytest.skip("YOLO face checkpoint not available in this environment")
    runtime = validate_thesis_face_detector("yolo11s_face")
    assert runtime["policy_id"] == "yolo11s_face"
    assert runtime["components"] == ["yolo"]
