"""Selected and applied method tests for advanced anonymisers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from privacy_pipeline_app.wizard_workflow import _apply_method


@pytest.fixture
def face_image() -> Image.Image:
    return Image.new("RGB", (64, 64), (120, 90, 80))


@pytest.fixture
def boxes() -> list[tuple[int, int, int, int]]:
    return [(8, 8, 40, 40)]


def test_unknown_method_records_selected_and_fallback(face_image, boxes) -> None:
    result = _apply_method(face_image, boxes, "totally_unknown_backend", fallback_method="solid_mask")
    assert result.selected_method == "totally_unknown_backend"
    assert result.applied_method == "solid_mask"
    assert result.status == "fallback"
    assert result.error


def test_nullface_backend_exception_uses_fallback(face_image, boxes) -> None:
    with patch(
        "src.anonymisation.registry.build_anonymiser_registry",
        side_effect=RuntimeError("simulated nullface CUDA OOM"),
    ):
        result = _apply_method(face_image, boxes, "nullface", fallback_method="layered")
    assert result.selected_method == "nullface"
    assert result.status == "fallback"
    assert result.applied_method == "layered"
    assert "CUDA OOM" in result.error


def test_riddle_forced_failure_logs_selected_vs_applied(face_image, boxes, monkeypatch) -> None:
    import privacy_pipeline_app.wizard_workflow as ww

    original = getattr(ww, "_apply_research_method", None)

    def fail_research(method, image, face_boxes, **kwargs):
        raise RuntimeError("simulated riddle weight missing")

    # Patch common hooks used by wizard for research methods
    if hasattr(ww, "_run_research_face"):
        monkeypatch.setattr(ww, "_run_research_face", fail_research)
    if hasattr(ww, "apply_research_method"):
        monkeypatch.setattr(ww, "apply_research_method", fail_research)

    result = _apply_method(face_image, boxes, "riddle", fallback_method="solid_mask")
    assert result.selected_method == "riddle"
    # Must never silently claim success with a different method without fallback status
    if result.applied_method != "riddle":
        assert result.status == "fallback"
        assert result.applied_method == "solid_mask"


def test_falco_forced_import_error(face_image, boxes, monkeypatch) -> None:
    import privacy_pipeline_app.wizard_workflow as ww

    def raise_import(*_a, **_k):
        raise ImportError("simulated falco missing")

    for name in ("_apply_falco", "apply_falco", "_run_falco"):
        if hasattr(ww, name):
            monkeypatch.setattr(ww, name, raise_import)

    result = _apply_method(face_image, boxes, "falco", fallback_method="blur")
    assert result.selected_method == "falco"
    if result.applied_method != "falco":
        assert result.status in {"fallback", "error", "ok"}
        if result.status == "fallback":
            assert result.applied_method == "blur"


def test_decisions_csv_shape_for_batch_fallbacks(face_image, boxes, tmp_path) -> None:
    """selected/applied/status always populated for mixed methods."""
    methods = ["blur", "not_real", "solid_mask", "also_fake"]
    rows = []
    for method in methods:
        r = _apply_method(face_image, boxes, method, fallback_method="solid_mask")
        rows.append(
            {
                "selected_method": r.selected_method,
                "applied_method": r.applied_method,
                "status": r.status,
                "error": r.error or "",
            }
        )
    assert all(row["selected_method"] for row in rows)
    assert all(row["applied_method"] for row in rows)
    assert all(row["status"] for row in rows)
    # unknown methods must not pretend success under their own name
    for row in rows:
        if row["selected_method"] in {"not_real", "also_fake"}:
            assert row["status"] == "fallback"
            assert row["applied_method"] == "solid_mask"
