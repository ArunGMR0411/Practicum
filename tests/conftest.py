"""Shared pytest bootstrap for thesis and app packages."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_SRC = PROJECT_ROOT / "app" / "src"

for path in (PROJECT_ROOT, APP_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Relative path snippets (posix) for tests that need CASTLE/weights/App inputs.
_E2E_PATH_SNIPPETS = (
    "tests/annotation_review/",
    "tests/anonymisation/test_generative_control_batch.py",
    "tests/core_structure/test_repository_structure.py",
    "tests/data_protocol/",
    "tests/detection/test_screen_detection.py",
    "tests/evaluation/test_ocr_evaluator.py",
    "tests/evaluation/test_reid_evaluation.py",
    "tests/pipeline_app/test_app_contracts_extended.py",
    "tests/pipeline_app/test_assets_required_for_e2e.py",
    "tests/pipeline_app/test_detector_preflight_resilience.py",
    "tests/pipeline_app/test_model_availability_and_compute_profiles.py",
    "tests/pipeline_app/test_pipeline_e2e.py",
    "tests/pipeline_app/test_production_app.py",
    "tests/pipeline_app/test_thesis_face_detector.py",
    "tests/pipeline_app/test_wizard_workflow.py",
    "tests/routing/test_cross_view_subset.py",
    "tests/routing/test_router.py",
)


def _local_e2e_assets_available() -> bool:
    castle = PROJECT_ROOT / "data" / "castle2024" / "raw"
    screen = PROJECT_ROOT / "app" / "models" / "multimodal_screen_yolo11s.pt"
    yolo_face = (
        PROJECT_ROOT / "data" / "models" / "face_detection_candidates" / "yolo11s_widerface.pt"
    )
    try:
        castle_ok = castle.is_dir() and any(castle.iterdir())
    except OSError:
        castle_ok = False
    return castle_ok and screen.is_file() and yolo_face.is_file()


def _is_e2e_path(path: Path) -> bool:
    rel = path.resolve().as_posix()
    root = PROJECT_ROOT.resolve().as_posix().rstrip("/") + "/"
    if rel.startswith(root):
        rel = rel[len(root) :]
    return any(snippet in rel for snippet in _E2E_PATH_SNIPPETS)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e_assets: requires local CASTLE frames, model weights, App inputs, or generated dumps",
    )
    config.addinivalue_line("markers", "public: clean-clone safe (synthetic fixtures)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    assets_ok = _local_e2e_assets_available()
    skip = pytest.mark.skip(
        reason=(
            "e2e_assets not available (CASTLE/weights). "
            "Public suite still runs; see tests/README.md for local full-stack."
        )
    )
    for item in items:
        path = Path(str(item.fspath))
        if _is_e2e_path(path) or "e2e_assets" in item.keywords:
            item.add_marker(pytest.mark.e2e_assets)
            if not assets_ok:
                item.add_marker(skip)
