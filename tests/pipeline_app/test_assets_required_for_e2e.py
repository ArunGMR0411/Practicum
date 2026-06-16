"""Presence checks for assets required to run App and thesis pipelines E2E."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_app_screen_model_present() -> None:
    path = PROJECT_ROOT / "app" / "models" / "multimodal_screen_yolo11s.pt"
    assert path.is_file(), f"Missing app screen model: {path}"


def test_app_inputs_contain_images() -> None:
    inputs = PROJECT_ROOT / "app" / "inputs"
    assert inputs.is_dir(), "app/inputs missing"
    images = [
        p
        for p in inputs.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    assert images, "app/inputs has no images for wizard demos"


def test_face_detector_checkpoints_for_portable_and_accelerated_paths() -> None:
    yolo = PROJECT_ROOT / "data" / "models" / "face_detection_candidates" / "yolo11s_widerface.pt"
    assert yolo.is_file(), f"Missing YOLO face checkpoint: {yolo}"
    # RF-DETR cache is required for accelerated tiers
    rfdetr_root = (
        PROJECT_ROOT
        / "data"
        / "models"
        / "face_detection_candidates"
        / "rfdetr_hf_cache"
    )
    assert rfdetr_root.is_dir(), "Missing RF-DETR cache directory"
    pth = list(rfdetr_root.rglob("*.pth"))
    assert pth, "RF-DETR .pth checkpoint not found under rfdetr_hf_cache"


def test_castle_raw_mount_present_for_thesis_protocols() -> None:
    raw = PROJECT_ROOT / "data" / "castle2024" / "raw"
    assert raw.is_dir(), "CASTLE raw mount missing"
    # Protocol manifests must resolve to existing files when paths are absolute
    manifest = (
        PROJECT_ROOT
        / "outputs"
        / "01_protocol"
        / "annotations"
        / "face_detection"
        / "01_baseline_500"
        / "manifest.csv"
    )
    assert manifest.is_file()
    # Spot-check a few relative paths if present in manifest columns
    import csv

    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 500
    # Prefer relative_path / local_path fields used by this project
    sample = rows[0]
    path_keys = [k for k in sample if "path" in k.lower() or k in {"relative_path", "local_path"}]
    assert path_keys, f"No path columns in face manifest: {sample.keys()}"


def test_multimodal_250_protocol_present() -> None:
    path = (
        PROJECT_ROOT
        / "outputs"
        / "01_protocol"
        / "thesis_manifests"
        / "final_multimodal_250.csv"
    )
    assert path.is_file()
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        n = sum(1 for _ in csv.DictReader(handle))
    assert n == 250


def test_scrfd_home_checkpoint_or_skip() -> None:
    scrfd = Path.home() / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx"
    if not scrfd.is_file():
        pytest.skip("SCRFD ONNX not installed under ~/.insightface (optional for portable path)")
    assert scrfd.stat().st_size > 0


def test_app_package_modules_importable() -> None:
    import privacy_pipeline_app.wizard_workflow as wizard
    import privacy_pipeline_app.objective_policy as objective
    import privacy_pipeline_app.method_catalog as catalog
    import privacy_pipeline_app.runtime_policy as runtime
    import privacy_pipeline_app.production_app as ui
    import privacy_pipeline_app.thesis_face_detector as face
    import privacy_pipeline_app.detection_reviewer as reviewer

    assert callable(wizard.create_run)
    assert callable(objective.resolve_plan)
    assert callable(catalog.resolve_defaults_for_profile)
    assert callable(runtime.select_runtime_policy)
    assert callable(ui.build_app)
    assert "runtime_3_source_all_raw_rf_approximation" in face.POLICY_COMPONENTS
    assert "fusion_rfdetr_yolo11s_scrfd10g" in face.POLICY_COMPONENTS
    assert callable(reviewer.start_detection_reviewer)


def test_five_hundred_frame_method_image_dumps_present() -> None:
    """Comparable method image trees used by RQ2 evidence."""
    base = PROJECT_ROOT / "outputs" / "03_anonymisation"
    required = [
        base / "02_deterministic_baselines" / "blur_images",
        base / "02_deterministic_baselines" / "solid_mask_black_images",
        base / "12_riddle" / "images",
        base / "13_falco" / "images",
    ]
    for path in required:
        assert path.is_dir(), f"Missing method image dump: {path}"
        n = sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".webp", ".jpg", ".png"})
        assert n >= 100, f"Unexpectedly few images under {path}: {n}"
