"""Wizard workflow: setup/scan, method selection, honest fallbacks, artefacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from privacy_pipeline_app import wizard_workflow as ww


@pytest.fixture()
def sample_images(tmp_path: Path) -> Path:
    source = tmp_path / "inputs"
    source.mkdir()
    for name in ("a.webp", "b.jpg", "skip.txt"):
        path = source / name
        if name.endswith(".txt"):
            path.write_text("ignore", encoding="utf-8")
        else:
            Image.new("RGB", (64, 48), (120, 40, 40)).save(path)
    return source


def test_list_images_filters_extensions(sample_images: Path) -> None:
    images = ww.list_images(sample_images, recursive=False)
    assert len(images) == 2
    assert all(p.suffix.lower() in {".webp", ".jpg"} for p in images)


def test_list_images_and_scan_emit_progress(
    sample_images: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[float, str]] = []

    def capture(fraction: float, message: str) -> None:
        events.append((fraction, message))

    images = ww.list_images(sample_images, recursive=False, progress_callback=capture)
    assert len(images) == 2
    assert events
    assert events[-1][0] == 1.0

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(ww, "APP_RUNS", runs)
    monkeypatch.setattr(
        ww,
        "probe_environment",
        lambda: {
            "cuda_available": True,
            "device": "cuda",
            "gpu_name": "Test GPU",
            "vram_total_mb": 8192,
            "cpu_count": 8,
        },
    )
    scan_events: list[tuple[float, str]] = []
    state = ww.create_run(
        source_dir=str(sample_images),
        recursive=False,
        strategy="objective_profile",
        fixed_method="layered",
        focus="balanced",
        include_multimodal=True,
        progress_callback=lambda f, m: scan_events.append((f, m)),
    )
    assert scan_events
    assert scan_events[-1][0] == 1.0
    scan_events.clear()
    state = ww.step_scan(state.run_dir, progress_callback=lambda f, m: scan_events.append((f, m))
    )
    assert state.stages_done["scan"] is True
    assert scan_events
    assert scan_events[-1][0] == 1.0
    assert any("manifest" in m.lower() or "scan" in m.lower() for _, m in scan_events)


def test_create_run_and_scan_write_manifest_and_plan(
    sample_images: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(ww, "APP_RUNS", runs)
    monkeypatch.setattr(
        ww,
        "probe_environment",
        lambda: {
            "cuda_available": True,
            "device": "cuda",
            "gpu_name": "Test GPU",
            "vram_total_mb": 8192,
            "cpu_count": 8,
        },
    )

    state = ww.create_run(
        source_dir=str(sample_images),
        recursive=False,
        strategy="objective_profile",
        fixed_method="layered",
        focus="balanced",
        include_multimodal=True,
    )
    assert state.stages_done["setup"] is True
    assert state.n_images == 2
    assert state.focus == "balanced"
    assert Path(state.run_dir, "metadata", "setup.json").exists()
    assert Path(state.run_dir, "metadata", "objective_plan.json").exists()
    assert Path(state.run_dir, "metadata", "system_profile.json").exists()

    state = ww.step_scan(state.run_dir)
    assert state.stages_done["scan"] is True
    manifest = Path(state.run_dir) / "input_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    assert len(rows) == 2
    assert state.scan_summary["n_images"] == 2


def test_select_method_respects_plan_and_no_face() -> None:
    method, reason = ww._select_method(
        "objective_profile",
        "layered",
        "utility_under_privacy_floor",
        face_count=0,
        text_count=1,
        screen_count=0,
        plan=None,
    )
    assert method == "copy"
    assert "No face" in reason

    plan = {
        "title": "Balanced",
        "stages": [
            {
                "stage": "Face anonymisation",
                "method_id": "layered",
                "display_name": "Layered blur",
                "why": "test",
                "recommendation": "Recommended",
            }
        ],
    }
    method, reason = ww._select_method(
        "objective_profile",
        "solid_mask",
        "utility_under_privacy_floor",
        face_count=2,
        text_count=0,
        screen_count=0,
        plan=plan,
    )
    assert method == "layered"
    assert "Layered blur" in reason


def test_apply_method_deterministic_ok() -> None:
    image = Image.new("RGB", (80, 80), (10, 20, 30))
    boxes = [(10, 10, 40, 40)]
    result = ww._apply_method(image, boxes, "solid_mask")
    assert result.status == "ok"
    assert result.selected_method == "solid_mask"
    assert result.applied_method == "solid_mask"
    assert result.image.size == image.size


def test_apply_method_copy_when_no_boxes() -> None:
    image = Image.new("RGB", (40, 40), (1, 2, 3))
    result = ww._apply_method(image, [], "layered")
    assert result.applied_method == "copy"
    assert result.status == "ok"


def test_apply_method_honest_fallback_for_unknown_method() -> None:
    image = Image.new("RGB", (60, 60), (5, 5, 5))
    boxes = [(5, 5, 30, 30)]
    result = ww._apply_method(image, boxes, "not_a_real_method", fallback_method="solid_mask")
    assert result.status == "fallback"
    assert result.selected_method == "not_a_real_method"
    assert result.applied_method == "solid_mask"
    assert "Unknown method" in result.error
    assert "FALLBACK" in result.reason_note


def test_apply_method_honest_fallback_when_advanced_registry_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (60, 60), (5, 5, 5))
    boxes = [(5, 5, 30, 30)]

    import src.anonymisation.registry as registry_mod

    monkeypatch.setattr(registry_mod, "build_anonymiser_registry", lambda: {})
    result = ww._apply_method(image, boxes, "nullface", fallback_method="solid_mask")
    assert result.status == "fallback"
    assert result.selected_method == "nullface"
    assert result.applied_method == "solid_mask"
    assert result.error
    assert "FALLBACK" in result.reason_note


def test_write_detection_artifacts_splits_modalities(tmp_path: Path) -> None:
    records = [
        {
            "image_id": "img1.webp",
            "local_path": "/tmp/img1.webp",
            "detector": "fusion_rfdetr_scrfd10g",
            "screen_sources": ["yolo_screen"],
            "text_policy": "recognised_text_ocr",
            "faces": [{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "score": 0.9}],
            "screens": [{"x1": 10, "y1": 20, "x2": 30, "y2": 40, "score": 0.8}],
            "texts": [{"x1": 5, "y1": 6, "x2": 7, "y2": 8, "score": 0.7}],
        }
    ]
    det_dir = tmp_path / "detections"
    ww.write_detection_artifacts(det_dir, records)
    assert (det_dir / "detections.jsonl").exists()
    faces = list(csv.DictReader((det_dir / "face_boxes.csv").open(encoding="utf-8")))
    screens = list(csv.DictReader((det_dir / "screen_boxes.csv").open(encoding="utf-8")))
    texts = list(csv.DictReader((det_dir / "text_boxes.csv").open(encoding="utf-8")))
    assert len(faces) == 1 and faces[0]["image_id"] == "img1.webp"
    assert len(screens) == 1
    assert len(texts) == 1


def test_step_anonymise_and_report_logs_selected_vs_applied(
    sample_images: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(ww, "APP_RUNS", runs)
    monkeypatch.setattr(
        ww,
        "probe_environment",
        lambda: {
            "cuda_available": True,
            "device": "cuda",
            "gpu_name": "Test GPU",
            "vram_total_mb": 8192,
            "cpu_count": 4,
        },
    )
    state = ww.create_run(str(sample_images), False, "objective_profile", "layered", "balanced", True)
    # Force layered face plan stage (catalog-style)
    state.plan["stages"] = [
        {
            "stage": "Face anonymisation",
            "method_id": "layered",
            "display_name": "Layered blur",
            "why": "test",
            "recommendation": "Recommended",
        },
        {
            "stage": "Screen redaction",
            "method_id": "fill",
            "display_name": "Solid fill",
            "why": "test",
            "recommendation": "Recommended",
        },
        {
            "stage": "Text redaction",
            "method_id": "blur",
            "display_name": "Gaussian blur",
            "why": "test",
            "recommendation": "Recommended",
        },
    ]
    state.stages_done["scan"] = True
    state.stages_done["detect"] = True
    state.preflight_accepted = True
    ww.save_state(state)

    # Manifest already from create_run? create_run does not write manifest until scan
    state = ww.step_scan(state.run_dir)

    # Synthetic detections: one image with a face, one without
    manifest_rows = list(
        csv.DictReader((Path(state.run_dir) / "input_manifest.csv").open(encoding="utf-8"))
    )
    records = []
    for i, row in enumerate(manifest_rows):
        faces = (
            [{"x1": 5, "y1": 5, "x2": 30, "y2": 30, "score": 0.95}] if i == 0 else []
        )
        records.append(
            {
                "image_id": row["image_id"],
                "local_path": row["local_path"],
                "detector": "test",
                "faces": faces,
                "screens": [],
                "texts": [],
                "screen_sources": [],
                "text_policy": "none",
            }
        )
    ww.write_detection_artifacts(Path(state.run_dir) / "detections", records)

    state = ww.step_anonymise(state.run_dir)
    assert state.stages_done["anonymise"] is True
    decisions = list(
        csv.DictReader((Path(state.run_dir) / "metadata" / "decisions.csv").open(encoding="utf-8"))
    )
    assert len(decisions) == 2
    assert {"selected_method", "applied_method", "status", "error"} <= set(decisions[0])
    methods = {d["applied_method"] for d in decisions}
    assert "layered" in methods
    assert "copy" in methods
    assert all(d["status"] == "ok" for d in decisions)
    assert (Path(state.run_dir) / "metadata" / "selected_vs_applied.json").exists()

    state = ww.step_report(state.run_dir)
    report = (Path(state.run_dir) / "report" / "success_report.md").read_text(encoding="utf-8")
    assert "Methods selected" in report
    assert "Methods applied" in report
    assert "does not guarantee complete anonymisation" in report.lower() or "Boundary" in report


def test_map_objective_and_pipeline_markdown() -> None:
    assert ww.map_objective("privacy") == "privacy_first"
    assert ww.map_objective("balanced") == "utility_under_privacy_floor"
    md = ww.pipeline_markdown(None)
    assert "Setup" in md and "Detect" in md
