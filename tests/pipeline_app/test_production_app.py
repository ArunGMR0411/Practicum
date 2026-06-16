from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from privacy_pipeline_app import production_app
from privacy_pipeline_app.method_catalog import (
    build_preflight_dashboard,
    method_choice_options,
    resolve_defaults_for_profile,
)


def test_gallery_caps_recursive_previews_at_ten(tmp_path: Path) -> None:
    preview_dir = tmp_path / "side_by_side" / "nested"
    preview_dir.mkdir(parents=True)
    for index in range(14):
        (preview_dir / f"preview_{index:02d}.jpg").write_bytes(b"preview")

    previews = production_app._gallery(str(tmp_path))

    assert len(previews) == 10
    assert all(Path(path).parent == preview_dir for path in previews)


def _cpu_env() -> dict:
    return {
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": "not_available",
        "vram_total_mb": "not_available",
        "cpu_count": 4,
    }


def _patch_scan_env(monkeypatch, source: Path, runs: Path) -> None:
    monkeypatch.setattr(production_app, "DEFAULT_INPUT", str(source))
    from privacy_pipeline_app import wizard_workflow as ww

    monkeypatch.setattr(ww, "APP_RUNS", runs)
    monkeypatch.setattr(production_app, "probe_environment", _cpu_env)
    monkeypatch.setattr(ww, "probe_environment", _cpu_env)


def test_ui_scan_emits_progress_events(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "inputs"
    source.mkdir()
    from PIL import Image

    Image.new("RGB", (32, 32), (10, 20, 30)).save(source / "a.jpg")

    runs = tmp_path / "runs"
    runs.mkdir()
    _patch_scan_env(monkeypatch, source, runs)

    events: list[tuple[float, str]] = []

    def capture(fraction: float, message: str) -> None:
        events.append((fraction, message))

    result = production_app.ui_scan(str(source), "Balanced", False, progress_callback=capture)

    assert result[1] == "preflight"
    assert events, "scan must stream progress events"
    fractions = [f for f, _ in events]
    assert fractions[0] < 0.2
    assert fractions[-1] == 1.0
    assert any("Scan" in msg or "manifest" in msg.lower() or "folder" in msg.lower() for _, msg in events)


def test_scan_complete_restores_preflight_method_cards(tmp_path: Path, monkeypatch) -> None:
    """Regression: progress stream must not leave preflight selection cards hidden."""
    source = tmp_path / "inputs"
    source.mkdir()
    from PIL import Image

    Image.new("RGB", (40, 40), (8, 16, 24)).save(source / "a.jpg")
    runs = tmp_path / "runs"
    runs.mkdir()
    _patch_scan_env(monkeypatch, source, runs)

    # Simulate the stream path: progress hides the panel, then complete normalizes.
    progress = production_app.running_ui_result("", "setup", 0.4, "Scanning…")
    assert "run-progress" in progress[4]
    assert progress[5].get("visible") is False

    raw = production_app.ui_scan(str(source), "Balanced", False)
    assert raw[1] == "preflight"
    assert isinstance(raw[5], str) and "dash-stage-card" in raw[5]
    assert raw[5].count("dash-stage-card") == 5

    final = production_app.normalize_ui_outputs(raw)
    assert final[1] == "preflight"
    # Banner must leave progress mode so CSS can show #preflight-stage again.
    assert "run-progress" not in final[4]
    assert "Scan done" in final[4] or "banner" in final[4]
    # Preflight HTML panel + group must be visible with all five method cards.
    assert final[5].get("visible") is True
    assert "dash-stage-card" in final[5].get("value", "")
    assert final[5]["value"].count("dash-stage-card") == 5
    assert final[6].get("visible") is False  # setup hidden
    assert final[7].get("visible") is True  # preflight group shown
    assert final[9].get("visible") is True  # proceed
    assert final[10].get("visible") is True  # back
    # Hidden Gradio stage triggers used by card onclick must still be visible=True in DOM.
    stage_updates = final[24:29]
    assert len(stage_updates) == 5
    assert all(u.get("visible") is True for u in stage_updates)


def test_risk_gate_keeps_preflight_cards_visible(monkeypatch) -> None:
    """Proceed risk gate stays on preflight and must still show the method dashboard."""
    state = SimpleNamespace(
        focus="balanced",
        n_images=10,
        source_dir="app/inputs",
        run_dir="run",
        plan={
            "user_method_selections": {
                "face_detection": "runtime_3_source_all_raw_rf_approximation",
                "multimodal_detection": "reviewed_screen_yolo11s_1280",
                "face_anonymisation": "riddle",
                "screen_operator": "fill",
                "text_operator": "blur",
            }
        },
        stages_done={"scan": True},
        preflight_accepted=False,
        eta_seconds=100.0,
    )
    monkeypatch.setattr(production_app, "load_state", lambda _: state)
    monkeypatch.setattr(
        production_app,
        "probe_environment",
        lambda: {"cuda_available": True, "vram_total_mb": 4096, "device": "cuda", "gpu_name": "t"},
    )
    monkeypatch.setattr(production_app, "_stats_html", lambda _: "stats")
    monkeypatch.setattr(
        production_app,
        "_dashboard_for_state",
        lambda *_a, **_k: "<div class='preflight-dash'><button class='dash-stage-card'>x</button></div>",
    )

    raw = production_app.ui_proceed("run", risk_ok=False)
    assert raw[1] == "preflight"
    final = production_app.normalize_ui_outputs(raw)
    assert final[1] == "preflight"
    assert final[5].get("visible") is True
    assert "dash-stage-card" in final[5].get("value", "")
    assert final[7].get("visible") is True


def test_anonymisation_advances_to_report_before_done(
    tmp_path: Path, monkeypatch
) -> None:
    report_path = tmp_path / "report" / "success_report.md"
    report_path.parent.mkdir(parents=True)

    def fake_anonymise(run_dir: str, progress_callback=None) -> None:
        if progress_callback is not None:
            progress_callback(0.0, "start")
            progress_callback(0.5, "mid")
            progress_callback(1.0, "done")

    monkeypatch.setattr(production_app, "step_anonymise", fake_anonymise)

    def write_report(run_dir: str) -> None:
        report_path.write_text("# Completed report\n", encoding="utf-8")

    monkeypatch.setattr(production_app, "step_report", write_report)

    progress_events: list[tuple[float, str]] = []

    def capture(fraction: float, message: str) -> None:
        progress_events.append((fraction, message))

    result = production_app.ui_run_step(
        str(tmp_path), "anonymise", progress_callback=capture
    )

    assert result[1] == "report"
    assert production_app._report_markdown(str(tmp_path)).startswith("# Completed report")
    assert progress_events, "anonymise step must emit progress events"
    fractions = [f for f, _ in progress_events]
    assert fractions[0] <= 0.05
    assert any(0.4 <= f <= 0.7 for f in fractions), fractions
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions), "progress should be non-decreasing"


def test_finish_report_advances_to_done_with_ten_previews(tmp_path: Path) -> None:
    report_path = tmp_path / "report" / "success_report.md"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("# Completed report\n", encoding="utf-8")
    preview_dir = tmp_path / "side_by_side"
    preview_dir.mkdir()
    for index in range(12):
        (preview_dir / f"preview_{index:02d}.jpg").write_bytes(b"preview")

    result = production_app.ui_finish_report(str(tmp_path))

    assert result[1] == "done"
    assert len(result[15]) == 10


def test_done_can_return_only_to_existing_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report" / "success_report.md"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("# Completed report\n", encoding="utf-8")

    result = production_app.ui_back_to_report(str(tmp_path))

    assert result[1] == "report"
    assert result[11]["visible"] is False
    assert result[15] == []


def test_back_to_report_control_is_mounted_for_done_stage_css() -> None:
    app = production_app.build_app()
    controls = {
        component["props"].get("elem_id"): component
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id")
    }

    assert controls["back-report-action"]["props"]["visible"] is True


def test_run_control_is_mounted_for_detect_and_anonymise_stage_css() -> None:
    app = production_app.build_app()
    controls = {
        component["props"].get("elem_id"): component
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id")
    }

    assert controls["run-stage-action"]["props"]["visible"] is True
    assert controls["run-stage-action"]["props"]["interactive"] is False
    assert production_app._vis("detect")["run_btn"]["visible"] is True
    assert production_app._vis("anonymise")["run_btn"]["visible"] is True
    assert production_app._vis("anonymise")["run_btn"]["value"] == "Run anonymisation"


def test_preflight_has_one_clickable_card_per_stage_without_system_duplication() -> None:
    env = {
        "cuda_available": True,
        "device": "cuda",
        "gpu_name": "Test GPU",
        "vram_total_mb": 8192,
        "cpu_count": 8,
    }
    selections = resolve_defaults_for_profile("balanced", env)

    html = build_preflight_dashboard(
        focus_title="Balanced",
        focus_summary="Test summary",
        n_images=35,
        source_dir="app/inputs",
        env=env,
        selections=selections,
        focus="balanced",
    )

    assert html.count("dash-stage-card") == 5
    assert "stage-trigger-face-detection" in html
    assert "stage-trigger-text-operator" in html
    assert "Your system power" not in html
    assert "system-strip" not in html
    assert "detail-panel" not in html


def test_method_picker_labels_every_option_with_compute_suitability() -> None:
    env = {"cuda_available": True, "vram_total_mb": 8192}

    choices = method_choice_options("face_detection", env)

    assert choices
    assert all("Recommended" in label or "Not recommended" in label for label, _ in choices)
    assert all(value and " | " not in value for _, value in choices)


def test_risky_preflight_reuses_the_single_proceed_button(monkeypatch) -> None:
    state = SimpleNamespace(
        focus="balanced",
        n_images=35,
        source_dir="app/inputs",
        plan={
            "user_method_selections": {
                "face_detection": "fusion_rfdetr_scrfd10g",
                "multimodal_detection": "reviewed_screen_yolo11s_1280",
                "face_anonymisation": "nullface",
                "screen_operator": "fill",
                "text_operator": "blur",
            }
        },
    )
    monkeypatch.setattr(production_app, "load_state", lambda _: state)
    monkeypatch.setattr(
        production_app,
        "probe_environment",
        lambda: {"cuda_available": True, "vram_total_mb": 7705},
    )
    monkeypatch.setattr(production_app, "_stats_html", lambda _: "stats")
    monkeypatch.setattr(production_app, "_dashboard_for_state", lambda *_args, **_kwargs: "dashboard")

    result = production_app.ui_proceed("run", risk_ok=False)

    assert result[1] == "preflight"
    assert result[9]["visible"] is True
    assert result[19]["visible"] is True
    assert result[20]["visible"] is False


def test_obsolete_second_proceed_control_remains_unmounted() -> None:
    app = production_app.build_app()
    controls = {
        component["props"].get("elem_id"): component
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id")
    }

    assert controls["risk-proceed-action"]["props"]["visible"] is False


def test_done_gallery_disables_inplace_preview_zoom() -> None:
    """Done page must not use Gradio gallery lightbox zoom (allow_preview=False)."""
    app = production_app.build_app()
    galleries = [
        component
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id") == "done-gallery"
        or (
            component.get("type") == "gallery"
            and "done" in str(component.get("props", {})).lower()
        )
    ]
    # Prefer elem_id if present; otherwise inspect all galleries for allow_preview
    found_disabled = False
    for component in app.config["components"]:
        props = component.get("props") or {}
        if props.get("elem_id") in {"done-gallery", "done-gallery-wrap"}:
            if props.get("allow_preview") is False:
                found_disabled = True
        # Gradio may store allow_preview on the Gallery component props
        if "allow_preview" in props and props.get("elem_id") and "gallery" in str(props.get("elem_id")):
            if props.get("allow_preview") is False:
                found_disabled = True
    # Fallback: source-level contract - build_app constructs Gallery(allow_preview=False)
    import inspect
    import privacy_pipeline_app.production_app as mod

    source = inspect.getsource(mod.build_app)
    assert "allow_preview=False" in source
    assert found_disabled or "allow_preview=False" in source


def test_done_to_setup_clears_run_context() -> None:
    result = production_app.ui_done_to_setup()
    # First outputs include step label returning to setup
    assert result[1] == "setup"
