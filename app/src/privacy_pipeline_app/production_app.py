#!/usr/bin/env python3
"""Egocentric privacy pipeline wizard: plan, preflight, and auto-flow."""

from __future__ import annotations

from html import escape
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Callable

import gradio as gr

from privacy_pipeline_app.detection_reviewer import start_detection_reviewer
from privacy_pipeline_app.method_catalog import (
    STAGE_KEYS,
    STAGE_LABELS,
    STAGE_OPTIONS,
    any_not_recommended,
    apply_selections_to_plan,
    build_preflight_dashboard,
    estimate_eta_from_selections,
    get_option,
    method_choice_options,
    method_id_from_display_name,
    resolve_defaults_for_profile,
    stage_card_labels,
)
from privacy_pipeline_app.objective_policy import resolve_plan
from privacy_pipeline_app.production_runner import probe_environment
from privacy_pipeline_app.wizard_workflow import (
    accept_preflight,
    create_run,
    load_state,
    mark_review_done,
    save_state,
    step_anonymise,
    step_detect,
    step_report,
    step_scan,
    write_json,
)

DEFAULT_INPUT = "app/inputs"
FOCUS_CHOICES = ["Privacy", "Balanced", "Utility"]

# setup → preflight → detect → review → anonymise → report → done
STEP_HINTS = {
    "setup": "Choose focus + folder",
    "preflight": "Confirm methods & ETA",
    "detect": "Find faces · screens · text",
    "review": "Optional box review",
    "anonymise": "Protect images",
    "report": "Review run report",
    "done": "View output previews",
}


def _step_bar(active: str) -> str:
    order = ["setup", "preflight", "detect", "review", "anonymise", "report", "done"]
    parts = []
    ai = order.index(active) if active in order else 0
    for i, s in enumerate(order):
        cls = "step"
        if i < ai:
            cls += " done"
        if s == active:
            cls += " active"
        parts.append(
            f"<span class='{cls} step-{s}' title='{STEP_HINTS.get(s, '')}'>"
            f"{s.capitalize()}</span>"
        )
    return f"<div class='steps'>{''.join(parts)}</div>"


def _stats_html(run_dir: str) -> str:
    if not run_dir:
        return "<div class='stats muted'>No run yet</div>"
    try:
        st = load_state(Path(run_dir))
    except Exception:
        return "<div class='stats muted'>No run data</div>"
    focus = (st.focus or "balanced").capitalize()
    return (
        f"<div class='stats'>"
        f"<span class='chip' title='Objective plan'>{focus}</span>"
        f"<span class='chip' title='Run'>#{st.run_id}</span>"
        f"<span class='chip' title='Images'>🖼 {st.n_images}</span>"
        f"<span class='chip face' title='Faces'>😀 {st.n_faces}</span>"
        f"<span class='chip screen' title='Screens'>📱 {st.n_screens}</span>"
        f"<span class='chip text' title='Text'>🔤 {st.n_texts}</span>"
        f"</div>"
    )


def _msg(text: str, ok: bool = True) -> str:
    return f"<div class='banner {'ok' if ok else 'err'}'>{text}</div>"


def _progress_msg(fraction: float, text: str) -> str:
    percentage = max(0, min(100, int(round(fraction * 100))))
    return (
        "<div class='run-progress' role='status' aria-live='polite'>"
        "<div class='run-progress-head'>"
        f"<strong>{escape(text)}</strong><span>{percentage}%</span>"
        "</div>"
        "<div class='run-progress-track'>"
        f"<span style='width:{percentage}%'></span>"
        "</div>"
        "<div class='run-progress-note'>Processing is active. Keep this page open.</div>"
        "</div>"
    )


def _update_visible(update_obj, *, visible: bool):
    """Return a Gradio update that forces visibility while keeping other fields.

    Gradio's ``gr.update(...)`` returns a plain dict with ``__type__ == 'update'``.
    """
    if isinstance(update_obj, dict):
        merged = {k: v for k, v in update_obj.items() if k != "__type__"}
        merged["visible"] = visible
        return gr.update(**merged)
    return gr.update(visible=visible)


def normalize_ui_outputs(r) -> tuple:
    """Expand short handler return tuples into the full Gradio ``outs`` vector.

    Critical: during streamed progress the preflight HTML panel is hidden. When
    a step lands on preflight again, this must restore ``visible=True`` on the
    dashboard panel and stage-trigger buttons so method cards remain selectable.
    """
    items = list(r)
    while len(items) < 18:
        items.append(gr.update())
    (
        rd,
        st,
        bar,
        stt,
        ban,
        html,
        v_setup,
        v_pf,
        v_scan,
        v_proc,
        v_back,
        v_run,
        v_skip,
        v_rev,
        v_fin,
        gal,
        _v_gal,
        v_link,
    ) = items[:18]
    if len(items) > 20:
        risk_html, risk_chk, risk_btn = items[18], items[19], items[20]
    else:
        risk_html = gr.update(visible=False, value="")
        risk_chk = gr.update(visible=False, value=False)
        risk_btn = gr.update(visible=False)

    on_preflight = st == "preflight"
    # Stage button updates: items[21:26], pick_modal items[26]
    if len(items) > 26:
        stage_ups = list(items[21:26])
        pick_up = items[26]
    elif len(items) > 21:
        stage_ups = list(items[21:26])
        while len(stage_ups) < 5:
            stage_ups.append(gr.update(visible=on_preflight))
        pick_up = gr.update(visible=False)
    else:
        stage_ups = [gr.update(visible=on_preflight) for _ in STAGE_KEYS]
        pick_up = gr.update(visible=False)

    if on_preflight:
        # Always re-show preflight chrome after a progress stream hid it.
        v_setup = gr.update(visible=False)
        v_pf = gr.update(visible=True)
        v_scan = gr.update(visible=False)
        v_proc = gr.update(visible=True)
        v_back = gr.update(visible=True)
        if isinstance(html, str):
            html_upd = gr.update(value=html, visible=True)
        else:
            html_upd = _update_visible(html, visible=True)
        # Hidden Gradio triggers still need to exist for dash-card onclick bridges.
        stage_ups = [_update_visible(up, visible=True) for up in stage_ups]
        while len(stage_ups) < 5:
            stage_ups.append(gr.update(visible=True))
        stage_ups = stage_ups[:5]
    else:
        if isinstance(html, str):
            html_upd = gr.update(value=html, visible=False)
        else:
            html_upd = html

    gal_list = gal if isinstance(gal, list) else []
    gal_upd = gr.update(value=gal_list, visible=(st == "done"))
    report_visible = st == "report"
    if not on_preflight:
        risk_html = gr.update(visible=False, value="")
        risk_chk = gr.update(visible=False, value=False)
        risk_btn = gr.update(visible=False)
        stage_ups = [gr.update(visible=False) for _ in STAGE_KEYS]
        pick_up = gr.update(visible=False)
    done_btn_up = gr.update(visible=(st == "done"))
    # Full preview image: show on Done; clear when leaving Done unless caller set a path
    if st == "done":
        preview_up = gr.update(visible=True)
        if isinstance(gal_list, list) and gal_list:
            first = gal_list[0]
            if isinstance(first, (list, tuple)):
                first = first[0]
            if isinstance(first, dict):
                first = first.get("image") or first.get("name") or first.get("path")
            if isinstance(first, str) and first:
                preview_up = gr.update(value=first, visible=True)
    else:
        preview_up = gr.update(value=None, visible=False)
    return (
        rd,
        st,
        bar,
        stt,
        ban,
        html_upd,
        v_setup,
        v_pf,
        v_scan,
        v_proc,
        v_back,
        v_run,
        v_skip,
        v_rev,
        v_fin,
        gal_upd,
        v_link,
        gr.update(visible=report_visible),
        gr.update(
            value=_report_markdown(rd) if report_visible else "",
            visible=report_visible,
        ),
        gr.update(visible=report_visible),
        gr.update(visible=st == "done"),
        risk_html,
        risk_chk,
        risk_btn,
        *stage_ups,
        pick_up,
        done_btn_up,
        preview_up,
    )


def running_ui_result(rd: str, current_step: str, fraction: float, message: str) -> tuple:
    """Full outputs vector while a long step is running (progress banner only)."""
    return (
        rd,
        current_step,
        _step_bar(current_step),
        _stats_html(rd),
        _progress_msg(fraction, message),
        gr.update(visible=False, value=""),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False, interactive=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=[], visible=False),
        gr.update(visible=False, value=""),
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False, value=""),
        gr.update(visible=False, value=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
    )


def _gallery(run_dir: str, limit: int = 10) -> list[str]:
    if not run_dir:
        return []
    folder = Path(run_dir) / "side_by_side"
    if not folder.exists():
        return []
    return [str(p) for p in sorted(folder.rglob("*.jpg"))[:limit]]


def _report_markdown(run_dir: str) -> str:
    if not run_dir:
        return "Report data is unavailable."
    report_path = Path(run_dir) / "report" / "success_report.md"
    if not report_path.exists():
        return f"Report file was not found at `{report_path}`."
    return report_path.read_text(encoding="utf-8")


def _vis(step: str) -> dict[str, gr.Update]:
    """Control which panels/buttons show."""
    return {
        "setup": gr.update(visible=step == "setup"),
        "preflight": gr.update(visible=step == "preflight"),
        "scan_btn": gr.update(visible=step == "setup"),
        "proceed_btn": gr.update(visible=step == "preflight"),
        "back_btn": gr.update(visible=step == "preflight"),
        "run_btn": gr.update(
            visible=step in {"detect", "anonymise"},
            interactive=step in {"detect", "anonymise"},
            value="Run anonymisation" if step == "anonymise" else "Run detection",
        ),
        "skip_btn": gr.update(visible=step == "review"),
        "review_btn": gr.update(visible=step == "review"),
        "finish_review_btn": gr.update(visible=step == "review"),
        "gallery": gr.update(visible=step == "done"),
    }


def _focus_key(label: str) -> str:
    raw = (label or "balanced").strip().lower()
    if raw.startswith("priv"):
        return "privacy"
    if raw.startswith("util"):
        return "utility"
    return "balanced"


def _current_selections(st) -> dict[str, str]:
    plan = st.plan or {}
    if plan.get("user_method_selections"):
        return dict(plan["user_method_selections"])
    out: dict[str, str] = {}
    for stage in plan.get("stages") or []:
        name = str(stage.get("stage") or "")
        mid = str(stage.get("method_id") or "")
        if "Face detection" == name:
            out["face_detection"] = mid
        elif "Screen / text" in name or "Screen and text" in name:
            out["multimodal_detection"] = mid
        elif "Face anonymisation" == name:
            out["face_anonymisation"] = mid
        elif "Screen redaction" == name:
            out["screen_operator"] = mid
        elif "Text redaction" == name:
            out["text_operator"] = mid
    if len(out) < 5:
        env = probe_environment()
        out = resolve_defaults_for_profile(st.focus or "balanced", env)
    return out


def _dashboard_for_state(st, env: dict, active_stage: str | None = None, detail_id: str | None = None) -> str:
    plan_obj = resolve_plan(st.focus or "balanced")
    selections = _current_selections(st)
    return build_preflight_dashboard(
        focus_title=plan_obj.title,
        focus_summary=plan_obj.summary,
        n_images=st.n_images,
        source_dir=st.source_dir,
        env=env,
        selections=selections,
        focus=st.focus or "balanced",
        active_stage=active_stage,
        detail_method_id=detail_id,
    )


def ui_scan(
    source_dir: str,
    focus_label: str,
    recursive: bool,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple:
    """Page 1 to scan to unified preflight dashboard."""

    def emit(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(1.0, fraction)), message)

    try:
        focus_key = _focus_key(focus_label)
        emit(0.02, "Creating run workspace")
        st = create_run(
            source_dir=(source_dir or DEFAULT_INPUT).strip(),
            recursive=bool(recursive),
            strategy="objective_profile",
            fixed_method="layered",
            focus=focus_key,
            include_multimodal=True,
            progress_callback=lambda fraction, message: emit(0.02 + fraction * 0.45, message),
        )
        emit(0.50, "Building image manifest and runtime estimate")
        st = step_scan(
            st.run_dir,
            progress_callback=lambda fraction, message: emit(0.50 + fraction * 0.30, message),
        )
        emit(0.82, "Building preflight dashboard")
        env = probe_environment()
        plan = resolve_plan(focus_key)
        from privacy_pipeline_app.runtime_policy import select_runtime_policy

        runtime_policy = select_runtime_policy(env)
        defaults = resolve_defaults_for_profile(focus_key, env)
        plan_dict = apply_selections_to_plan(plan.to_dict(), defaults, env)
        plan_dict["runtime_policy_id"] = runtime_policy.policy_id
        if not plan_dict.get("runtime_policy"):
            plan_dict["runtime_policy"] = runtime_policy.to_dict()
        eta = estimate_eta_from_selections(st.n_images, defaults, env)
        st.eta_seconds = eta
        st.plan = plan_dict
        st.fixed_method = defaults.get("face_anonymisation", "layered")
        save_state(st)
        write_json(Path(st.run_dir) / "metadata" / "objective_plan.json", st.plan)

        emit(0.95, "Finalising method cards")
        html = _dashboard_for_state(st, env)
        labels = stage_card_labels(defaults, env)
        v = _vis("preflight")
        emit(1.0, f"Scan done · {st.n_images} images")
        return (
            st.run_dir,
            "preflight",
            _step_bar("preflight"),
            _stats_html(st.run_dir),
            _msg(f"Scan done. {st.n_images} images. Select any stage to compare methods."),
            html,
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            [],
            v["gallery"],
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=False),
            gr.update(visible=False),
            gr.update(value=labels["face_detection"], visible=True),
            gr.update(value=labels["multimodal_detection"], visible=True),
            gr.update(value=labels["face_anonymisation"], visible=True),
            gr.update(value=labels["screen_operator"], visible=True),
            gr.update(value=labels["text_operator"], visible=True),
            gr.update(visible=False),
        )
    except Exception as exc:
        v = _vis("setup")
        return (
            "",
            "setup",
            _step_bar("setup"),
            _stats_html(""),
            _msg(f"Scan failed: {type(exc).__name__}: {exc}", False),
            "",
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            [],
            v["gallery"],
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def ui_open_stage_picker(run_dir: str, stage_key: str) -> tuple:
    """Open the method picker modal for one stage."""
    if not run_dir or stage_key not in STAGE_KEYS:
        return gr.update(visible=False), gr.update(), gr.update(), gr.update()
    try:
        st = load_state(Path(run_dir))
        env = probe_environment()
        selections = _current_selections(st)
        current_id = selections.get(stage_key, "")
        opt = get_option(stage_key, current_id)
        choices = method_choice_options(stage_key, env)
        current_name = opt.display_name if opt else (choices[0][1] if choices else None)
        title = f"### {STAGE_LABELS.get(stage_key, stage_key)}"
        return (
            gr.update(visible=True),
            gr.update(value=title),
            gr.update(choices=choices, value=current_name, visible=True),
            gr.update(value=stage_key),
        )
    except Exception as exc:
        return gr.update(visible=False), gr.update(value=str(exc)), gr.update(), gr.update(value="")


def ui_apply_stage_pick(run_dir: str, stage_key: str, display_name: str) -> tuple:
    """Apply a method chosen in the picker modal."""
    if not run_dir or stage_key not in STAGE_KEYS:
        return gr.update(), gr.update(), gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    try:
        method_id = method_id_from_display_name(stage_key, display_name or "")
        if not method_id:
            return (
                gr.update(),
                _msg("Unknown method.", False),
                gr.update(visible=False),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
        st = load_state(Path(run_dir))
        env = probe_environment()
        selections = _current_selections(st)
        selections[stage_key] = method_id
        plan = apply_selections_to_plan(st.plan or {}, selections, env)
        eta = estimate_eta_from_selections(st.n_images, selections, env)
        st.plan = plan
        st.eta_seconds = eta
        st.fixed_method = selections.get("face_anonymisation", st.fixed_method)
        save_state(st)
        write_json(Path(st.run_dir) / "metadata" / "objective_plan.json", st.plan)
        html = _dashboard_for_state(st, env, active_stage=stage_key, detail_id=method_id)
        from privacy_pipeline_app.method_catalog import compute_badge

        opt = get_option(stage_key, method_id)
        badge = compute_badge(opt, env) if opt else ""
        labels = stage_card_labels(selections, env)
        msg = f"Updated {opt.display_name if opt else method_id}. {badge}. ETA about {eta/60:.1f} min."
        return (
            html,
            _msg(msg),
            gr.update(visible=False),
            gr.update(value=labels["face_detection"]),
            gr.update(value=labels["multimodal_detection"]),
            gr.update(value=labels["face_anonymisation"]),
            gr.update(value=labels["screen_operator"]),
            gr.update(value=labels["text_operator"]),
            gr.update(value=""),
        )
    except Exception as exc:
        return (
            gr.update(),
            _msg(f"Could not update method: {exc}", False),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def ui_close_stage_picker() -> tuple:
    return gr.update(visible=False), gr.update(value="")


def ui_back() -> tuple:
    v = _vis("setup")
    return (
        "",
        "setup",
        _step_bar("setup"),
        _stats_html(""),
        _msg("Choose focus again, then Scan"),
        "",
        v["setup"],
        v["preflight"],
        v["scan_btn"],
        v["proceed_btn"],
        v["back_btn"],
        v["run_btn"],
        v["skip_btn"],
        v["review_btn"],
        v["finish_review_btn"],
        [],
        v["gallery"],
        gr.update(visible=False, value=""),
        gr.update(visible=False, value=""),
        gr.update(visible=False, value=False),
        gr.update(visible=False),
    )


def ui_proceed(run_dir: str, risk_ok: bool = False) -> tuple:
    """Proceed to detect, or require risk confirm if any method is not compute-fit."""
    try:
        st = load_state(Path(run_dir))
        env = probe_environment()
        selections = _current_selections(st)
        risky = any_not_recommended(selections, env)
        if risky and not risk_ok:
            lines = "<br>".join(escape(x) for x in risky)
            risk_html = (
                f"<div class='risk-box'><strong>Risk confirmation needed</strong>"
                f"<p>You selected methods that are Not recommended for this system:</p>"
                f"<p>{lines}</p>"
                f"<p>These choices may run out of memory, take substantially longer, or fall back. "
                f"Tick the box below, then select Proceed to Detect again, or choose a Recommended method.</p></div>"
            )
            return (
                run_dir,
                "preflight",
                _step_bar("preflight"),
                _stats_html(run_dir),
                _msg("Confirm risk before continuing, or change methods on the dashboard."),
                _dashboard_for_state(st, env),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                [],
                gr.update(visible=False),
                gr.update(visible=False, value=""),
                gr.update(visible=True, value=risk_html),
                gr.update(value=False, visible=True),
                gr.update(visible=False),
            )

        accept_preflight(run_dir)
        v = _vis("detect")
        return (
            run_dir,
            "detect",
            _step_bar("detect"),
            _stats_html(run_dir),
            _msg("Plan locked. Select Run detection below to begin."),
            gr.update(visible=False),
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            [],
            v["gallery"],
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=False),
            gr.update(visible=False),
        )
    except Exception as exc:
        v = _vis("preflight")
        return (
            run_dir,
            "preflight",
            _step_bar("preflight"),
            _stats_html(run_dir),
            _msg(f"Proceed failed: {exc}", False),
            gr.update(),
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            [],
            v["gallery"],
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=False),
            gr.update(visible=False),
        )


def ui_run_step(
    run_dir: str,
    step: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple:
    try:
        if step == "detect":
            st = step_detect(run_dir, progress_callback=progress_callback)
            next_step = "review"
            msg = f"Detected · faces {st.n_faces} · screens {st.n_screens} · text {st.n_texts}"
            gal: list[str] = []
        elif step == "anonymise":
            # Map anonymise-internal 0–1 progress into ~0.02–0.88 so the
            # Reserve progress for report generation.
            def anonymise_progress(fraction: float, message: str) -> None:
                if progress_callback is not None:
                    progress_callback(0.02 + max(0.0, min(1.0, fraction)) * 0.86, message)

            if progress_callback is not None:
                progress_callback(0.02, "Applying the selected anonymisation policy")
            step_anonymise(
                run_dir,
                progress_callback=anonymise_progress if progress_callback is not None else None,
            )
            if progress_callback is not None:
                progress_callback(0.92, "Creating the final report and comparison views")
            step_report(run_dir)
            if progress_callback is not None:
                progress_callback(1.0, "Report ready")
            next_step = "report"
            msg = "Anonymisation complete · review the report before viewing output previews"
            gal = []
        else:
            next_step = step
            msg = "Nothing to run"
            gal = []
        v = _vis(next_step)
        return (
            run_dir,
            next_step,
            _step_bar(next_step),
            _stats_html(run_dir),
            _msg(msg),
            gr.update(visible=False),
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            gal,
            v["gallery"],
            gr.update(visible=False, value=""),
        )
    except Exception as exc:
        v = _vis(step)
        return (
            run_dir,
            step,
            _step_bar(step),
            _stats_html(run_dir),
            _msg(f"Failed: {type(exc).__name__}: {exc}", False),
            gr.update(),
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            [],
            v["gallery"],
            gr.update(visible=False, value=""),
        )


def ui_finish_report(run_dir: str) -> tuple:
    """Advance from the generated report to the final preview page."""
    report_path = Path(run_dir) / "report" / "success_report.md"
    if not report_path.exists():
        step_report(run_dir)
    v = _vis("done")
    return (
        run_dir,
        "done",
        _step_bar("done"),
        _stats_html(run_dir),
        _msg("Run complete. Previews below. Press Done to start a new run."),
        gr.update(visible=False),
        v["setup"],
        v["preflight"],
        v["scan_btn"],
        v["proceed_btn"],
        v["back_btn"],
        v["run_btn"],
        v["skip_btn"],
        v["review_btn"],
        v["finish_review_btn"],
        _gallery(run_dir),
        v["gallery"],
        gr.update(visible=False, value=""),
    )


def ui_done_to_setup() -> tuple:
    """Leave Done page and return to a clean Setup for a new job."""
    v = _vis("setup")
    return (
        "",
        "setup",
        _step_bar("setup"),
        _stats_html(""),
        _msg("Ready for a new run. Choose profile and folder, then Scan."),
        "",
        v["setup"],
        v["preflight"],
        v["scan_btn"],
        v["proceed_btn"],
        v["back_btn"],
        v["run_btn"],
        v["skip_btn"],
        v["review_btn"],
        v["finish_review_btn"],
        [],
        v["gallery"],
        gr.update(visible=False, value=""),
    )


def ui_back_to_report(run_dir: str) -> tuple:
    """Return from Done to the existing report without reopening earlier stages."""
    report_path = Path(run_dir) / "report" / "success_report.md"
    if not report_path.exists():
        step_report(run_dir)
    v = _vis("report")
    return (
        run_dir,
        "report",
        _step_bar("report"),
        _stats_html(run_dir),
        _msg("Report ready · select Continue to Done when finished"),
        gr.update(visible=False),
        v["setup"],
        v["preflight"],
        v["scan_btn"],
        v["proceed_btn"],
        v["back_btn"],
        v["run_btn"],
        v["skip_btn"],
        v["review_btn"],
        v["finish_review_btn"],
        [],
        v["gallery"],
        gr.update(visible=False, value=""),
    )


def ui_skip(run_dir: str) -> tuple:
    try:
        mark_review_done(run_dir)
        v = _vis("anonymise")
        return (
            run_dir,
            "anonymise",
            _step_bar("anonymise"),
            _stats_html(run_dir),
            _msg("Review skipped · Run Anonymise"),
            gr.update(visible=False),
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            [],
            v["gallery"],
            gr.update(visible=False, value=""),
        )
    except Exception as exc:
        v = _vis("review")
        return (
            run_dir,
            "review",
            _step_bar("review"),
            _stats_html(run_dir),
            _msg(f"Skip failed: {exc}", False),
            gr.update(visible=False),
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            [],
            v["gallery"],
            gr.update(visible=False, value=""),
        )


def ui_review(run_dir: str) -> tuple:
    try:
        url = start_detection_reviewer(run_dir)
        link = (
            f"<div class='review-link'>"
            f"<a href='{url}' target='_blank' rel='noopener'>Open reviewer ↗</a>"
            f"<span class='muted'> · green face · blue screen · amber text</span></div>"
        )
        v = _vis("review")
        return (
            run_dir,
            "review",
            _step_bar("review"),
            _stats_html(run_dir),
            _msg("Reviewer opened · Save/Done there, then Finish review"),
            gr.update(visible=False),
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            [],
            v["gallery"],
            gr.update(visible=True, value=link),
        )
    except Exception as exc:
        v = _vis("review")
        return (
            run_dir,
            "review",
            _step_bar("review"),
            _stats_html(run_dir),
            _msg(f"Reviewer failed: {exc}", False),
            gr.update(visible=False),
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            [],
            v["gallery"],
            gr.update(visible=False, value=""),
        )


def ui_finish_review(run_dir: str) -> tuple:
    try:
        from privacy_pipeline_app.wizard_workflow import refresh_state_from_detections

        try:
            refresh_state_from_detections(run_dir)
        except Exception:
            pass
        mark_review_done(run_dir)
        v = _vis("anonymise")
        return (
            run_dir,
            "anonymise",
            _step_bar("anonymise"),
            _stats_html(run_dir),
            _msg("Review finished · Run Anonymise"),
            gr.update(visible=False),
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            [],
            v["gallery"],
            gr.update(visible=False, value=""),
        )
    except Exception as exc:
        v = _vis("review")
        return (
            run_dir,
            "review",
            _step_bar("review"),
            _stats_html(run_dir),
            _msg(f"Finish failed: {exc}", False),
            gr.update(visible=False),
            v["setup"],
            v["preflight"],
            v["scan_btn"],
            v["proceed_btn"],
            v["back_btn"],
            v["run_btn"],
            v["skip_btn"],
            v["review_btn"],
            v["finish_review_btn"],
            [],
            v["gallery"],
            gr.update(visible=False, value=""),
        )


OUTS = [
    "run_dir",
    "step",
    "step_bar",
    "stats",
    "banner",
    "preflight_panel",
    "setup_group",
    "preflight_group",
    "scan_btn",
    "proceed_btn",
    "back_btn",
    "run_btn",
    "skip_btn",
    "review_btn",
    "finish_review_btn",
    "gallery",
    "gallery",  # visibility update as second gallery output? No - fix
]

CSS = """
.gradio-container {
  --app-bg:#07090d;
  --app-panel:#10141b;
  --app-panel-soft:#151b24;
  --app-line:#252d39;
  --app-text:#f4f7fb;
  --app-muted:#9099a8;
  --app-accent:#67e8c2;
  --app-accent-strong:#35cfa6;
  --block-background-fill:#10141b;
  --block-border-color:#252d39;
  --block-label-background-fill:transparent;
  --block-label-text-color:#f4f7fb;
  --input-background-fill:#0c1118;
  --input-border-color:#252d39;
  --body-text-color:#f4f7fb;
  --body-text-color-subdued:#9099a8;
  --button-secondary-background-fill:#121821;
  --button-secondary-background-fill-hover:#19212c;
  --button-secondary-border-color:#2a3441;
  --button-secondary-text-color:#e8edf4;
  max-width:100% !important;
  width:100% !important;
  min-height:100vh !important;
  margin:0 !important;
  padding:20px 28px 40px !important;
  overflow-y:auto !important;
  box-sizing:border-box !important;
  color:var(--app-text) !important;
  background:
    radial-gradient(circle at 12% -8%, rgba(103,232,194,.13), transparent 31%),
    radial-gradient(circle at 95% 12%, rgba(77,126,255,.10), transparent 27%),
    var(--app-bg) !important;
}
footer { display:none !important; }
.hero { display:flex; align-items:center; min-height:52px; }
.hero h1 { margin:0; color:var(--app-text); font-size:1.42rem; font-weight:680; letter-spacing:-.03em; }
.steps { display:flex; width:100%; align-items:center; justify-content:space-between; gap:10px; margin:8px 0 4px; }
.step { min-width:72px; padding:6px 12px; border:1px solid var(--app-line); border-radius:999px;
  background:rgba(16,20,27,.88); color:var(--app-muted); font-size:.72rem; text-align:center; white-space:nowrap; }
.step.done { border-color:rgba(103,232,194,.35); background:rgba(103,232,194,.09); color:#a9f4dd; }
.step.active { border-color:var(--app-accent); background:var(--app-accent); color:#07120f; font-weight:700; }
/* Done page: small thumbnail grid + separate full-width preview (no Gradio in-place zoom). */
#done-gallery-wrap {
  width: 100% !important;
  max-height: none !important;
  overflow: visible !important;
  padding: 10px !important;
  border: 1px solid var(--app-line);
  border-radius: 14px;
  background: rgba(16, 20, 27, 0.55);
}
#done-gallery-wrap .done-gallery {
  max-height: min(28vh, 240px) !important;
  overflow: auto !important;
}
#done-gallery-wrap .grid-container {
  display: grid !important;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)) !important;
  gap: 8px !important;
}
#done-gallery-wrap .grid-container img {
  display: block !important;
  width: 100% !important;
  height: 84px !important;
  object-fit: contain !important;
  object-position: center !important;
  background: #0b0f14 !important;
}
/* Selected full preview: always entire side-by-side image */
#done-full-preview,
#done-full-preview .image-container,
#done-full-preview .image-frame {
  width: 100% !important;
  max-height: min(58vh, 560px) !important;
  background: #0b0f14 !important;
}
#done-full-preview img,
#done-full-preview .image-frame img,
#done-full-preview .image-container img {
  display: block !important;
  width: 100% !important;
  height: auto !important;
  max-width: 100% !important;
  max-height: min(58vh, 560px) !important;
  min-width: 0 !important;
  min-height: 0 !important;
  object-fit: contain !important;
  object-position: center !important;
  margin: 0 auto !important;
}
.done-preview-hint {
  margin: 0 0 8px;
  color: var(--app-muted);
  font-size: 0.78rem;
}
#done-home-action, button.done-home-btn {
  margin-top: 12px !important;
  min-height: 44px !important;
  width: 100% !important;
  max-width: 280px !important;
}
.stats { display:flex; min-height:28px; align-items:center; gap:8px; flex-wrap:wrap; }
.chip { padding:5px 10px; border:1px solid var(--app-line); border-radius:999px;
  background:rgba(16,20,27,.78); color:#c5ccd6; font-size:.76rem; }
.chip.face, .chip.screen, .chip.text { border-color:rgba(103,232,194,.25); background:rgba(103,232,194,.07); color:#b9f6e4; }
.muted { color:var(--app-muted) !important; font-size:.8rem; }
.banner { padding:11px 14px; border:1px solid var(--app-line); border-radius:12px; font-size:.86rem; }
.banner.ok { border-color:rgba(103,232,194,.26); background:rgba(103,232,194,.075); color:#b9f6e4; }
.banner.err { border-color:rgba(251,113,133,.38); background:rgba(251,113,133,.08); color:#fecdd3; }
#setup-stage, #preflight-stage, #report-stage, #back-report-action, #done-home-action, #done-gallery-wrap { display:none !important; }
#run-stage-action { display:none !important; }
#risk-proceed-action { display:none !important; }
.gradio-container:has(.step-setup.active) #setup-stage,
.gradio-container:has(.step-preflight.active) #preflight-stage,
.gradio-container:has(.step-report.active) #report-stage,
.gradio-container:has(.step-done.active) #back-report-action,
.gradio-container:has(.step-done.active) #done-home-action,
.gradio-container:has(.step-done.active) #done-gallery-wrap { display:block !important; }
.gradio-container:has(.step-done.active) #done-full-preview { display:block !important; }
.gradio-container:has(.step-detect.active):not(:has(.run-progress)) #run-stage-action,
.gradio-container:has(.step-anonymise.active):not(:has(.run-progress)) #run-stage-action {
  display:block !important;
}
.gradio-container:has(.run-progress) #run-stage-action { display:none !important; }
/* Keep the progress banner unobstructed while long stages run. */
.gradio-container:has(.run-progress) #setup-stage,
.gradio-container:has(.run-progress) #preflight-stage,
.gradio-container:has(.run-progress) #report-stage {
  display:none !important;
}
.run-progress { padding:13px 14px; border:1px solid rgba(103,232,194,.34); border-radius:13px;
  background:rgba(103,232,194,.075); color:#dffcf3; }
.run-progress-head { display:flex; align-items:center; justify-content:space-between; gap:16px; font-size:.82rem; }
.run-progress-head strong { color:#eafdf7; font-weight:650; }
.run-progress-head span { color:var(--app-accent); font-variant-numeric:tabular-nums; font-weight:750; }
.run-progress-track { height:7px; margin-top:10px; overflow:hidden; border-radius:999px; background:#202934; }
.run-progress-track span { display:block; height:100%; border-radius:inherit;
  background:linear-gradient(90deg,var(--app-accent-strong),var(--app-accent)); transition:width .24s ease; }
.run-progress-note { margin-top:7px; color:var(--app-muted); font-size:.72rem; }
.setup-card, .preflight {
  overflow:visible; padding:14px !important;
  border:1px solid var(--app-line) !important; border-radius:16px !important;
  background:rgba(16,20,27,.94) !important; box-shadow:0 18px 55px rgba(0,0,0,.22);
}
.setup-surface, .setup-surface > div { border:0 !important; background:transparent !important; }
.setup-card > *, .setup-card .html-container, .setup-card .form, .setup-card fieldset,
.setup-card .block, .setup-card .wrap { background:transparent !important; }
.setup-card span[data-testid="block-info"], .setup-card .block-label { padding:0 0 8px !important;
  color:#dce3eb !important; background:transparent !important; font-size:.78rem !important; font-weight:650 !important; }
.setup-card input[type="text"] { border:1px solid var(--app-line) !important; background:#0c1118 !important;
  color:var(--app-text) !important; box-shadow:none !important; }
.focus-card { padding:14px !important; }
.section-title { margin:0 0 10px; color:var(--app-text); font-size:.82rem; font-weight:700; }
.focus-row { align-items:stretch !important; padding:0 !important; border:0 !important;
  background:transparent !important; box-shadow:none !important; }
.focus-row .block, .focus-row .html-container, .focus-row .prose { padding:0 !important; margin:0 !important; }
.focus-row fieldset { min-width:0 !important; }
.focus-row fieldset .wrap { display:flex !important; flex-direction:column !important; align-items:stretch !important; gap:7px !important; }
.focus-row fieldset .wrap label { width:100% !important; height:48px !important; min-height:48px !important; margin:0 !important;
  border:1px solid var(--app-line) !important; border-radius:10px !important; background:#0c1118 !important;
  color:#dce3eb !important; box-shadow:none !important; }
.focus-row fieldset .wrap label:has(input:checked) { border-color:rgba(103,232,194,.62) !important;
  background:rgba(103,232,194,.12) !important; color:#c8faeb !important; }
.focus-row input[type="radio"] { appearance:none !important; width:15px !important; height:15px !important;
  border:2px solid #697483 !important; border-radius:50% !important; background:#0c1118 !important; }
.focus-row input[type="radio"]:checked { border:4px solid var(--app-accent) !important; background:#0c1118 !important; }
.focus-descriptions { display:grid; grid-template-rows:repeat(3, 48px); gap:7px; padding:0 !important; margin:0 !important;
  color:#c2cad5 !important; font-size:.82rem; line-height:1.3; }
.focus-descriptions > div { display:flex; align-items:center; padding-left:14px; border-left:1px solid var(--app-line);
  color:#c2cad5 !important; opacity:1 !important; }
.source-row { align-items:center !important; }
.source-row > div:first-child { border-right:1px solid var(--app-line); padding-right:14px; }
.folder-option { margin-top:4px; padding-left:8px; color:var(--app-text) !important; }
.folder-option label, .folder-option span { color:#dce3eb !important; }
.folder-option .info { color:var(--app-muted) !important; }
.preflight-head h2 { margin:0 0 4px; color:var(--app-text); font-size:1.05rem; }
.preflight-head .sub { margin:0 0 8px; color:var(--app-muted); font-size:.85rem; }
.plan-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; margin-top:12px; overflow:visible; }
.dash-grid { display:grid; grid-template-columns:repeat(5,minmax(150px,1fr)); gap:10px; margin-top:12px; }
.dash-snap-grid { margin-bottom:8px; }
.dash-snap { position:relative; display:flex; min-height:108px; flex-direction:column; align-items:flex-start; gap:7px;
  padding:13px 42px 13px 13px; border:1px solid var(--app-line); border-radius:13px;
  background:var(--app-panel-soft); color:var(--app-text); text-align:left; appearance:none; }
.dash-stage-card { cursor:pointer; transition:border-color .16s ease, background .16s ease, transform .16s ease; }
.dash-stage-card:hover, .dash-stage-card:focus-visible { border-color:rgba(103,232,194,.62);
  background:rgba(103,232,194,.075); transform:translateY(-1px); outline:none; }
.dash-stage { color:var(--app-muted); font-size:.66rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
.dash-method { color:var(--app-text); font-size:.84rem; font-weight:680; line-height:1.3; }
.dash-card-open { position:absolute; top:50%; right:13px; display:grid; width:24px; height:24px;
  place-items:center; border:1px solid var(--app-line); border-radius:50%; color:var(--app-accent);
  font-size:1rem; transform:translateY(-50%); }
.stage-change-row { display:none !important; }
.pick-modal { position:fixed !important; inset:0 !important; z-index:10000 !important; display:none !important;
  align-items:center !important; justify-content:center !important; padding:24px !important;
  overflow:hidden !important; box-sizing:border-box !important; border:0 !important;
  background:rgba(2,5,9,.82) !important; backdrop-filter:blur(7px); }
.pick-modal:has(.form:not(.hidden)) { display:flex !important; }
.pick-modal > .pick-modal { position:static !important; inset:auto !important; z-index:auto !important;
  width:100% !important; height:100% !important; padding:0 !important; background:transparent !important;
  backdrop-filter:none !important; }
.pick-modal-card { width:min(720px,calc(100vw - 40px)) !important; max-height:min(760px,calc(100vh - 40px));
  overflow-x:hidden !important; overflow-y:auto !important; box-sizing:border-box !important;
  padding:20px !important; border:1px solid var(--app-line) !important; border-radius:16px !important;
  background:#10151d !important; box-shadow:0 34px 100px rgba(0,0,0,.62); }
.pick-modal-card > .pick-modal-card { width:100% !important; max-height:none !important; overflow:visible !important;
  padding:0 !important; border:0 !important; border-radius:0 !important; background:transparent !important;
  box-shadow:none !important; }
.pick-modal-card > div, .pick-modal-card .form, .pick-modal-card .block,
.pick-modal-card .html-container, .pick-modal-card .prose {
  border-color:var(--app-line) !important; background:transparent !important; color:var(--app-text) !important;
}
.pick-modal-card .form { gap:12px !important; }
.pick-modal-card h3, .pick-modal-card p, .pick-modal-card label,
.pick-modal-card label span, .pick-modal-card .wrap span { color:var(--app-text) !important; }
.pick-modal-card fieldset { min-width:0 !important; }
.pick-modal-card fieldset .wrap { gap:8px !important; }
.pick-modal-card label { min-height:44px !important; border:1px solid var(--app-line) !important;
  background:#0c1118 !important; white-space:normal !important; overflow-wrap:anywhere; }
.pick-modal-card label:has(input:checked) { border-color:rgba(103,232,194,.62) !important;
  background:rgba(103,232,194,.09) !important; }
.pick-modal-card span[data-testid="block-info"], .pick-modal-card .block-label {
  padding:0 0 6px !important; background:transparent !important; color:#c8d0db !important;
  font-size:.76rem !important; font-weight:650 !important;
}
.picker-note { margin:2px 0 8px; color:var(--app-muted); font-size:.78rem; line-height:1.45; }
html, body { width:100% !important; max-width:100% !important; margin:0 !important; }
.gradio-container, .main, .wrap, .contain { max-width:100% !important; width:100% !important; }
.badge-ok { font-size:.68rem; font-weight:700; color:#052e1c; background:#6ee7b7; padding:3px 8px;
  border-radius:999px; white-space:nowrap; width:fit-content; }
.badge-warn { font-size:.68rem; font-weight:700; color:#451a03; background:#fdba74; padding:3px 8px;
  border-radius:999px; white-space:nowrap; width:fit-content; }
.system-strip { margin-top:10px; padding:10px 12px; border:1px solid var(--app-line); border-radius:12px;
  background:rgba(16,20,27,.55); }
.system-strip ul { margin:6px 0 0; padding-left:18px; color:var(--app-muted); font-size:.78rem; }
.kicker { color:var(--app-accent); font-size:.66rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.detail-panel { margin-top:12px; }
.method-detail { padding:12px 14px; border:1px solid var(--app-line); border-radius:12px; background:var(--app-panel-soft); }
.method-detail-head { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
.method-detail-body p { margin:0 0 8px; color:var(--app-text); font-size:.84rem; line-height:1.45; }
.method-detail-power ul { margin:6px 0 0; padding-left:18px; color:var(--app-muted); font-size:.78rem; }
.stretch-banner, .risk-box { margin-top:10px; padding:12px 14px; border:1px solid #b45309; border-radius:12px;
  background:rgba(251,146,60,.12); color:#ffedd5; font-size:.84rem; line-height:1.45; }
.pick-dialog { width:min(480px,calc(100vw - 34px)); padding:0; border:1px solid #303947; border-radius:16px;
  background:#10151d; color:#f4f7fb; box-shadow:0 30px 80px rgba(0,0,0,.55); }
.pick-dialog::backdrop { background:rgba(2,5,9,.78); backdrop-filter:blur(6px); }
.pick-dialog-head { display:flex; justify-content:space-between; gap:12px; padding:16px 18px 12px;
  border-bottom:1px solid #252d39; }
.pick-dialog-head h3 { margin:4px 0 0; font-size:1.05rem; }
.pick-close { border:1px solid #303947; border-radius:8px; background:#171d26; color:#e8edf4;
  padding:6px 10px; cursor:pointer; font-size:.75rem; }
.method-option-list { display:flex; flex-direction:column; gap:8px; padding:14px 18px; }
.method-option { border:1px solid #303947; border-radius:10px; background:#171d26; color:#e8edf4;
  padding:10px 12px; text-align:left; cursor:pointer; font-size:.86rem; }
.method-option:hover, .method-option.is-selected { border-color:#67e8c2; background:rgba(103,232,194,.08); }
.pick-dialog .muted { padding:0 18px 16px; color:#8f9aaa; font-size:.76rem; }
.plan-card { position:relative; display:flex; min-height:94px; flex-direction:column; align-items:flex-start;
  justify-content:center; padding:14px 42px 14px 14px !important; border:1px solid var(--app-line) !important;
  border-radius:13px !important; background:var(--app-panel-soft) !important; color:var(--app-text) !important;
  text-align:left !important; cursor:pointer !important; box-shadow:none !important; transition:border-color .16s ease, transform .16s ease, background .16s ease; }
.plan-card:hover, .plan-card:focus-visible { border-color:rgba(103,232,194,.58) !important;
  background:rgba(103,232,194,.075) !important; transform:translateY(-1px); outline:none !important; }
.plan-card-stage { display:block; color:var(--app-muted); font-size:.68rem; font-weight:650;
  text-transform:uppercase; letter-spacing:.065em; }
.plan-card-method { display:block; margin-top:6px; color:var(--app-text); font-size:.83rem; font-weight:680; line-height:1.3; }
.plan-card-open { position:absolute; right:13px; top:50%; width:22px; height:22px; overflow:hidden;
  color:transparent; transform:translateY(-50%); }
.plan-card-open::after { content:'+'; position:absolute; inset:0; display:grid; place-items:center;
  border:1px solid var(--app-line); border-radius:50%; color:var(--app-accent); font-size:1rem; font-weight:500; }
.plan-dialog { width:min(620px,calc(100vw - 34px)); padding:0; border:1px solid #303947; border-radius:18px;
  background:#10151d; color:#f4f7fb; box-shadow:0 34px 100px rgba(0,0,0,.62); }
.plan-dialog::backdrop { background:rgba(2,5,9,.78); backdrop-filter:blur(6px); }
.plan-dialog-head { display:flex; align-items:flex-start; justify-content:space-between; gap:18px;
  padding:20px 22px 16px; border-bottom:1px solid #252d39; }
.plan-dialog-head h3 { margin:3px 0 0; color:#f4f7fb; font-size:1.2rem; letter-spacing:-.02em; }
.plan-dialog-kicker { color:#67e8c2; font-size:.66rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; }
.plan-dialog-close, .plan-dialog-done { border:1px solid #303947 !important; border-radius:9px !important;
  background:#171d26 !important; color:#e8edf4 !important; cursor:pointer !important; }
.plan-dialog-close { padding:7px 10px !important; font-size:.75rem !important; }
.plan-detail-list { display:grid; grid-template-columns:145px minmax(0,1fr); gap:0; margin:0; padding:8px 22px; }
.plan-detail-list dt, .plan-detail-list dd { margin:0; padding:12px 0; border-bottom:1px solid #222a35; }
.plan-detail-list dt { color:#8f9aaa; font-size:.75rem; font-weight:650; }
.plan-detail-list dd { color:#dce3eb; font-size:.82rem; line-height:1.45; }
.plan-detail-list code { color:#a9f4dd; font-size:.76rem; overflow-wrap:anywhere; }
.plan-dialog-done { margin:8px 22px 20px; padding:9px 16px !important; }
.review-link a { color:var(--app-accent); font-weight:650; }
.report-card { padding:18px !important; border:1px solid var(--app-line) !important;
  border-radius:16px !important; background:rgba(16,20,27,.94) !important; }
.report-head { display:flex; align-items:baseline; justify-content:space-between; gap:16px;
  padding-bottom:12px; border-bottom:1px solid var(--app-line); }
.report-head span { color:var(--app-text); font-size:1rem; font-weight:720; }
.report-head small { color:var(--app-muted); font-size:.74rem; }
.action-row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.run-action button, button.run-action { width:100% !important; min-height:46px !important;
  border:1px solid var(--app-accent-strong) !important; border-radius:12px !important;
  background:linear-gradient(135deg,var(--app-accent),var(--app-accent-strong)) !important;
  color:#07120f !important; font-weight:750 !important; opacity:1 !important; cursor:pointer !important;
  box-shadow:0 10px 28px rgba(53,207,166,.16) !important; }
.run-action button:hover, button.run-action:hover { filter:brightness(1.05); }
.run-action button:disabled, button.run-action:disabled { cursor:wait !important;
  background:#172029 !important; border-color:#34414e !important; color:#a9f4dd !important; }
button.primary { border-color:var(--app-accent-strong) !important; background:linear-gradient(135deg,var(--app-accent),var(--app-accent-strong)) !important;
  color:#07120f !important; box-shadow:0 10px 28px rgba(53,207,166,.16) !important; }
@media (max-width:720px) {
  .gradio-container { padding:14px 12px 28px !important; }
  .steps { gap:4px; overflow-x:auto; }
  .step { min-width:64px; padding:5px 8px; }
  .dash-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media (max-width:460px) {
  .plan-grid { grid-template-columns:1fr; }
  .dash-grid { grid-template-columns:1fr; }
  .focus-row, .source-row { flex-direction:column !important; }
  .focus-descriptions { padding-top:0; }
  .focus-descriptions > div { min-height:38px; }
  .source-row > div:first-child { border-right:0; border-bottom:1px solid var(--app-line); padding:0 0 14px; }
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Egocentric Privacy Pipeline") as demo:
        run_dir = gr.State("")
        step = gr.State("setup")

        gr.HTML(
            f"""
            <div class="hero">
              <div>
                <h1>Egocentric Privacy Pipeline</h1>
              </div>
            </div>
            """
        )
        step_bar = gr.HTML(_step_bar("setup"))
        stats = gr.HTML(_stats_html(""))
        banner = gr.HTML(_msg("Choose a protection profile, then scan your source images"))

        with gr.Group(
            visible=True,
            elem_id="setup-stage",
            elem_classes=["setup-surface"],
        ) as setup_group:
            with gr.Group(elem_classes=["setup-card", "focus-card"]):
                gr.HTML('<div class="section-title">Protection profile</div>')
                with gr.Row(elem_classes=["focus-row"]):
                    focus = gr.Radio(
                        choices=FOCUS_CHOICES,
                        value="Balanced",
                        show_label=False,
                        container=False,
                        scale=1,
                    )
                    gr.HTML(
                        """
                        <div class="focus-descriptions">
                          <div>Maximum privacy protection.</div>
                          <div>Evidence-supported privacy and utility trade-off.</div>
                          <div>Preserve visual utility where eligible.</div>
                        </div>
                        """,
                        scale=1,
                    )
            with gr.Row(elem_classes=["setup-card", "source-row"]):
                source_dir = gr.Textbox(
                    label="Folder of source images",
                    value=DEFAULT_INPUT,
                    scale=1,
                )
                recursive = gr.Checkbox(
                    label="Include subdirectories - Also scan nested folders.",
                    value=False,
                    scale=1,
                    elem_classes=["folder-option"],
                )
            scan_btn = gr.Button("Scan & show plan", variant="primary")

        with gr.Group(visible=False, elem_id="preflight-stage") as preflight_group:
            preflight_panel = gr.HTML("")
            with gr.Row(elem_classes=["stage-change-row"]):
                btn_face_det = gr.Button(
                    "Face detection",
                    elem_id="stage-trigger-face-detection",
                    elem_classes=["stage-trigger"],
                )
                btn_mm_det = gr.Button(
                    "Screen and text detection",
                    elem_id="stage-trigger-multimodal-detection",
                    elem_classes=["stage-trigger"],
                )
                btn_face_anon = gr.Button(
                    "Face anonymisation",
                    elem_id="stage-trigger-face-anonymisation",
                    elem_classes=["stage-trigger"],
                )
                btn_screen_op = gr.Button(
                    "Screen redaction",
                    elem_id="stage-trigger-screen-operator",
                    elem_classes=["stage-trigger"],
                )
                btn_text_op = gr.Button(
                    "Text redaction",
                    elem_id="stage-trigger-text-operator",
                    elem_classes=["stage-trigger"],
                )
            with gr.Group(visible=False, elem_classes=["pick-modal"]) as pick_modal:
                with gr.Group(elem_classes=["pick-modal-card"]):
                    pick_title = gr.Markdown("### Select a method")
                    gr.HTML(
                        "<p class='picker-note'>Recommendation status is calculated from the "
                        "available compute. Selecting a method updates the estimated run time.</p>"
                    )
                    pick_stage = gr.State("")
                    pick_radio = gr.Radio(choices=[], label="Available methods", interactive=True)
                    with gr.Row():
                        pick_cancel = gr.Button("Close")
                        pick_apply = gr.Button("Use this method", variant="primary")
            risk_box = gr.HTML(visible=False)
            risk_check = gr.Checkbox(
                label="I understand the risk and want to run Not recommended methods on this system",
                value=False,
                visible=False,
            )
            with gr.Row(elem_classes=["action-row"]):
                back_btn = gr.Button("Back")
                proceed_btn = gr.Button("Proceed to Detect", variant="primary")
                risk_proceed_btn = gr.Button(
                    "",
                    visible=False,
                    elem_id="risk-proceed-action",
                )

        run_btn = gr.Button(
            "Run detection",
            variant="primary",
            visible=True,
            interactive=False,
            elem_id="run-stage-action",
            elem_classes=["run-action"],
        )
        skip_btn = gr.Button("Skip review", visible=False)
        review_btn = gr.Button("Review boxes", visible=False)
        finish_review_btn = gr.Button("Finish review", visible=False, variant="primary")

        review_link = gr.HTML(visible=False)
        with gr.Group(
            visible=False,
            elem_id="report-stage",
            elem_classes=["report-card"],
        ) as report_group:
            gr.HTML(
                "<div class='report-head'><span>Run report</span>"
                "<small>Review the processing summary before finishing.</small></div>"
            )
            report_body = gr.Markdown("")
            finish_report_btn = gr.Button("Continue to Done", variant="primary")

        back_report_btn = gr.Button(
            "Back to Report",
            visible=True,
            elem_id="back-report-action",
        )
        with gr.Group(visible=True, elem_id="done-gallery-wrap", elem_classes=["done-gallery-wrap"]):
            gr.HTML(
                "<p class='done-preview-hint'>Click a thumbnail to open the full before/after image below "
                "(entire pair is shown, not cropped).</p>"
            )
            gallery = gr.Gallery(
                label="Thumbnails (maximum 10)",
                columns=4,
                rows=None,
                height=200,
                object_fit="contain",
                fit_columns=True,
                # Critical: Gradio in-place preview fills a short box and crops wide SBS images.
                preview=False,
                allow_preview=False,
                visible=False,
                elem_id="done-preview-gallery",
                elem_classes=["done-gallery"],
            )
            selected_preview = gr.Image(
                label="Full before / after",
                type="filepath",
                visible=False,
                interactive=False,
                buttons=["download"],
                elem_id="done-full-preview",
                elem_classes=["done-full-preview"],
            )
            done_home_btn = gr.Button(
                "Done",
                variant="primary",
                visible=True,
                elem_id="done-home-action",
                elem_classes=["done-home-btn"],
            )

        stage_btns = [btn_face_det, btn_mm_det, btn_face_anon, btn_screen_op, btn_text_op]

        outs = [
            run_dir,
            step,
            step_bar,
            stats,
            banner,
            preflight_panel,
            setup_group,
            preflight_group,
            scan_btn,
            proceed_btn,
            back_btn,
            run_btn,
            skip_btn,
            review_btn,
            finish_review_btn,
            gallery,
            review_link,
            report_group,
            report_body,
            finish_report_btn,
            back_report_btn,
            risk_box,
            risk_check,
            risk_proceed_btn,
            *stage_btns,
            pick_modal,
            done_home_btn,
            selected_preview,
        ]

        def wrap_simple(fn, *args):
            return normalize_ui_outputs(fn(*args))

        def stream_worker(initial_rd: str, current_step: str, initial_message: str, work):
            """Run ``work(report_progress)`` on a thread and stream progress yields."""
            events: Queue = Queue()

            def report_progress(fraction: float, message: str) -> None:
                events.put(("progress", fraction, message))

            def worker() -> None:
                # ui_scan / ui_run_step already convert exceptions into banner tuples.
                result = work(report_progress)
                events.put(("complete", result))

            Thread(target=worker, daemon=True).start()
            yield running_ui_result(initial_rd, current_step, 0.0, initial_message)

            while True:
                event = events.get()
                if event[0] == "progress":
                    _, fraction, message = event
                    yield running_ui_result(initial_rd, current_step, fraction, message)
                    continue
                # Restore full UI (including preflight method cards) after progress hid it.
                yield normalize_ui_outputs(event[1])
                return

        def stream_run(rd: str, current_step: str):
            initial_message = (
                "Starting face, screen, and text detection"
                if current_step == "detect"
                else "Starting anonymisation"
            )

            def work(report_progress):
                return ui_run_step(rd, current_step, progress_callback=report_progress)

            yield from stream_worker(rd, current_step, initial_message, work)

        def stream_scan(src, foc, rec):
            def work(report_progress):
                return ui_scan(src, foc, rec, progress_callback=report_progress)

            yield from stream_worker("", "setup", "Starting scan of source folder", work)

        def stream_proceed_and_detect(rd, risk_ok):
            yield running_ui_result(
                rd, "preflight", 0.05, "Validating preflight and detector readiness"
            )
            proceed_result = ui_proceed(rd, risk_ok=bool(risk_ok))
            if proceed_result[1] != "detect":
                # Risk gate or error: must restore preflight method selection UI.
                yield normalize_ui_outputs(proceed_result)
                return
            yield running_ui_result(rd, "detect", 0.0, "Preflight accepted · starting detection")
            yield from stream_run(rd, "detect")

        scan_btn.click(
            fn=stream_scan,
            inputs=[source_dir, focus, recursive],
            outputs=outs,
            show_progress="hidden",
        )
        back_btn.click(
            fn=lambda: wrap_simple(ui_back), inputs=[], outputs=outs, show_progress="hidden"
        )
        proceed_btn.click(
            fn=stream_proceed_and_detect,
            inputs=[run_dir, risk_check],
            outputs=outs,
            show_progress="hidden",
        )
        def _open_pick(stage_key: str):
            def _fn(rd):
                return ui_open_stage_picker(rd, stage_key)

            return _fn

        btn_face_det.click(
            fn=_open_pick("face_detection"),
            inputs=[run_dir],
            outputs=[pick_modal, pick_title, pick_radio, pick_stage],
            show_progress="hidden",
        )
        btn_mm_det.click(
            fn=_open_pick("multimodal_detection"),
            inputs=[run_dir],
            outputs=[pick_modal, pick_title, pick_radio, pick_stage],
            show_progress="hidden",
        )
        btn_face_anon.click(
            fn=_open_pick("face_anonymisation"),
            inputs=[run_dir],
            outputs=[pick_modal, pick_title, pick_radio, pick_stage],
            show_progress="hidden",
        )
        btn_screen_op.click(
            fn=_open_pick("screen_operator"),
            inputs=[run_dir],
            outputs=[pick_modal, pick_title, pick_radio, pick_stage],
            show_progress="hidden",
        )
        btn_text_op.click(
            fn=_open_pick("text_operator"),
            inputs=[run_dir],
            outputs=[pick_modal, pick_title, pick_radio, pick_stage],
            show_progress="hidden",
        )
        pick_apply.click(
            fn=ui_apply_stage_pick,
            inputs=[run_dir, pick_stage, pick_radio],
            outputs=[
                preflight_panel,
                banner,
                pick_modal,
                btn_face_det,
                btn_mm_det,
                btn_face_anon,
                btn_screen_op,
                btn_text_op,
                pick_stage,
            ],
            show_progress="hidden",
        )
        pick_cancel.click(
            fn=ui_close_stage_picker,
            inputs=[],
            outputs=[pick_modal, pick_stage],
            show_progress="hidden",
        )
        run_btn.click(
            fn=stream_run,
            inputs=[run_dir, step],
            outputs=outs,
            show_progress="hidden",
        )
        skip_btn.click(
            fn=lambda rd: wrap_simple(ui_skip, rd),
            inputs=[run_dir],
            outputs=outs,
            show_progress="hidden",
        )
        review_btn.click(
            fn=lambda rd: wrap_simple(ui_review, rd),
            inputs=[run_dir],
            outputs=outs,
            show_progress="hidden",
        )
        finish_review_btn.click(
            fn=lambda rd: wrap_simple(ui_finish_review, rd),
            inputs=[run_dir],
            outputs=outs,
            show_progress="hidden",
        )
        finish_report_btn.click(
            fn=lambda rd: wrap_simple(ui_finish_report, rd),
            inputs=[run_dir],
            outputs=outs,
            show_progress="hidden",
        )
        def _resolve_gallery_path(item) -> str | None:
            if item is None:
                return None
            if isinstance(item, (list, tuple)) and item:
                return _resolve_gallery_path(item[0])
            if isinstance(item, dict):
                for key in ("image", "name", "path", "orig_name"):
                    if item.get(key):
                        return str(item[key])
                return None
            return str(item)

        def ui_gallery_select(gallery_value, evt: gr.SelectData):
            """Show the full side-by-side image when a thumbnail is clicked."""
            if gallery_value is None:
                return gr.update()
            try:
                idx = int(evt.index) if evt is not None else 0
            except Exception:
                idx = 0
            if not isinstance(gallery_value, (list, tuple)) or idx < 0 or idx >= len(gallery_value):
                return gr.update()
            path = _resolve_gallery_path(gallery_value[idx])
            if not path:
                return gr.update()
            return gr.update(value=path, visible=True)

        gallery.select(
            fn=ui_gallery_select,
            inputs=[gallery],
            outputs=[selected_preview],
            show_progress="hidden",
        )
        done_home_btn.click(
            fn=lambda: wrap_simple(ui_done_to_setup),
            inputs=[],
            outputs=outs,
            show_progress="hidden",
        )
        back_report_btn.click(
            fn=lambda rd: wrap_simple(ui_back_to_report, rd),
            inputs=[run_dir],
            outputs=outs,
            show_progress="hidden",
        )

    return demo


def main() -> None:
    from privacy_pipeline_app.runtime_env import configure_app_runtime

    configure_app_runtime()
    build_app().launch(theme=gr.themes.Soft(), css=CSS)


if __name__ == "__main__":
    main()
