#!/usr/bin/env python3
"""Multi-step privacy pipeline workflow for research handover.

Stages: setup → scan → detect → (optional review) → anonymise → report
State is persisted under app/outputs/runs/<run_id>/ so work can pause between steps.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
_APP_SRC = ROOT / "app" / "src"
for _p in (str(ROOT), str(_APP_SRC)):
    if _p not in __import__("sys").path:
        __import__("sys").path.insert(0, _p)

from privacy_pipeline_app.production_runner import (  # noqa: E402
    FIXED_MODES,
    SELECTOR_OBJECTIVES,
    SUPPORTED_EXTENSIONS,
    probe_environment,
)

# Module-level caches so detectors are not reloaded per image
_THESIS_FACE_DETECTORS: dict[str, Any] = {}
_MULTIMODAL_STACKS: dict[str, Any] = {}
_DETECT_WARNINGS: list[str] = []

APP_RUNS = ROOT / "app" / "outputs" / "runs"
APP_INPUTS = ROOT / "app" / "inputs"
DONE_PREVIEW_LIMIT = 10

STAGES = ["setup", "scan", "detect", "review", "anonymise", "report"]

# Seconds per image estimates by method family and device class
ETA_SECONDS = {
    "cpu": {
        "blur": 0.35,
        "pixelate": 0.35,
        "solid_mask": 0.25,
        "layered": 0.55,
        "objective_profile": 0.70,
        "oapr": 0.70,  # legacy alias
        "copy": 0.05,
    },
    "cuda": {
        "blur": 0.12,
        "pixelate": 0.12,
        "solid_mask": 0.08,
        "layered": 0.18,
        "objective_profile": 0.25,
        "oapr": 0.25,  # legacy alias
        "copy": 0.02,
    },
}

METHOD_REASONS = {
    "blur": {
        "privacy": "Reduces facial detail and lowers re-identification risk while keeping scene structure.",
        "utility": "Preserves most non-face context; lower utility cost than solid masking.",
        "compute": "Fast deterministic operator; suitable for constrained hardware.",
    },
    "pixelate": {
        "privacy": "Coarsens face regions; weaker privacy than solid mask or layered methods.",
        "utility": "Keeps coarse scene layout; often preferred when utility is prioritised.",
        "compute": "Very low compute cost.",
    },
    "solid_mask": {
        "privacy": "Strongest deterministic face identity suppression among default methods.",
        "utility": "High utility cost on face regions; scene outside boxes is preserved.",
        "compute": "Minimal compute; fill operation only.",
    },
    "layered": {
        "privacy": "Combines blur, downscale and light noise for stronger privacy than plain blur.",
        "utility": "Better scene retention than solid mask; visual-safe default for balanced use.",
        "compute": "Slightly higher than blur; still deterministic and fast.",
    },
    "copy": {
        "privacy": "No face action when no face risk detected; residual unknown risk may remain.",
        "utility": "Full utility preservation for that frame.",
        "compute": "Copy only.",
    },
}


@dataclass
class WizardState:
    run_id: str
    run_dir: str
    source_dir: str = ""
    recursive: bool = True
    strategy: str = "objective_profile"  # objective_profile or fixed; oapr is an alias
    fixed_method: str = "layered"
    objective: str = "privacy_first"
    include_multimodal: bool = True
    stages_done: dict[str, bool] = field(default_factory=lambda: {s: False for s in STAGES})
    n_images: int = 0
    n_faces: int = 0
    n_texts: int = 0
    n_screens: int = 0
    n_images_with_faces: int = 0
    n_images_with_text: int = 0
    n_images_with_screen: int = 0
    scan_summary: dict[str, Any] = field(default_factory=dict)
    detect_summary: dict[str, Any] = field(default_factory=dict)
    anonymise_summary: dict[str, Any] = field(default_factory=dict)
    eta_seconds: float = 0.0
    recommendation: str = ""
    message: str = ""
    updated_at: str = ""
    focus: str = "balanced"  # privacy | balanced | utility
    plan: dict[str, Any] = field(default_factory=dict)
    preflight_accepted: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def save_state(state: WizardState) -> None:
    state.updated_at = utc_now()
    write_json(Path(state.run_dir) / "state.json", asdict(state))


def load_state(run_dir: Path) -> WizardState:
    from dataclasses import fields as dc_fields

    data = read_json(state_path(run_dir))
    known = {f.name for f in dc_fields(WizardState)}
    filtered = {k: v for k, v in data.items() if k in known}
    return WizardState(**filtered)


def list_images(
    source: Path,
    recursive: bool,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[Path]:
    """List supported images under ``source``.

    Optional ``progress_callback(fraction, message)`` reports crawl progress.
    Total entry count is unknown up front, so the fraction grows asymptotically
    toward 0.9 while scanning, then reaches 1.0 after sorting.
    """

    def emit(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(1.0, fraction)), message)

    emit(0.0, "Starting folder scan")
    iterator = source.rglob("*") if recursive else source.glob("*")
    images: list[Path] = []
    checked = 0
    for path in iterator:
        checked += 1
        try:
            is_file = path.is_file()
        except OSError:
            continue
        if is_file and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
        if checked == 1 or checked % 20 == 0:
            # Unknown total: asymptotic progress so the bar keeps moving.
            fraction = 0.85 * (1.0 - pow(2.718281828, -checked / 120.0))
            emit(
                fraction,
                f"Scanning folder… {len(images)} images found ({checked} entries checked)",
            )
    emit(0.92, f"Sorting {len(images)} images")
    images = sorted(images)
    emit(1.0, f"Found {len(images)} images")
    return images


def pipeline_markdown(state: WizardState | None) -> str:
    labels = {
        "setup": "1 Setup",
        "scan": "2 Scan",
        "detect": "3 Detect",
        "review": "4 Review",
        "anonymise": "5 Anonymise",
        "report": "6 Report",
    }
    if state is None:
        parts = [f"🔴 **{labels[s]}**" for s in STAGES]
        return " → ".join(parts)
    parts = []
    for s in STAGES:
        done = state.stages_done.get(s, False)
        # review is optional: show amber if detect done but not reviewed
        if s == "review" and state.stages_done.get("detect") and not done and not state.stages_done.get("anonymise"):
            parts.append(f"🟡 **{labels[s]}** (optional)")
        elif done:
            parts.append(f"🟢 **{labels[s]}**")
        else:
            parts.append(f"🔴 **{labels[s]}**")
    return " → ".join(parts)


def map_objective(focus: str) -> str:
    focus = (focus or "balanced").strip().lower()
    if focus in {"privacy", "privacy-focused", "privacy_first"}:
        return "privacy_first"
    if focus in {"utility", "utility-focused", "utility_priority"}:
        return "utility_priority"
    if focus in {"balanced", "utility_under_privacy_floor"}:
        return "utility_under_privacy_floor"
    if focus in SELECTOR_OBJECTIVES:
        return focus
    return "privacy_first"


def device_class(env: dict[str, Any]) -> str:
    return "cuda" if env.get("cuda_available") else "cpu"


def estimate_eta(n_images: int, strategy: str, fixed_method: str, env: dict[str, Any]) -> tuple[float, str]:
    cls = device_class(env)
    table = ETA_SECONDS[cls]
    if strategy == "fixed":
        method = fixed_method if fixed_method in table else "layered"
        sec = n_images * table[method]
        rec = f"Fixed method `{method}`. ETA ~{sec/60:.1f} min on {cls}."
    else:
        sec = n_images * table.get("objective_profile", table.get("oapr", 0.25))
        rec = (
            f"Recommended: **objective profile** (App fixed visual-safe defaults per Privacy/Balanced/Utility). "
            f"Not the scientific OAPR 286/81/133 condition-aware router. "
            f"ETA ~{sec/60:.1f} min on {cls}."
        )
    return float(sec), rec


def create_run(
    source_dir: str,
    recursive: bool,
    strategy: str,
    fixed_method: str,
    focus: str,
    include_multimodal: bool = True,
    progress_callback: Callable[[float, str], None] | None = None,
) -> WizardState:
    from privacy_pipeline_app.objective_policy import resolve_plan
    from privacy_pipeline_app.runtime_policy import estimate_runtime, select_runtime_policy

    def emit(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(1.0, fraction)), message)

    emit(0.02, "Validating source folder")
    source = Path(source_dir).expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")
    run_id = utc_stamp()
    run_dir = APP_RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "detections").mkdir(exist_ok=True)
    (run_dir / "anonymised").mkdir(exist_ok=True)
    (run_dir / "side_by_side").mkdir(exist_ok=True)
    (run_dir / "metadata").mkdir(exist_ok=True)
    (run_dir / "report").mkdir(exist_ok=True)

    emit(0.05, "Scanning source folder for images")
    images = list_images(
        source,
        recursive,
        progress_callback=lambda fraction, message: emit(0.05 + fraction * 0.55, message),
    )
    emit(0.65, "Probing system compute profile")
    env = probe_environment()
    emit(0.75, "Building objective plan and runtime estimate")
    from src.policy.registry import get_app_policy_semantics

    plan = resolve_plan(focus)
    objective = plan.objective_id
    runtime_policy = select_runtime_policy(env)
    runtime_estimate = estimate_runtime(len(images), runtime_policy, env, APP_RUNS)
    eta = runtime_estimate.total_seconds
    policy_semantics = get_app_policy_semantics()
    rec = (
        f"{plan.title}: App **objective_profile** (not scientific OAPR "
        f"`{policy_semantics['scientific_policy_id']}`)."
    )
    _ = include_multimodal
    # Resolve strategy aliases.
    strategy_norm = (strategy or "objective_profile").strip().lower()
    legacy_alias_requested = strategy_norm in {"oapr", "profile"}
    if strategy_norm in {"oapr", "objective_profile", "profile"}:
        strategy_norm = "objective_profile"
    _ = fixed_method

    # Face method from plan (visual-safe defaults). Keep setup/plan/state aligned.
    face_method = plan.face_anonymisation.method_id
    if face_method not in FIXED_MODES:
        face_method = "layered"

    plan_payload = plan.to_dict()
    # Keep the setup and stage methods aligned.
    for stage in plan_payload.get("stages") or []:
        if stage.get("stage") == "Face anonymisation":
            stage["method_id"] = face_method
            break
    plan_payload["runtime_policy_id"] = runtime_policy.policy_id
    plan_payload["runtime_policy"] = runtime_policy.to_dict()
    plan_payload["eta_source"] = runtime_estimate.source
    plan_payload["eta_seconds_per_image"] = runtime_estimate.seconds_per_image
    plan_payload["app_policy_id"] = policy_semantics["app_policy_id"]
    plan_payload["scientific_policy_id"] = policy_semantics["scientific_policy_id"]
    plan_payload["simplification"] = policy_semantics["simplification"]
    plan_payload["legacy_alias_requested"] = legacy_alias_requested
    plan_payload["legacy_alias_note"] = (
        "Strategy alias resolved to objective_profile."
        if legacy_alias_requested
        else None
    )
    emit(0.90, "Saving run workspace")
    state = WizardState(
        run_id=run_id,
        run_dir=str(run_dir),
        source_dir=str(source),
        recursive=recursive,
        strategy=strategy_norm,
        fixed_method=face_method,
        objective=objective,
        include_multimodal=True,
        n_images=len(images),
        eta_seconds=eta,
        recommendation=rec,
        message=f"Run created · {plan.title} · objective_profile · {len(images)} images · review preflight before detect.",
        focus=plan.focus,
        plan=plan_payload,
        preflight_accepted=False,
    )
    state.stages_done["setup"] = True
    save_state(state)
    write_json(run_dir / "metadata" / "system_profile.json", env)
    write_json(run_dir / "metadata" / "objective_plan.json", plan_payload)
    write_json(run_dir / "metadata" / "policy_semantics.json", policy_semantics)
    write_json(
        run_dir / "metadata" / "setup.json",
        {
            "source_dir": str(source),
            "focus": plan.focus,
            "objective": state.objective,
            "plan_title": plan.title,
            "face_method": face_method,
            "app_policy_id": policy_semantics["app_policy_id"],
            "scientific_policy_id": policy_semantics["scientific_policy_id"],
            "simplification": policy_semantics["simplification"],
            "runtime_policy": runtime_policy.to_dict(),
            "eta_source": runtime_estimate.source,
            "eta_seconds": eta,
            "n_images": len(images),
        },
    )
    emit(1.0, f"Run created · {len(images)} images")
    return state


def accept_preflight(run_dir: str) -> WizardState:
    state = load_state(Path(run_dir))
    if not state.stages_done.get("scan"):
        raise RuntimeError("Run Scan before accepting preflight.")
    from privacy_pipeline_app.thesis_face_detector import resolve_runnable_face_policy

    preferred = str(
        ((state.plan.get("runtime_policy") or {}).get("face_policy_id"))
        or "runtime_3_source_all_raw_rf_approximation"
    )
    # Apply the selected catalog method.
    for stage in state.plan.get("stages") or []:
        if stage.get("stage") in {"Face detection", "face_detection"} and stage.get("method_id"):
            preferred = str(stage["method_id"])
            break
    runtime = resolve_runnable_face_policy(preferred)
    # Persist the policy that will actually run so Detect does not re-request a missing tier.
    rp = dict(state.plan.get("runtime_policy") or {})
    rp["face_policy_id"] = runtime["policy_id"]
    rp["face_display_name"] = runtime.get("policy_id", "")
    if runtime.get("fallback_applied"):
        rp["face_policy_fallback_from"] = preferred
        rp["face_policy_fallback_reason"] = runtime.get("fallback_reason", "")
    state.plan["runtime_policy"] = rp
    state.plan["runtime_policy_id"] = str(
        state.plan.get("runtime_policy_id") or rp.get("policy_id") or ""
    )
    for stage in state.plan.get("stages") or []:
        if stage.get("stage") in {"Face detection", "face_detection"}:
            stage["method_id"] = runtime["policy_id"]
            stage["display_name"] = runtime["policy_id"]
            if runtime.get("fallback_applied"):
                stage["why"] = runtime.get("fallback_reason", stage.get("why", ""))
            break
    write_json(Path(state.run_dir) / "metadata" / "detector_preflight.json", runtime)
    write_json(Path(state.run_dir) / "metadata" / "objective_plan.json", state.plan)
    state.preflight_accepted = True
    if runtime.get("fallback_applied"):
        state.message = (
            f"Preflight accepted · detector stepped down to `{runtime['policy_id']}` "
            f"(requested `{preferred}` unavailable)."
        )
    else:
        state.message = "Preflight accepted · ready to detect with the selected plan."
    save_state(state)
    return state


def step_scan(
    run_dir: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> WizardState:
    def emit(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(1.0, fraction)), message)

    emit(0.02, "Loading run state for scan")
    state = load_state(Path(run_dir))
    source = Path(state.source_dir)
    emit(0.05, "Re-scanning source folder")
    images = list_images(
        source,
        state.recursive,
        progress_callback=lambda fraction, message: emit(0.05 + fraction * 0.35, message),
    )
    by_ext: Counter[str] = Counter(p.suffix.lower() for p in images)
    # Write input manifest
    manifest = Path(state.run_dir) / "input_manifest.csv"
    total_images = max(len(images), 1)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "relative_path", "local_path", "bytes"])
        writer.writeheader()
        for index, p in enumerate(images, start=1):
            try:
                rel = str(p.relative_to(source))
            except ValueError:
                rel = p.name
            try:
                size_bytes = p.stat().st_size
            except OSError:
                size_bytes = 0
            writer.writerow(
                {
                    "image_id": p.name,
                    "relative_path": rel,
                    "local_path": str(p),
                    "bytes": size_bytes,
                }
            )
            if index == 1 or index == len(images) or index % 5 == 0:
                emit(
                    0.42 + 0.40 * (index / total_images),
                    f"Writing image manifest {index} of {len(images)}",
                )
    emit(0.85, "Estimating runtime from system profile")
    state.n_images = len(images)
    env = probe_environment()
    from privacy_pipeline_app.runtime_policy import estimate_runtime, select_runtime_policy

    runtime_policy = select_runtime_policy(env)
    runtime_estimate = estimate_runtime(len(images), runtime_policy, env, APP_RUNS)
    eta = runtime_estimate.total_seconds
    state.eta_seconds = eta
    state.recommendation = f"{runtime_policy.tier.title()} runtime policy selected from available compute."
    state.plan["runtime_policy_id"] = runtime_policy.policy_id
    state.plan["runtime_policy"] = runtime_policy.to_dict()
    state.plan["eta_source"] = runtime_estimate.source
    state.plan["eta_seconds_per_image"] = runtime_estimate.seconds_per_image
    state.scan_summary = {
        "n_images": len(images),
        "extensions": dict(by_ext),
        "source_dir": str(source),
        "manifest": str(manifest),
        "eta_seconds": eta,
        "eta_minutes": round(eta / 60.0, 2),
        "recommendation": state.recommendation,
        "runtime_policy": runtime_policy.to_dict(),
        "eta_source": runtime_estimate.source,
    }
    state.stages_done["scan"] = True
    state.message = f"Scan complete: {len(images)} images."
    save_state(state)
    write_json(Path(state.run_dir) / "metadata" / "scan_summary.json", state.scan_summary)
    emit(1.0, f"Scan complete: {len(images)} images")
    return state


def _get_thesis_face_detector(policy_id: str):
    if policy_id not in _THESIS_FACE_DETECTORS:
        from privacy_pipeline_app.thesis_face_detector import ThesisFaceDetector

        _THESIS_FACE_DETECTORS[policy_id] = ThesisFaceDetector(policy_id=policy_id)
    return _THESIS_FACE_DETECTORS[policy_id]


def _detect_faces_best(
    image_rgb: Image.Image,
) -> tuple[list[tuple[int, int, int, int, float]], dict[str, Any]]:
    """Run the bounded three-source face policy selected for the App."""
    return _get_thesis_face_detector("runtime_3_source_all_raw_rf_approximation").detect(image_rgb)


def _get_multimodal_stack(policy: dict[str, Any]):
    """Precision-localised multimodal stack selected for the runtime tier."""
    policy_id = str(policy.get("multimodal_policy_id") or "reviewed_screen_yolo11s_1280")
    if policy_id not in _MULTIMODAL_STACKS:
        from src.detection.multimodal_precision_stack import MultimodalPrecisionStack

        _MULTIMODAL_STACKS[policy_id] = MultimodalPrecisionStack(
            image_size=int(policy.get("multimodal_image_size") or 1280),
            text_canvas_size=int(policy.get("text_canvas_size") or 2560),
            text_confidence=float(policy.get("text_confidence") or 0.30),
            text_use_gpu=bool(policy.get("text_use_gpu", True)),
        )
    return _MULTIMODAL_STACKS[policy_id]


BOX_CSV_FIELDS = ["image_id", "local_path", "x1", "y1", "x2", "y2", "score", "source"]


def write_detection_artifacts(det_dir: Path, records: list[dict[str, Any]]) -> None:
    """Write detections.jsonl plus separate face/screen/text CSVs."""
    det_dir.mkdir(parents=True, exist_ok=True)
    det_path = det_dir / "detections.jsonl"
    face_csv = det_dir / "face_boxes.csv"
    screen_csv = det_dir / "screen_boxes.csv"
    text_csv = det_dir / "text_boxes.csv"

    with (
        det_path.open("w", encoding="utf-8") as dj,
        face_csv.open("w", newline="", encoding="utf-8") as ff,
        screen_csv.open("w", newline="", encoding="utf-8") as sf,
        text_csv.open("w", newline="", encoding="utf-8") as tf,
    ):
        fw = csv.DictWriter(ff, fieldnames=BOX_CSV_FIELDS)
        sw = csv.DictWriter(sf, fieldnames=BOX_CSV_FIELDS)
        tw = csv.DictWriter(tf, fieldnames=BOX_CSV_FIELDS)
        fw.writeheader()
        sw.writeheader()
        tw.writeheader()
        for rec in records:
            dj.write(json.dumps(rec) + "\n")
            image_id = rec["image_id"]
            local_path = rec["local_path"]
            face_src = rec.get("detector", "face")
            screen_src = "+".join(rec.get("screen_sources") or []) or "screen"
            text_src = rec.get("text_policy") or "recognised_text_ocr"
            for b in rec.get("faces", []):
                fw.writerow(
                    {
                        "image_id": image_id,
                        "local_path": local_path,
                        "x1": b["x1"],
                        "y1": b["y1"],
                        "x2": b["x2"],
                        "y2": b["y2"],
                        "score": b.get("score", 1.0),
                        "source": face_src,
                    }
                )
            for b in rec.get("screens", []):
                sw.writerow(
                    {
                        "image_id": image_id,
                        "local_path": local_path,
                        "x1": b["x1"],
                        "y1": b["y1"],
                        "x2": b["x2"],
                        "y2": b["y2"],
                        "score": b.get("score", 1.0),
                        "source": screen_src,
                    }
                )
            for b in rec.get("texts", []):
                tw.writerow(
                    {
                        "image_id": image_id,
                        "local_path": local_path,
                        "x1": b["x1"],
                        "y1": b["y1"],
                        "x2": b["x2"],
                        "y2": b["y2"],
                        "score": b.get("score", 1.0),
                        "source": text_src,
                    }
                )


def load_detection_records(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "detections" / "detections.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def refresh_state_from_detections(run_dir: str) -> WizardState:
    """Recompute face/text/screen counts after manual review edits."""
    state = load_state(Path(run_dir))
    records = load_detection_records(run_dir)
    n_faces = n_texts = n_screens = 0
    n_img_faces = n_img_text = n_img_screen = 0
    for rec in records:
        fc, tc, sc = len(rec.get("faces", [])), len(rec.get("texts", [])), len(rec.get("screens", []))
        n_faces += fc
        n_texts += tc
        n_screens += sc
        if fc:
            n_img_faces += 1
        if tc:
            n_img_text += 1
        if sc:
            n_img_screen += 1
    state.n_faces = n_faces
    state.n_texts = n_texts
    state.n_screens = n_screens
    state.n_images_with_faces = n_img_faces
    state.n_images_with_text = n_img_text
    state.n_images_with_screen = n_img_screen
    if state.detect_summary:
        state.detect_summary.update(
            {
                "n_faces": n_faces,
                "n_texts": n_texts,
                "n_screens": n_screens,
                "n_images_with_faces": n_img_faces,
                "n_images_with_text": n_img_text,
                "n_images_with_screen": n_img_screen,
                "reviewed": True,
            }
        )
    state.message = (
        f"Detections updated after review: {n_faces} faces, {n_texts} text, {n_screens} screens."
    )
    save_state(state)
    write_json(Path(state.run_dir) / "metadata" / "detect_summary.json", state.detect_summary)
    return state


def step_detect(
    run_dir: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> WizardState:
    """Run the hardware-selected face and precision-localised multimodal policies."""
    global _DETECT_WARNINGS
    _DETECT_WARNINGS = []

    def emit(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(1.0, fraction)), message)

    emit(0.01, "Validating the run and loading its image manifest")
    state = load_state(Path(run_dir))
    if not state.stages_done.get("scan"):
        raise RuntimeError("Run Scan before Detect.")
    if not state.preflight_accepted:
        raise RuntimeError("Accept the preflight plan before Detect.")
    state.include_multimodal = True

    manifest = Path(state.run_dir) / "input_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))

    runtime_policy = dict(state.plan.get("runtime_policy") or {})
    face_policy_id = str(
        runtime_policy.get("face_policy_id") or "runtime_3_source_all_raw_rf_approximation"
    )
    emit(0.04, "Loading the face detector policy selected for this system")
    face_detector = _get_thesis_face_detector(face_policy_id)

    det_dir = Path(state.run_dir) / "detections"
    det_path = det_dir / "detections.jsonl"
    face_boxes_csv = det_dir / "face_boxes.csv"
    screen_boxes_csv = det_dir / "screen_boxes.csv"
    text_boxes_csv = det_dir / "text_boxes.csv"

    n_faces = n_texts = n_screens = 0
    n_img_faces = n_img_text = n_img_screen = 0
    n_errors = 0
    t0 = time.perf_counter()
    records: list[dict[str, Any]] = []

    face_items = [(row["image_id"], Path(row["local_path"])) for row in rows]
    face_label = str(runtime_policy.get("face_display_name") or face_policy_id)
    emit(0.08, f"Running face detection ({face_label})")

    def face_progress(fraction: float, message: str) -> None:
        # Face ensemble occupies ~8%–48% of overall detect progress.
        emit(0.08 + max(0.0, min(1.0, fraction)) * 0.40, message)

    try:
        face_results = face_detector.detect_paths(
            face_items,
            progress_callback=face_progress,
        )
    except Exception as exc:
        raise RuntimeError(f"Thesis face-detector batch failed: {type(exc).__name__}: {exc}") from exc
    face_detector_policy = face_detector.policy_id
    face_detector_runtime = dict(face_detector.runtime)

    # The face ensemble is no longer needed after its boxes are materialised.
    # Releasing it before OCR keeps both policies GPU-backed on modest VRAM.
    emit(0.49, "Releasing face models and loading screen/text stack")
    _THESIS_FACE_DETECTORS.pop(face_policy_id, None)
    del face_detector
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        mm_stack = _get_multimodal_stack(runtime_policy)
    except Exception as exc:
        raise RuntimeError(f"Multimodal stack unavailable: {type(exc).__name__}: {exc}") from exc

    emit(0.52, "Face detection complete. Starting precise screen and text localisation")
    total_rows = max(1, len(rows))
    for row_index, row in enumerate(rows, start=1):
        path = Path(row["local_path"])
        image_id = row["image_id"]
        faces: list[tuple[int, int, int, int, float]] = []
        texts: list[tuple[int, int, int, int, float]] = []
        screens: list[tuple[int, int, int, int, float]] = []
        detector = face_detector_policy
        face_telemetry: dict[str, Any] = {}
        mm_sources: list[str] = []
        emit(
            0.52 + (0.43 * (row_index - 1) / total_rows),
            f"Screen and text detection: image {row_index} of {len(rows)}",
        )
        try:
            rgb = Image.open(path).convert("RGB")
            faces, face_telemetry = face_results[image_id]
            mm = mm_stack.detect(rgb)
            screens = list(mm.screens)
            texts = list(mm.texts)
            mm_sources = list(mm.screen_sources)
            for w in mm.warnings:
                _DETECT_WARNINGS.append(w)
        except Exception as exc:
            n_errors += 1
            detector = f"error:{type(exc).__name__}"
            _DETECT_WARNINGS.append(f"{image_id}: {type(exc).__name__}: {exc}")

        if faces:
            n_img_faces += 1
            n_faces += len(faces)
        if texts:
            n_img_text += 1
            n_texts += len(texts)
        if screens:
            n_img_screen += 1
            n_screens += len(screens)

        records.append(
            {
                "image_id": image_id,
                "local_path": str(path),
                "detector": detector,
                "face_detector_telemetry": face_telemetry,
                "multimodal_policy": mm_stack.policy_id,
                "text_policy": mm.text_policy,
                "screen_sources": mm_sources,
                "faces": [{"x1": a, "y1": b, "x2": c, "y2": d, "score": s} for a, b, c, d, s in faces],
                "texts": [{"x1": a, "y1": b, "x2": c, "y2": d, "score": s} for a, b, c, d, s in texts],
                "screens": [{"x1": a, "y1": b, "x2": c, "y2": d, "score": s} for a, b, c, d, s in screens],
                "face_count": len(faces),
                "text_count": len(texts),
                "screen_count": len(screens),
            }
        )
        emit(
            0.52 + (0.43 * row_index / total_rows),
            f"Finished screen/text image {row_index} of {len(rows)}",
        )

    emit(0.97, "Writing detection boxes and run metadata")
    write_detection_artifacts(det_dir, records)

    elapsed = time.perf_counter() - t0
    state.n_faces = n_faces
    state.n_texts = n_texts
    state.n_screens = n_screens
    state.n_images_with_faces = n_img_faces
    state.n_images_with_text = n_img_text
    state.n_images_with_screen = n_img_screen
    warnings = list(dict.fromkeys(_DETECT_WARNINGS))[:40]
    state.detect_summary = {
        "n_images": len(rows),
        "n_faces": n_faces,
        "n_texts": n_texts,
        "n_screens": n_screens,
        "n_images_with_faces": n_img_faces,
        "n_images_with_text": n_img_text,
        "n_images_with_screen": n_img_screen,
        "n_errors": n_errors,
        "runtime_seconds": round(elapsed, 3),
        "detections_jsonl": str(det_path),
        "face_boxes_csv": str(face_boxes_csv),
        "screen_boxes_csv": str(screen_boxes_csv),
        "text_boxes_csv": str(text_boxes_csv),
        "include_multimodal": True,
        "face_detector_policy": face_detector_policy,
        "runtime_policy_id": str(state.plan.get("runtime_policy_id") or ""),
        "multimodal_policy": mm_stack.policy_id,
        "multimodal_runtime": mm_stack.runtime,
        "face_detector_runtime": face_detector_runtime,
        "warnings": warnings,
    }
    state.stages_done["detect"] = True
    state.message = (
        f"Detection complete: {n_faces} faces, {n_texts} text regions, {n_screens} screens "
        f"across {len(rows)} images ({elapsed:.1f}s)."
    )
    if warnings:
        state.message += f" Warnings: {len(warnings)} (see detect_summary)."
    save_state(state)
    write_json(Path(state.run_dir) / "metadata" / "detect_summary.json", state.detect_summary)
    emit(1.0, "Detection complete")
    return state


def mark_review_done(run_dir: str) -> WizardState:
    state = load_state(Path(run_dir))
    if not state.stages_done.get("detect"):
        raise RuntimeError("Run Detect before Review.")
    state.stages_done["review"] = True
    state.message = "Manual review marked complete (or skipped)."
    save_state(state)
    return state


@dataclass
class ApplyMethodResult:
    """Face anonymisation result, including fallback details."""

    image: Image.Image
    selected_method: str
    applied_method: str
    status: str  # ok | fallback
    error: str = ""
    reason_note: str = ""


def _apply_method(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    method: str,
    *,
    fallback_method: str = "solid_mask",
) -> ApplyMethodResult:
    """Apply the chosen face method. Never claim success if a fallback was used."""
    try:
        from privacy_pipeline_app.runtime_env import configure_app_runtime

        configure_app_runtime()
    except Exception:
        pass
    from privacy_pipeline_app.pipeline_demo import METHODS

    if method == "copy" or not boxes:
        return ApplyMethodResult(
            image=image.copy(),
            selected_method=method if boxes or method == "copy" else "copy",
            applied_method="copy",
            status="ok",
        )

    # Visual-safe deterministic path
    if method in METHODS:
        return ApplyMethodResult(
            image=METHODS[method](image, boxes),
            selected_method=method,
            applied_method=method,
            status="ok",
        )

    fallback = fallback_method if fallback_method in METHODS else "solid_mask"

    # Optional methods use the shared anonymiser registry.
    from privacy_pipeline_app.method_catalog import RESEARCH_FACE_METHOD_IDS

    if method in RESEARCH_FACE_METHOD_IDS:
        err = ""
        try:
            from src.anonymisation.registry import build_anonymiser_registry

            registry = build_anonymiser_registry()
            key_map = {
                "nullface": "nullface",
                "reverse_personalization": "reverse_personalization",
                "riddle": "riddle",
                "falco": "falco",
                "diffusion": "diffusion",
                "fams": "fams",
                "stylegan": "stylegan",
            }
            key = key_map.get(method, method)
            anon = registry.get(key)
            if anon is None:
                for name, inst in registry.items():
                    if key in name.lower() or name.lower() == key:
                        anon = inst
                        break
            if anon is None:
                err = (
                    f"Anonymiser '{method}' not found in registry "
                    "(research method may be batch-only on this install)."
                )
            else:
                preflight = getattr(anon, "reason", "") or ""
                if preflight:
                    err = str(preflight)
                else:
                    result = anon.anonymise(image, boxes)
                    out_img = result.image if hasattr(result, "image") else result
                    return ApplyMethodResult(
                        image=out_img,
                        selected_method=method,
                        applied_method=method,
                        status="ok",
                    )
        except Exception as exc:
            # Limit error text stored in CSV output.
            err = f"{type(exc).__name__}: {exc}"
            if len(err) > 500:
                err = err[:500] + "..."

        # Apply the configured fallback and record both method IDs.
        return ApplyMethodResult(
            image=METHODS[fallback](image, boxes),
            selected_method=method,
            applied_method=fallback,
            status="fallback",
            error=err or f"{method} failed with no error detail",
            reason_note=(
                f" FALLBACK: selected `{method}` but applied `{fallback}` "
                f"because the advanced method failed ({err or 'unknown error'})."
            ),
        )

    # Unknown methods use the deterministic fallback.
    return ApplyMethodResult(
        image=METHODS[fallback](image, boxes),
        selected_method=method,
        applied_method=fallback,
        status="fallback",
        error=f"Unknown method '{method}'",
        reason_note=f" FALLBACK: unknown method `{method}`; applied `{fallback}`.",
    )


def _select_method(
    strategy: str,
    fixed_method: str,
    objective: str,
    face_count: int,
    text_count: int,
    screen_count: int,
    plan: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Pick face anonymisation from the evidence plan for the chosen objective."""
    if face_count == 0:
        suffix = " Multimodal regions are handled separately." if text_count or screen_count else ""
        return "copy", "No face risk detected; no face operator applied." + suffix

    # Use the Preflight objective plan.
    if plan:
        for stage in plan.get("stages") or []:
            if stage.get("stage") == "Face anonymisation":
                mid = stage.get("method_id") or fixed_method
                title = plan.get("title") or objective
                why = stage.get("why") or ""
                rec = stage.get("recommendation") or ""
                note = f" [{rec}]" if rec else ""
                return mid, f"{title} plan · face: {stage.get('display_name', mid)}{note}. {why}"

    # Fallback by objective id
    if objective == "privacy_first":
        return "solid_mask", "Privacy-first plan: solid mask (deployment privacy terminal action)."
    if objective == "utility_priority":
        return "blur", "Utility-first plan: blur (eligible utility-oriented method)."
    return "layered", "Balanced plan: layered (default balanced ELIGIBLE action)."


def step_anonymise(
    run_dir: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> WizardState:
    """Apply the selected face/screen/text policy to every image in the run.

    ``progress_callback(fraction, message)`` is optional. When provided, the
    workflow reports 0.0–1.0 progress as each image is anonymised so the UI
    progress bar can update during long generative methods (StyleGAN, etc.).
    """

    def emit(fraction: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(1.0, fraction)), message)

    emit(0.01, "Loading detections and preparing anonymisation")
    state = load_state(Path(run_dir))
    if not state.stages_done.get("detect"):
        raise RuntimeError("Run Detect before Anonymise.")
    det_path = Path(state.run_dir) / "detections" / "detections.jsonl"
    detections = {}
    if det_path.exists():
        for line in det_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                detections[rec["image_id"]] = rec

    manifest = Path(state.run_dir) / "input_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    decisions_path = Path(state.run_dir) / "metadata" / "decisions.csv"
    method_counts: Counter[str] = Counter()  # applied methods (what pixels used)
    selected_counts: Counter[str] = Counter()
    fallback_count = 0
    runtimes: list[float] = []
    preview_count = 0
    t0 = time.perf_counter()
    total_rows = max(len(rows), 1)
    method_label = str(state.fixed_method or "selected policy")

    with decisions_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "image_id",
            "local_path",
            "selected_method",
            "applied_method",
            "reason",
            "face_count",
            "text_count",
            "screen_count",
            "runtime_seconds",
            "status",
            "error",
            "output_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, row in enumerate(rows, start=1):
            image_id = row["image_id"]
            # Reserve 0.02–0.95 for the per-image loop so callers can still
            # Report progress after this step returns.
            emit(
                0.02 + (0.93 * (row_index - 1) / total_rows),
                f"Anonymising image {row_index} of {len(rows)} ({method_label})",
            )
            path = Path(row["local_path"])
            start = time.perf_counter()
            try:
                det = detections.get(image_id, {})
                faces = det.get("faces", [])
                texts = det.get("texts", [])
                screens = det.get("screens", [])
                face_boxes = [(int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])) for b in faces]
                method, reason = _select_method(
                    state.strategy,
                    state.fixed_method,
                    state.objective,
                    len(faces),
                    len(texts),
                    len(screens),
                    plan=state.plan,
                )
                original = Image.open(path).convert("RGB")
                apply_result = _apply_method(original, face_boxes, method)
                anonymised = apply_result.image
                if apply_result.status == "fallback":
                    fallback_count += 1
                    reason += apply_result.reason_note
                # Multimodal operators from objective plan
                from src.detection.multimodal_stack import apply_screen_operator, apply_text_operator

                screen_mode = "fill"
                text_mode = "blur"
                for stage in (state.plan or {}).get("stages") or []:
                    if stage.get("stage") == "Screen redaction":
                        screen_mode = stage.get("method_id") or screen_mode
                    if stage.get("stage") == "Text redaction":
                        text_mode = stage.get("method_id") or text_mode

                screen_scored = [
                    (int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]), float(b.get("score", 0.5)))
                    for b in screens
                ]
                text_scored = [
                    (int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]), float(b.get("score", 0.5)))
                    for b in texts
                ]
                if screen_scored:
                    anonymised, screen_ops = apply_screen_operator(
                        anonymised, screen_scored, mode=screen_mode
                    )
                    reason += f" Screens ({len(screen_ops)}): plan={screen_mode}."
                if text_scored:
                    anonymised, text_ops = apply_text_operator(anonymised, text_scored, mode=text_mode)
                    reason += f" Residual text ({len(text_ops)}): plan={text_mode}."
                out_path = Path(state.run_dir) / "anonymised" / Path(image_id).with_suffix(".jpg")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                anonymised.save(out_path, quality=92)
                # The Done page is a preview, not a second copy of every result.
                if preview_count < DONE_PREVIEW_LIMIT:
                    sbs = (
                        Path(state.run_dir)
                        / "side_by_side"
                        / Path(image_id).with_suffix(".jpg")
                    )
                    sbs.parent.mkdir(parents=True, exist_ok=True)
                    w = min(900, original.width)
                    scale = w / original.width
                    left = original.resize((w, int(original.height * scale)))
                    right = anonymised.resize(left.size)
                    canvas = Image.new("RGB", (left.width * 2, left.height), (255, 255, 255))
                    canvas.paste(left, (0, 0))
                    canvas.paste(right, (left.width, 0))
                    canvas.save(sbs, quality=88)
                    preview_count += 1
                elapsed = time.perf_counter() - start
                method_counts[apply_result.applied_method] += 1
                selected_counts[apply_result.selected_method] += 1
                runtimes.append(elapsed)
                writer.writerow(
                    {
                        "image_id": image_id,
                        "local_path": str(path),
                        "selected_method": apply_result.selected_method,
                        "applied_method": apply_result.applied_method,
                        "reason": reason,
                        "face_count": len(faces),
                        "text_count": len(texts),
                        "screen_count": len(screens),
                        "runtime_seconds": round(elapsed, 4),
                        "status": apply_result.status,
                        "error": apply_result.error,
                        "output_path": str(out_path),
                    }
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start
                writer.writerow(
                    {
                        "image_id": image_id,
                        "local_path": str(path),
                        "selected_method": "error",
                        "applied_method": "error",
                        "reason": "",
                        "face_count": 0,
                        "text_count": 0,
                        "screen_count": 0,
                        "runtime_seconds": round(elapsed, 4),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "output_path": "",
                    }
                )
            emit(
                0.02 + (0.93 * row_index / total_rows),
                f"Finished image {row_index} of {len(rows)} ({method_label})",
            )

    emit(0.97, "Writing anonymisation metadata")
    total = time.perf_counter() - t0
    state.anonymise_summary = {
        "n_images": len(rows),
        "method_counts": dict(method_counts),
        "selected_method_counts": dict(selected_counts),
        "fallback_count": fallback_count,
        "runtime_total_seconds": round(total, 3),
        "runtime_mean_seconds": round(sum(runtimes) / len(runtimes), 4) if runtimes else 0.0,
        "decisions_csv": str(decisions_path),
        "anonymised_dir": str(Path(state.run_dir) / "anonymised"),
        "preview_count": preview_count,
        "preview_limit": DONE_PREVIEW_LIMIT,
    }
    state.stages_done["anonymise"] = True
    state.stages_done["review"] = state.stages_done.get("review", False) or True  # allow skip
    msg = f"Anonymisation complete: applied {dict(method_counts)} in {total:.1f}s."
    if fallback_count:
        msg += f" WARNING: {fallback_count} image(s) fell back from the selected advanced method."
    state.message = msg
    save_state(state)
    write_json(Path(state.run_dir) / "metadata" / "method_counts.json", dict(method_counts))
    write_json(
        Path(state.run_dir) / "metadata" / "selected_vs_applied.json",
        {
            "selected_method_counts": dict(selected_counts),
            "applied_method_counts": dict(method_counts),
            "fallback_count": fallback_count,
        },
    )
    write_json(
        Path(state.run_dir) / "metadata" / "runtime.json",
        {
            "runtime_total_seconds": total,
            "runtime_mean_seconds": state.anonymise_summary["runtime_mean_seconds"],
            "n_ok": sum(method_counts.values()),
            "n_fallback": fallback_count,
        },
    )
    emit(1.0, "Anonymisation complete")
    return state


def step_report(run_dir: str) -> WizardState:
    state = load_state(Path(run_dir))
    if not state.stages_done.get("anonymise"):
        raise RuntimeError("Run Anonymise before Report.")
    decisions = Path(state.run_dir) / "metadata" / "decisions.csv"
    rows = list(csv.DictReader(decisions.open(encoding="utf-8"))) if decisions.exists() else []
    applied_counts = Counter(
        (r.get("applied_method") or r.get("selected_method") or "unknown")
        for r in rows
        if r.get("status") in {"ok", "fallback"}
    )
    selected_counts = Counter(
        (r.get("selected_method") or "unknown")
        for r in rows
        if r.get("status") in {"ok", "fallback"}
    )
    fallback_rows = [r for r in rows if r.get("status") == "fallback"]
    error_rows = [r for r in rows if r.get("status") == "error"]
    lines = [
        "# Privacy pipeline success report",
        "",
        f"- Run ID: `{state.run_id}`",
        f"- Source: `{state.source_dir}`",
        f"- Strategy / App policy: `{state.strategy}` (objective_profile; not scientific OAPR materialisation)",
        f"- Fixed method (if used): `{state.fixed_method}`",
        f"- Objective profile: `{state.objective}`",
        f"- Policy semantics: see `metadata/policy_semantics.json`",
        f"- Images: `{state.n_images}`",
        f"- Faces detected: `{state.n_faces}` (images with faces: {state.n_images_with_faces})",
        f"- Text regions: `{state.n_texts}` (images: {state.n_images_with_text})",
        f"- Screen regions: `{state.n_screens}` (images: {state.n_images_with_screen})",
        f"- Anonymisation runtime (s): `{state.anonymise_summary.get('runtime_total_seconds')}`",
        f"- Mean per image (s): `{state.anonymise_summary.get('runtime_mean_seconds')}`",
        f"- Fallback images (selected method failed, safer method applied): `{len(fallback_rows)}`",
        f"- Error images: `{len(error_rows)}`",
        "",
        "## Methods selected (user / plan intent)",
        "",
    ]
    for method, n in sorted(selected_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- **{method}**: {n}")
    lines += ["", "## Methods applied (what the pixels actually used)", ""]
    for method, n in sorted(applied_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- **{method}**: {n}")
        why = METHOD_REASONS.get(method, {})
        if why:
            lines.append(f"  - Privacy: {why.get('privacy', '')}")
            lines.append(f"  - Utility: {why.get('utility', '')}")
            lines.append(f"  - Compute: {why.get('compute', '')}")
    if fallback_rows:
        lines += [
            "",
            "## Fallbacks (important)",
            "",
            "These images requested an advanced method but the app applied a safer fallback.",
            "Black boxes usually mean `solid_mask` was used after NullFace/generative failure.",
            "",
        ]
        for r in fallback_rows[:40]:
            err = (r.get("error") or "").replace("\n", " ")
            if len(err) > 180:
                err = err[:180] + "..."
            lines.append(
                f"- `{r.get('image_id')}`: selected `{r.get('selected_method')}` "
                f"→ applied `{r.get('applied_method')}`"
                + (f" ({err})" if err else "")
            )
        if len(fallback_rows) > 40:
            lines.append(f"- ... and {len(fallback_rows) - 40} more (see decisions.csv)")
    lines += [
        "",
        "## Outputs",
        "",
        f"- Anonymised images: `{state.run_dir}/anonymised/`",
        f"- Side-by-side previews: `{state.run_dir}/side_by_side/`",
        f"- Decisions: `{state.run_dir}/metadata/decisions.csv`",
        f"- Selected vs applied: `{state.run_dir}/metadata/selected_vs_applied.json`",
        f"- Detections: `{state.run_dir}/detections/`",
        "",
        "## Boundary",
        "",
        "This tool reduces face (and optional text/screen) privacy risk for research use.",
        "It does not guarantee complete anonymisation or legal compliance by itself.",
        "DCU governance, testing, and monitoring remain required before operational adoption.",
        "",
    ]
    report_md = Path(state.run_dir) / "report" / "success_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state.stages_done["report"] = True
    state.message = f"Report ready: {report_md}"
    save_state(state)
    return state


def format_stage_summary(state: WizardState) -> str:
    return "\n".join(
        [
            pipeline_markdown(state),
            "",
            f"**Run:** `{state.run_id}`",
            f"**Message:** {state.message}",
            f"**Source:** `{state.source_dir}`",
            f"**Images:** {state.n_images}",
            f"**ETA (pre-flight):** ~{state.eta_seconds/60:.1f} min",
            f"**Recommendation:** {state.recommendation}",
            "",
            "### Scan",
            f"```json\n{json.dumps(state.scan_summary, indent=2)}\n```" if state.scan_summary else "_Not run_",
            "",
            "### Detect",
            f"```json\n{json.dumps(state.detect_summary, indent=2)}\n```" if state.detect_summary else "_Not run_",
            "",
            "### Anonymise",
            f"```json\n{json.dumps(state.anonymise_summary, indent=2)}\n```" if state.anonymise_summary else "_Not run_",
        ]
    )
