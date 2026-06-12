#!/usr/bin/env python3
"""Emit machine-readable App readiness status for audit archival."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "app" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "app" / "src"))

OUT = ROOT / "outputs/07_app_validation/04_app_readiness_status.json"


def main() -> None:
    from privacy_pipeline_app.runtime_env import configure_app_runtime
    from privacy_pipeline_app.runtime_policy import select_runtime_policy
    from privacy_pipeline_app.thesis_face_detector import FACE_POLICY_FALLBACK_ORDER
    from src.policy.registry import get_app_policy_semantics, get_profile_defaults

    configure_app_runtime(force=True)
    try:
        from privacy_pipeline_app.wizard_workflow import probe_environment as probe
    except Exception:
        def probe():
            import torch
            return {
                "cuda_available": bool(torch.cuda.is_available()),
                "vram_total_mb": int(torch.cuda.get_device_properties(0).total_memory / 1e6)
                if torch.cuda.is_available()
                else 0,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            }

    env = probe() if callable(probe) else probe
    # wizard may export probe_environment differently
    try:
        from privacy_pipeline_app import wizard_workflow as ww
        if hasattr(ww, "probe_environment"):
            env = ww.probe_environment()
    except Exception:
        pass

    # fallback simple env
    if not isinstance(env, dict):
        import torch
        env = {
            "cuda_available": torch.cuda.is_available(),
            "vram_total_mb": int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
            if torch.cuda.is_available()
            else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }

    runtime_policy = select_runtime_policy(env)
    semantics = get_app_policy_semantics()
    defaults = get_profile_defaults("balanced")

    missing_assets = []
    asset_checks = {
        "screen_yolo": ROOT / "app/models/multimodal_screen_yolo11s.pt",
        "yolo11_face": ROOT / "data/models/face_detection_candidates/yolo11s_widerface.pt",
        "rfdetr": ROOT / "data/models/face_detection_candidates/rfdetr_medium_face.pth",
        "hardened_rf_deploy": ROOT
        / "outputs/02_face_detection/12_detector_error_hardening/deploy_error_hardened_all_raw_rf_iou0_45.joblib",
    }
    for name, path in asset_checks.items():
        if not path.is_file():
            missing_assets.append({"asset": name, "path": str(path.relative_to(ROOT)), "present": False})
        else:
            missing_assets.append({"asset": name, "path": str(path.relative_to(ROOT)), "present": True})

    # Static readiness only: do not instantiate or execute detector models.
    from src.detection.error_hardened_rf_policy import RUNTIME_FACE_DETECTOR_POLICY_ID

    preferred_face = RUNTIME_FACE_DETECTOR_POLICY_ID
    runtime_spec = __import__("src.policy.registry", fromlist=["load_policy_registry"]).load_policy_registry()["app_runtime_detector"]
    face_runtime = {
        "policy_id": preferred_face,
        "candidate_sources": runtime_spec["candidate_sources"],
        "model_path": runtime_spec["filter_model_path"],
        "model_sha256": runtime_spec["filter_model_sha256"],
        "fallback_applied": False,
        "scientific_score_assigned": False,
        "preflight_mode": "static_no_model_execution",
    }
    fallback_methods = list(FACE_POLICY_FALLBACK_ORDER)
    status = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "app_policy_id": semantics["app_policy_id"],
        "scientific_policy_id": semantics["scientific_policy_id"],
        "simplification": semantics["simplification"],
        "selected_compute_tier": runtime_policy.policy_id,
        "tier_default_face_detector_policy": runtime_policy.face_policy_id,
        "selected_face_detector_policy": (face_runtime or {}).get("policy_id")
        or runtime_policy.face_policy_id,
        "preferred_face_detector_policy": preferred_face,
        "face_detector_runtime": face_runtime,
        "balanced_profile_defaults": defaults,
        "fallback_face_detector_order": fallback_methods,
        "assets": missing_assets,
        "missing_required_for_full_accelerated": [
            a["asset"] for a in missing_assets if not a["present"] and a["asset"] in {"screen_yolo", "yolo11_face"}
        ],
        "ready_for_visual_safe_demo": all(
            a["present"] for a in missing_assets if a["asset"] in {"screen_yolo", "yolo11_face"}
        ),
        "ready_for_public_smoke_without_weights": True,
        "notes": [
            "Visual-safe App path uses objective_profile defaults (not scientific 286/81/133 OAPR).",
            "Public smoke protocol does not require CASTLE or detector weights.",
            "ready_for_visual_safe_demo requires screen YOLO + YOLO face weights; smoke path is always available.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
