"""Hardware-aware deployment policy and measured runtime calibration.

Runtime tier method IDs come from ``configs/policy_registry.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any

from src.policy.registry import get_runtime_tier_spec, select_runtime_tier_id


@dataclass(frozen=True)
class RuntimePolicy:
    policy_id: str
    tier: str
    face_policy_id: str
    face_display_name: str
    face_evidence: str
    multimodal_policy_id: str
    multimodal_display_name: str
    multimodal_image_size: int
    text_canvas_size: int
    text_confidence: float
    text_use_gpu: bool
    estimated_seconds_per_image: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeEstimate:
    total_seconds: float
    seconds_per_image: float
    source: str
    sample_count: int


def select_runtime_policy(env: dict[str, Any]) -> RuntimePolicy:
    """Choose the strongest validated detector package that fits the system."""
    cuda = bool(env.get("cuda_available"))
    try:
        vram_mb = int(env.get("vram_total_mb") or 0)
    except (TypeError, ValueError):
        vram_mb = 0

    tier_id = select_runtime_tier_id(cuda_available=cuda, vram_total_mb=vram_mb)
    spec = get_runtime_tier_spec(tier_id)
    tier_label = {
        "accelerated_full": "accelerated",
        "accelerated_efficient": "accelerated efficient",
        "accelerated_compact": "accelerated compact",
        "portable_cpu": "portable",
    }.get(tier_id, tier_id)
    mm_name = {
        "reviewed_screen_yolo11s_1280": "Reviewed screen detector + recognised text OCR",
        "reviewed_screen_yolo11s_960": "Compact reviewed screen detector + recognised text OCR",
        "reviewed_screen_yolo11s_640": "Portable reviewed screen detector + recognised text OCR",
    }.get(str(spec["multimodal_policy_id"]), str(spec["multimodal_policy_id"]))
    return RuntimePolicy(
        policy_id=tier_id,
        tier=tier_label,
        face_policy_id=str(spec["face_policy_id"]),
        face_display_name=str(spec["face_display_name"]),
        face_evidence=str(spec["face_evidence"]),
        multimodal_policy_id=str(spec["multimodal_policy_id"]),
        multimodal_display_name=mm_name,
        multimodal_image_size=int(spec["multimodal_image_size"]),
        text_canvas_size=int(spec["text_canvas_size"]),
        text_confidence=float(spec["text_confidence"]),
        text_use_gpu=bool(spec["text_use_gpu"]),
        estimated_seconds_per_image=float(spec["estimated_seconds_per_image"]),
    )


def _normalise_gpu_name(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def estimate_runtime(
    n_images: int,
    policy: RuntimePolicy,
    env: dict[str, Any],
    runs_dir: Path,
) -> RuntimeEstimate:
    """Prefer timings from completed runs using the same hardware and policy."""
    exact: list[float] = []
    same_system: list[float] = []
    gpu_name = _normalise_gpu_name(env.get("gpu_name"))
    for state_path in sorted(runs_dir.glob("*/state.json")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            summary = state.get("detect_summary") or {}
            count = int(summary.get("n_images") or 0)
            elapsed = float(summary.get("runtime_seconds") or 0.0)
            if count <= 0 or elapsed <= 0 or int(summary.get("n_errors") or 0) > 0:
                continue
            system_path = state_path.parent / "metadata" / "system_profile.json"
            system = json.loads(system_path.read_text(encoding="utf-8")) if system_path.exists() else {}
            if _normalise_gpu_name(system.get("gpu_name")) != gpu_name:
                continue
            seconds_per_image = elapsed / count
            if count < 10:
                continue
            same_system.append(seconds_per_image)
            runtime_id = str((state.get("plan") or {}).get("runtime_policy_id") or "")
            if runtime_id == policy.policy_id:
                exact.append(seconds_per_image)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue

    if exact:
        per_image = float(median(exact[-5:]))
        source = "measured on this system with the selected policy"
        sample_count = len(exact)
    elif same_system:
        per_image = float(same_system[-1])
        source = "latest completed measurement from this system"
        sample_count = len(same_system)
    else:
        per_image = policy.estimated_seconds_per_image
        source = "hardware-tier estimate; recalibrates after the first run"
        sample_count = 0
    return RuntimeEstimate(
        total_seconds=max(0, n_images) * per_image,
        seconds_per_image=per_image,
        source=source,
        sample_count=sample_count,
    )
