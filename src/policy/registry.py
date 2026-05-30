"""Load the single machine-readable policy registry.

Authoritative file: ``configs/policy_registry.json``.
App modules, tests, and integrated runners should import from here rather than
duplicating privacy/balanced/utility defaults.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "policy_registry.json"

APP_POLICY_ID = "objective_profile"
SCIENTIFIC_VISUAL_SAFE_POLICY_ID = "oapr_visual_safe_balanced_500"


@lru_cache(maxsize=1)
def load_policy_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Policy registry missing: {REGISTRY_PATH}")
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def get_profile(focus: str) -> dict[str, Any]:
    reg = load_policy_registry()
    key = _normalise_focus(focus)
    profiles = reg["app_policy"]["profiles"]
    if key not in profiles:
        raise KeyError(f"Unknown objective profile {focus!r}; known={sorted(profiles)}")
    return dict(profiles[key])


def get_profile_defaults(focus: str) -> dict[str, str]:
    profile = get_profile(focus)
    return {
        "face_detection": str(profile["face_detection"]),
        "multimodal_detection": str(profile["multimodal_detection"]),
        "face_anonymisation": str(profile["face_anonymisation"]),
        "screen_operator": str(profile["screen_operator"]),
        "text_operator": str(profile["text_operator"]),
    }


def get_app_policy_semantics() -> dict[str, Any]:
    reg = load_policy_registry()
    app = reg["app_policy"]
    scientific = reg["scientific_policies"]["oapr_visual_safe_balanced_500"]
    return {
        "scientific_policy_id": app.get("scientific_policy_id", SCIENTIFIC_VISUAL_SAFE_POLICY_ID),
        "app_policy_id": app.get("app_policy_id", APP_POLICY_ID),
        "legacy_aliases": list(app.get("legacy_aliases") or []),
        "display_name": app.get("display_name"),
        "simplification": list(app.get("simplification") or []),
        "scientific_route_counts": dict(scientific.get("route_counts") or {}),
        "scientific_display_name": scientific.get("display_name"),
        "registry_path": str(REGISTRY_PATH.relative_to(PROJECT_ROOT)),
        "schema_version": reg.get("schema_version"),
    }


def get_runtime_tier_spec(tier_id: str) -> dict[str, Any]:
    reg = load_policy_registry()
    tiers = reg["runtime_tiers"]
    if tier_id not in tiers:
        raise KeyError(f"Unknown runtime tier {tier_id!r}")
    return dict(tiers[tier_id])


def select_runtime_tier_id(*, cuda_available: bool, vram_total_mb: int) -> str:
    """Choose a runtime tier from CUDA availability and VRAM."""
    if cuda_available and vram_total_mb >= 12 * 1024:
        return "accelerated_full"
    if cuda_available and vram_total_mb >= 6 * 1024:
        return "accelerated_efficient"
    if cuda_available:
        return "accelerated_compact"
    return "portable_cpu"


def _normalise_focus(focus: str) -> str:
    f = (focus or "balanced").strip().lower()
    if f in {"privacy", "privacy-focused", "privacy_first"}:
        return "privacy"
    if f in {"utility", "utility-focused", "utility_priority"}:
        return "utility"
    if f in {"balanced", "utility_under_privacy_floor"}:
        return "balanced"
    if f in {"privacy", "balanced", "utility"}:
        return f
    return "balanced"


def is_app_policy_alias(name: str) -> bool:
    reg = load_policy_registry()
    aliases = {str(a).lower() for a in reg["app_policy"].get("legacy_aliases") or []}
    aliases.add(str(reg["app_policy"].get("app_policy_id", APP_POLICY_ID)).lower())
    return str(name or "").strip().lower() in aliases
