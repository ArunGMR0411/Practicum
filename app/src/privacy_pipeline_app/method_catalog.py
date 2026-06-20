#!/usr/bin/env python3
"""Preflight method catalog: compute-fit badges and qualitative trade-offs."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True)
class MethodOption:
    method_id: str
    display_name: str
    stage_key: str
    # What the method is good at (qualitative product language)
    privacy_note: str
    utility_note: str
    speed_note: str
    visual_note: str
    # Compute requirements only
    needs_cuda: bool = False
    min_vram_mb: int = 0
    compute_class: str = "light"  # light | medium | heavy
    typical_vram_note: str = "Works on most systems"
    eta_seconds_per_image: float = 0.1
    runnable: bool = True


STAGE_KEYS = (
    "face_detection",
    "multimodal_detection",
    "face_anonymisation",
    "screen_operator",
    "text_operator",
)

STAGE_LABELS = {
    "face_detection": "Face detection",
    "multimodal_detection": "Screen and text detection",
    "face_anonymisation": "Face anonymisation",
    "screen_operator": "Screen redaction",
    "text_operator": "Text redaction",
}

FACE_DETECTION_OPTIONS: list[MethodOption] = [
    MethodOption(
        "runtime_3_source_all_raw_rf_approximation",
        "Three-source hardened RF runtime approximation",
        "face_detection",
        privacy_note="Evidence-backed bounded three-source runtime tier",
        utility_note="Scientific seven-source score is not assigned",
        speed_note="Slower (multi-detector + RF filter)",
        visual_note="No direct visual change (detection only)",
        needs_cuda=True,
        min_vram_mb=10 * 1024,
        compute_class="heavy",
        typical_vram_note="Typically needs about 10 GB+ GPU memory",
        eta_seconds_per_image=1.5,
    ),
    MethodOption(
        "fusion_rfdetr_yolo11s_scrfd10g",
        "RF-DETR + YOLO11Face + SCRFD (supporting)",
        "face_detection",
        privacy_note="High-recall multi-detector fusion (supporting, not primary)",
        utility_note="May flag more boxes than the hardened RF primary",
        speed_note="Slower",
        visual_note="No direct visual change (detection only)",
        needs_cuda=True,
        min_vram_mb=10 * 1024,
        compute_class="heavy",
        typical_vram_note="Typically needs about 10 GB+ GPU memory",
        eta_seconds_per_image=1.4,
    ),
    MethodOption(
        "fusion_rfdetr_scrfd10g",
        "RF-DETR + SCRFD",
        "face_detection",
        privacy_note="Strong face finding",
        utility_note="Balanced box load",
        speed_note="Moderate",
        visual_note="No direct visual change (detection only)",
        needs_cuda=True,
        min_vram_mb=6 * 1024,
        compute_class="medium",
        typical_vram_note="Typically needs about 6 GB+ GPU memory",
        eta_seconds_per_image=1.2,
    ),
    MethodOption(
        "fixed_fusion_yolo11s1280_scrfd10g",
        "YOLO11Face + SCRFD",
        "face_detection",
        privacy_note="Good face finding",
        utility_note="Fewer heavy models",
        speed_note="Faster than full triple fusion",
        visual_note="No direct visual change (detection only)",
        needs_cuda=True,
        min_vram_mb=4 * 1024,
        compute_class="medium",
        typical_vram_note="Typically needs about 4 GB+ GPU memory",
        eta_seconds_per_image=1.0,
    ),
    MethodOption(
        "yolo11s_face",
        "YOLO11Face only",
        "face_detection",
        privacy_note="May miss harder faces",
        utility_note="Fewer false boxes on light systems",
        speed_note="Fastest portable path",
        visual_note="No direct visual change (detection only)",
        needs_cuda=False,
        min_vram_mb=0,
        compute_class="light",
        typical_vram_note="Runs on CPU or small GPUs",
        eta_seconds_per_image=0.6,
    ),
]

MULTIMODAL_DETECTION_OPTIONS: list[MethodOption] = [
    MethodOption(
        "reviewed_screen_yolo11s_1280",
        "Screen YOLO 1280 + text OCR",
        "multimodal_detection",
        privacy_note="Better at finding screens and text",
        utility_note="May mark more regions",
        speed_note="Slower",
        visual_note="No direct visual change (detection only)",
        needs_cuda=True,
        min_vram_mb=6 * 1024,
        compute_class="medium",
        typical_vram_note="Typically needs about 6 GB+ GPU memory",
        eta_seconds_per_image=2.0,
    ),
    MethodOption(
        "reviewed_screen_yolo11s_960",
        "Screen YOLO 960 + text OCR",
        "multimodal_detection",
        privacy_note="Solid screen and text finding",
        utility_note="Moderate region load",
        speed_note="Moderate",
        visual_note="No direct visual change (detection only)",
        needs_cuda=True,
        min_vram_mb=3 * 1024,
        compute_class="medium",
        typical_vram_note="Typically needs about 3 GB+ GPU memory",
        eta_seconds_per_image=1.6,
    ),
    MethodOption(
        "reviewed_screen_yolo11s_640",
        "Screen YOLO 640 + text OCR",
        "multimodal_detection",
        privacy_note="May miss small screens or fine text",
        utility_note="Fewer heavy regions",
        speed_note="Best on limited hardware",
        visual_note="No direct visual change (detection only)",
        needs_cuda=False,
        min_vram_mb=0,
        compute_class="light",
        typical_vram_note="Runs on CPU or small GPUs",
        eta_seconds_per_image=2.5,
    ),
]

FACE_ANON_OPTIONS: list[MethodOption] = [
    MethodOption(
        "solid_mask",
        "Solid mask",
        "face_anonymisation",
        privacy_note="Strongest identity wipe among simple methods",
        utility_note="Removes face appearance completely",
        speed_note="Very fast",
        visual_note="Hard black blocks on faces",
        compute_class="light",
        typical_vram_note="Negligible GPU need",
        eta_seconds_per_image=0.08,
    ),
    MethodOption(
        "layered",
        "Layered blur",
        "face_anonymisation",
        privacy_note="Strong privacy with some structure kept",
        utility_note="Keeps more scene context than solid mask",
        speed_note="Fast",
        visual_note="Softer face regions than solid mask",
        compute_class="light",
        typical_vram_note="Negligible GPU need",
        eta_seconds_per_image=0.12,
    ),
    MethodOption(
        "blur",
        "Gaussian blur",
        "face_anonymisation",
        privacy_note="Weaker than mask or layered",
        utility_note="Preserves more natural look",
        speed_note="Fast",
        visual_note="Smoother faces, more scene utility",
        compute_class="light",
        typical_vram_note="Negligible GPU need",
        eta_seconds_per_image=0.10,
    ),
    MethodOption(
        "pixelate",
        "Pixelate",
        "face_anonymisation",
        privacy_note="Weaker privacy than blur or mask",
        utility_note="Highest simple utility among light methods",
        speed_note="Fast",
        visual_note="Blocky faces, scene mostly intact",
        compute_class="light",
        typical_vram_note="Negligible GPU need",
        eta_seconds_per_image=0.10,
    ),
    # Research / generative methods: selectable with compute badges only.
    # They are NOT profile defaults (visual-quality gate failed for default policy).
    MethodOption(
        "nullface",
        "NullFace",
        "face_anonymisation",
        privacy_note="Generative identity change (research comparator)",
        utility_note="Aims for natural-looking faces",
        speed_note="Much slower",
        visual_note="Research-only for defaults; visual quality gated on egocentric hard cases",
        needs_cuda=True,
        min_vram_mb=8 * 1024,
        compute_class="heavy",
        typical_vram_note="Typically needs about 8 GB+ GPU memory",
        eta_seconds_per_image=8.0,
    ),
    MethodOption(
        "diffusion",
        "Diffusion (low-step)",
        "face_anonymisation",
        privacy_note="Diffusion-based face rewrite (research comparator)",
        utility_note="Can preserve scene context when it succeeds",
        speed_note="Slow",
        visual_note="Research-only for defaults; visual quality gated",
        needs_cuda=True,
        min_vram_mb=10 * 1024,
        compute_class="heavy",
        typical_vram_note="Typically needs about 10 GB+ GPU memory",
        eta_seconds_per_image=20.0,
    ),
    MethodOption(
        "riddle",
        "RiDDLE",
        "face_anonymisation",
        privacy_note="Strong numeric privacy on comparable tables (research comparator)",
        utility_note="High metric scores when it completes",
        speed_note="Slow",
        visual_note="Research-only for defaults; fails egocentric visual/pose gate",
        needs_cuda=True,
        min_vram_mb=12 * 1024,
        compute_class="heavy",
        typical_vram_note="Typically needs about 12 GB+ GPU memory",
        eta_seconds_per_image=25.0,
    ),
    MethodOption(
        "falco",
        "FALCO",
        "face_anonymisation",
        privacy_note="Reference-based generative anonymisation (research comparator)",
        utility_note="Aims for natural faces with high compute cost",
        speed_note="Very slow",
        visual_note="Research-only for defaults; visual quality gated",
        needs_cuda=True,
        min_vram_mb=12 * 1024,
        compute_class="heavy",
        typical_vram_note="Typically needs about 12 GB+ GPU memory",
        eta_seconds_per_image=40.0,
    ),
    MethodOption(
        "fams",
        "FAMS",
        "face_anonymisation",
        privacy_note="Attribute-related generative branch (research comparator)",
        utility_note="Quality-limited after systematic tuning",
        speed_note="Slow",
        visual_note="Research-only for defaults; quality-limited",
        needs_cuda=True,
        min_vram_mb=10 * 1024,
        compute_class="heavy",
        typical_vram_note="Typically needs about 10 GB+ GPU memory",
        eta_seconds_per_image=30.0,
    ),
    MethodOption(
        "reverse_personalization",
        "Reverse Personalization",
        "face_anonymisation",
        privacy_note="Strong generative anonymisation when it works (research comparator)",
        utility_note="Can look very natural",
        speed_note="Very slow",
        visual_note="Research-only for defaults; high failure/runtime risk",
        needs_cuda=True,
        min_vram_mb=12 * 1024,
        compute_class="heavy",
        typical_vram_note="Typically needs about 12 GB+ GPU memory",
        eta_seconds_per_image=60.0,
    ),
    MethodOption(
        "stylegan",
        "StyleGAN / StyleID branch",
        "face_anonymisation",
        privacy_note="Style-based generative face rewrite (research comparator)",
        utility_note="Quality-limited after systematic tuning",
        speed_note="Slow",
        visual_note="Research-only for defaults; quality-limited",
        needs_cuda=True,
        min_vram_mb=10 * 1024,
        compute_class="heavy",
        typical_vram_note="Typically needs about 10 GB+ GPU memory",
        eta_seconds_per_image=35.0,
    ),
]

def _load_canonical_eligibility() -> dict[str, Any]:
    """Load single eligibility artefact (routing + App must consult this)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    path = root / "outputs/03_anonymisation/20_canonical_method_eligibility/02_canonical_method_eligibility.json"
    if path.is_file():
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    return {}


_ELIG = _load_canonical_eligibility()

# Optional face methods. Use the eligibility file when available.
RESEARCH_FACE_METHOD_IDS: frozenset[str] = frozenset(
    _ELIG.get("research_only_app_ids")
    or {
        "nullface",
        "diffusion",
        "riddle",
        "falco",
        "fams",
        "reverse_personalization",
        "stylegan",
    }
)

VISUAL_SAFE_FACE_METHOD_IDS: frozenset[str] = frozenset(
    _ELIG.get("app_default_eligible_app_ids")
    or {"solid_mask", "layered", "blur", "pixelate", "copy"}
)

CANONICAL_ELIGIBILITY_PATH = (
    "outputs/03_anonymisation/20_canonical_method_eligibility/01_canonical_method_eligibility.csv"
)


def is_default_eligible_app_method(method_id: str) -> bool:
    """True if method may be used as an App / scientific default (not research-only)."""
    mid = (method_id or "").strip().lower()
    if mid in VISUAL_SAFE_FACE_METHOD_IDS:
        return True
    return mid not in RESEARCH_FACE_METHOD_IDS and mid in {
        "solid_mask",
        "layered",
        "blur",
        "pixelate",
        "copy",
    }

SCREEN_OPERATOR_OPTIONS: list[MethodOption] = [
    MethodOption(
        "fill",
        "Solid fill",
        "screen_operator",
        privacy_note="Strong wipe of display content",
        utility_note="Loses screen context",
        speed_note="Very fast",
        visual_note="Hard black screens",
        compute_class="light",
        typical_vram_note="Negligible GPU need",
        eta_seconds_per_image=0.02,
    ),
    MethodOption(
        "blur",
        "Strong blur",
        "screen_operator",
        privacy_note="Strong content hide",
        utility_note="Keeps coarse layout",
        speed_note="Very fast",
        visual_note="Softer than solid fill",
        compute_class="light",
        typical_vram_note="Negligible GPU need",
        eta_seconds_per_image=0.03,
    ),
    MethodOption(
        "pixelate",
        "Pixelate",
        "screen_operator",
        privacy_note="Coarse content hide",
        utility_note="More scene structure kept",
        speed_note="Very fast",
        visual_note="Blocky screens",
        compute_class="light",
        typical_vram_note="Negligible GPU need",
        eta_seconds_per_image=0.03,
    ),
]

TEXT_OPERATOR_OPTIONS: list[MethodOption] = [
    MethodOption(
        "fill",
        "Solid fill",
        "text_operator",
        privacy_note="Hard residual text wipe",
        utility_note="Can punch holes in the scene",
        speed_note="Very fast",
        visual_note="Black text patches",
        compute_class="light",
        typical_vram_note="Negligible GPU need",
        eta_seconds_per_image=0.02,
    ),
    MethodOption(
        "blur",
        "Gaussian blur",
        "text_operator",
        privacy_note="Good text hide",
        utility_note="Softer scene impact than fill",
        speed_note="Very fast",
        visual_note="Blurred text regions",
        compute_class="light",
        typical_vram_note="Negligible GPU need",
        eta_seconds_per_image=0.02,
    ),
    MethodOption(
        "pixelate",
        "Pixelate",
        "text_operator",
        privacy_note="Lighter text hide",
        utility_note="Most scene structure kept",
        speed_note="Very fast",
        visual_note="Blocky text patches",
        compute_class="light",
        typical_vram_note="Negligible GPU need",
        eta_seconds_per_image=0.02,
    ),
]

STAGE_OPTIONS: dict[str, list[MethodOption]] = {
    "face_detection": FACE_DETECTION_OPTIONS,
    "multimodal_detection": MULTIMODAL_DETECTION_OPTIONS,
    "face_anonymisation": FACE_ANON_OPTIONS,
    "screen_operator": SCREEN_OPERATOR_OPTIONS,
    "text_operator": TEXT_OPERATOR_OPTIONS,
}

def _load_profile_defaults() -> dict[str, dict[str, str]]:
    """Authoritative profile defaults from configs/policy_registry.json."""
    from src.policy.registry import get_profile_defaults

    return {
        "privacy": get_profile_defaults("privacy"),
        "balanced": get_profile_defaults("balanced"),
        "utility": get_profile_defaults("utility"),
    }


# Profile starting picks (outcome preference only; compute may step them down).
# Built from the single policy registry — do not hard-code duplicates here.
PROFILE_DEFAULTS: dict[str, dict[str, str]] = _load_profile_defaults()


def _vram_mb(env: dict[str, Any]) -> int:
    try:
        return int(env.get("vram_total_mb") or 0)
    except (TypeError, ValueError):
        return 0


def system_power_lines(env: dict[str, Any]) -> list[str]:
    return [
        f"Device: {env.get('device', 'unknown')}",
        f"GPU: {env.get('gpu_name', 'not reported')}",
        f"VRAM: {env.get('vram_total_mb', 'not reported')} MB",
        f"CPUs: {env.get('cpu_count', 'not reported')}",
        f"CUDA: {env.get('cuda_available', False)}",
    ]


def fits_compute(option: MethodOption, env: dict[str, Any]) -> bool:
    """Recommended vs Not recommended is compute-only."""
    cuda = bool(env.get("cuda_available"))
    vram = _vram_mb(env)
    if option.needs_cuda and not cuda:
        return False
    if option.min_vram_mb <= 0:
        return True
    if vram <= 0:
        # Unknown VRAM: allow light/medium if CUDA present, reject heavy
        if option.compute_class == "heavy":
            return False
        return True if (cuda or not option.needs_cuda) else False
    return vram >= option.min_vram_mb


def compute_badge(option: MethodOption, env: dict[str, Any]) -> str:
    return "Recommended" if fits_compute(option, env) else "Not recommended"


def get_option(stage_key: str, method_id: str) -> MethodOption | None:
    for opt in STAGE_OPTIONS.get(stage_key, []):
        if opt.method_id == method_id:
            return opt
    return None


def normalize_focus(focus: str) -> str:
    raw = (focus or "balanced").strip().lower()
    if raw.startswith("priv"):
        return "privacy"
    if raw.startswith("util"):
        return "utility"
    return "balanced"


def resolve_defaults_for_profile(focus: str, env: dict[str, Any]) -> dict[str, str]:
    """Profile defaults, stepped down only when compute cannot run them."""
    focus_key = normalize_focus(focus)
    defaults = dict(PROFILE_DEFAULTS[focus_key])
    for stage in STAGE_KEYS:
        preferred = defaults[stage]
        opt = get_option(stage, preferred)
        if opt is not None and fits_compute(opt, env):
            continue
        # Step down to the strongest option that still fits compute
        # (catalog order is roughly heavy -> light for detectors).
        chosen = preferred
        for candidate in STAGE_OPTIONS[stage]:
            if fits_compute(candidate, env):
                chosen = candidate.method_id
                break
        defaults[stage] = chosen
    return defaults


def estimate_eta_from_selections(
    n_images: int,
    selections: dict[str, str],
    env: dict[str, Any],
) -> float:
    scale = 0.5 if env.get("cuda_available") else 1.0
    total = 0.0
    for key, mid in selections.items():
        opt = get_option(key, mid)
        if opt is None:
            continue
        if "detection" in key:
            total += opt.eta_seconds_per_image * scale
        else:
            total += opt.eta_seconds_per_image
    return max(0, n_images) * total


def any_not_recommended(selections: dict[str, str], env: dict[str, Any]) -> list[str]:
    risky: list[str] = []
    for key, mid in selections.items():
        opt = get_option(key, mid)
        if opt is None:
            continue
        if not fits_compute(opt, env):
            risky.append(f"{STAGE_LABELS[key]}: {opt.display_name}")
    return risky


def qualitative_vs_default(selected: MethodOption, default: MethodOption | None) -> str:
    if default is None or selected.method_id == default.method_id:
        return "This is your current plan pick for this stage."
    return (
        f"Compared with {default.display_name}: "
        f"{selected.privacy_note.lower()}; "
        f"{selected.utility_note.lower()}; "
        f"{selected.speed_note.lower()}; "
        f"{selected.visual_note.lower()}."
    )


def method_detail_html(
    option: MethodOption,
    env: dict[str, Any],
    *,
    n_images: int,
    profile_default_id: str,
    selections: dict[str, str],
) -> str:
    badge = compute_badge(option, env)
    badge_cls = "badge-ok" if badge == "Recommended" else "badge-warn"
    power = system_power_lines(env)
    default_opt = get_option(option.stage_key, profile_default_id)
    # ETA if this method is used with rest of current selections
    trial = dict(selections)
    trial[option.stage_key] = option.method_id
    eta = estimate_eta_from_selections(n_images, trial, env)
    eta_min = eta / 60.0
    trade = qualitative_vs_default(option, default_opt)

    if badge == "Recommended":
        body = (
            f"<p>Your system is capable of running <strong>{escape(option.display_name)}</strong>.</p>"
            f"<p>{escape(trade)}</p>"
            f"<p>Estimated folder time with this choice: about <strong>{eta_min:.1f} min</strong>.</p>"
        )
    else:
        body = (
            f"<p>You chose <strong>{escape(option.display_name)}</strong>. "
            f"This is <strong>Not recommended</strong> for your system.</p>"
            f"<p>Why: {escape(option.typical_vram_note)}. "
            f"Your machine reports VRAM "
            f"<strong>{escape(str(env.get('vram_total_mb', 'unknown')))} MB</strong>, "
            f"GPU <strong>{escape(str(env.get('gpu_name', 'not reported')))}</strong>, "
            f"CUDA <strong>{escape(str(env.get('cuda_available', False)))}</strong>.</p>"
            f"<p>We can still try to run it. Risk: out of memory, very long runs, or fallback behaviour. "
            f"Estimated folder time: about <strong>{eta_min:.1f} min</strong> if it completes.</p>"
            f"<p>You can switch back to a Recommended method, or keep this choice and confirm the risk on Proceed.</p>"
            f"<p>{escape(trade)}</p>"
        )

    power_html = "".join(f"<li>{escape(line)}</li>" for line in power)
    return (
        f"<div class='method-detail'>"
        f"<div class='method-detail-head'>"
        f"<strong>{escape(option.display_name)}</strong> "
        f"<span class='{badge_cls}'>{escape(badge)}</span>"
        f"</div>"
        f"<div class='method-detail-body'>{body}</div>"
        f"<div class='method-detail-power'><div class='kicker'>Your system</div>"
        f"<ul>{power_html}</ul></div>"
        f"</div>"
    )


def apply_selections_to_plan(
    plan: dict[str, Any],
    selections: dict[str, str],
    env: dict[str, Any],
) -> dict[str, Any]:
    from privacy_pipeline_app.runtime_policy import select_runtime_policy

    out = dict(plan)
    ordered: list[dict[str, Any]] = []
    for key in STAGE_KEYS:
        mid = selections.get(key, "")
        opt = get_option(key, mid)
        if opt is None:
            continue
        badge = compute_badge(opt, env)
        ordered.append(
            {
                "stage": STAGE_LABELS[key],
                "method_id": opt.method_id,
                "display_name": opt.display_name,
                "why": opt.privacy_note,
                "evidence": opt.typical_vram_note,
                "eta_seconds_per_image": opt.eta_seconds_per_image,
                "recommendation": badge,
                "user_selected": True,
            }
        )
    out["stages"] = ordered
    out["user_method_selections"] = selections
    out["stretch_methods"] = any_not_recommended(selections, env)

    policy = select_runtime_policy(env)
    rp = dict(out.get("runtime_policy") or policy.to_dict())
    if "face_detection" in selections:
        face_opt = get_option("face_detection", selections["face_detection"])
        if face_opt:
            rp["face_policy_id"] = face_opt.method_id
            rp["face_display_name"] = face_opt.display_name
            rp["face_evidence"] = face_opt.typical_vram_note
    if "multimodal_detection" in selections:
        mm_opt = get_option("multimodal_detection", selections["multimodal_detection"])
        if mm_opt:
            rp["multimodal_policy_id"] = mm_opt.method_id
            rp["multimodal_display_name"] = mm_opt.display_name
            if "1280" in mm_opt.method_id:
                rp["multimodal_image_size"] = 1280
            elif "960" in mm_opt.method_id:
                rp["multimodal_image_size"] = 960
            else:
                rp["multimodal_image_size"] = 640
    out["runtime_policy"] = rp
    out["runtime_policy_id"] = rp.get("policy_id", policy.policy_id)
    return out


def stage_card_labels(selections: dict[str, str], env: dict[str, Any]) -> dict[str, str]:
    """Button labels for each stage card (used by Gradio buttons)."""
    labels: dict[str, str] = {}
    for key in STAGE_KEYS:
        mid = selections.get(key, "")
        opt = get_option(key, mid)
        if opt is None:
            labels[key] = f"{STAGE_LABELS[key]}\nChange"
            continue
        badge = compute_badge(opt, env)
        labels[key] = f"{STAGE_LABELS[key]}\n{opt.display_name}\n{badge}\nChange"
    return labels


def method_choice_names(stage_key: str) -> list[str]:
    """Clean names only (no recommended badges) for the picker modal."""
    return [opt.display_name for opt in STAGE_OPTIONS.get(stage_key, [])]


def method_choice_options(
    stage_key: str, env: dict[str, Any]
) -> list[tuple[str, str]]:
    """Picker labels with compute suitability while preserving clean values."""
    return [
        (
            f"{opt.display_name} | {compute_badge(opt, env)} | "
            f"{opt.typical_vram_note}",
            opt.display_name,
        )
        for opt in STAGE_OPTIONS.get(stage_key, [])
    ]


def method_id_from_display_name(stage_key: str, display_name: str) -> str | None:
    for opt in STAGE_OPTIONS.get(stage_key, []):
        if opt.display_name == display_name:
            return opt.method_id
    return None


def build_preflight_dashboard(
    *,
    focus_title: str,
    focus_summary: str,
    n_images: int,
    source_dir: str,
    env: dict[str, Any],
    selections: dict[str, str],
    focus: str,
    active_stage: str | None = None,
    detail_method_id: str | None = None,
) -> str:
    """Render the single, clickable preflight summary."""
    eta = estimate_eta_from_selections(n_images, selections, env)
    eta_min = eta / 60.0
    stretch = any_not_recommended(selections, env)
    stretch_note = ""
    if stretch:
        stretch_note = (
            "<div class='stretch-banner'>One or more methods are Not recommended for this system. "
            "On Proceed you must confirm the risk before detection starts.</div>"
        )

    cards = []
    for key in STAGE_KEYS:
        mid = selections.get(key, "")
        opt = get_option(key, mid)
        if opt is None:
            continue
        badge = compute_badge(opt, env)
        badge_cls = "badge-ok" if badge == "Recommended" else "badge-warn"
        trigger_id = f"stage-trigger-{key.replace('_', '-')}"
        cards.append(
            f"<button type='button' class='dash-snap dash-stage-card' "
            f"onclick=\"document.getElementById('{trigger_id}').click()\" "
            f"aria-label='Configure {escape(STAGE_LABELS[key])}'>"
            f"<span class='dash-stage'>{escape(STAGE_LABELS[key])}</span>"
            f"<span class='dash-method'>{escape(opt.display_name)}</span>"
            f"<span class='{badge_cls}'>{escape(badge)}</span>"
            f"<span class='dash-card-open' aria-hidden='true'>+</span>"
            f"</button>"
        )

    return f"""
    <div class="preflight-dash">
      <div class="preflight-head">
        <h2>{escape(focus_title)} plan</h2>
        <p class="sub">{escape(focus_summary)}</p>
      </div>
      <div class="stats">
        <span class="chip">📁 {n_images} images</span>
        <span class="chip">⏱ ~{eta_min:.1f} min</span>
        <span class="chip" title="{escape(source_dir)}">📂 input</span>
      </div>
      {stretch_note}
      <p class="muted">Select a stage to compare methods. Each option is checked against the available compute before it is applied.</p>
      <div class="dash-grid dash-snap-grid">{''.join(cards)}</div>
    </div>
    """
