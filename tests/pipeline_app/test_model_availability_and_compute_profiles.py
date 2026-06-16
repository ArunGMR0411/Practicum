"""Model availability (no heavy GPU load) + compute-profile selection (faked envs).

Does NOT load RiDDLE/FALCO/NullFace weights onto the GPU (avoids OOM).
Verifies:
  - every catalog face method is registered / has a clear availability signal
  - Recommended badges follow compute only (fake H100/A100/8GB/CPU)
  - wizard ``_select_method`` picks the user/plan method id correctly
  - App runtime env configures backend paths without manual shell exports
  - light methods actually apply through the real App path
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from privacy_pipeline_app.method_catalog import (
    FACE_ANON_OPTIONS,
    FACE_DETECTION_OPTIONS,
    MULTIMODAL_DETECTION_OPTIONS,
    RESEARCH_FACE_METHOD_IDS,
    STAGE_KEYS,
    apply_selections_to_plan,
    compute_badge,
    fits_compute,
    get_option,
    resolve_defaults_for_profile,
)
from privacy_pipeline_app.runtime_env import PROJECT_ROOT, configure_app_runtime
from privacy_pipeline_app.runtime_policy import select_runtime_policy
from privacy_pipeline_app.wizard_workflow import _apply_method, _select_method


def _env(name: str, cuda: bool, vram_gb: float) -> dict:
    return {
        "cuda_available": cuda,
        "device": "cuda" if cuda else "cpu",
        "gpu_name": name,
        "vram_total_mb": int(vram_gb * 1024) if cuda else 0,
        "cpu_count": 16,
    }


PROFILES = {
    "H100_80GB": _env("NVIDIA H100 80GB HBM3", True, 80),
    "A100_40GB": _env("NVIDIA A100-SXM4-40GB", True, 40),
    "RTX4090_24GB": _env("NVIDIA GeForce RTX 4090", True, 24),
    "RTX4060_8GB": _env("NVIDIA GeForce RTX 4060", True, 8),
    "RTX3060_6GB": _env("NVIDIA GeForce RTX 3060", True, 6),
    "CPU_ONLY": _env("not_available", False, 0),
}


def test_configure_app_runtime_without_manual_exports() -> None:
    snap = configure_app_runtime(force=True)
    assert snap["PROJECT_ROOT"] == str(PROJECT_ROOT)
    # Integrated defaults via run_web / configure_app_runtime (no shell export step)
    if (PROJECT_ROOT / "third_party" / "riddle").is_dir():
        assert Path(snap["RIDDLE_SOURCE_ROOT"]).name == "riddle"
    if (PROJECT_ROOT / "data" / "models" / "riddle").is_dir():
        assert Path(snap["RIDDLE_ASSET_ROOT"]).name == "riddle"
    if (PROJECT_ROOT / "third_party" / "falco").is_dir():
        assert Path(snap["FALCO_SOURCE_ROOT"]).name == "falco"


def test_core_detector_assets_exist_on_disk() -> None:
    yolo = PROJECT_ROOT / "data/models/face_detection_candidates/yolo11s_widerface.pt"
    screen = PROJECT_ROOT / "app/models/multimodal_screen_yolo11s.pt"
    assert yolo.is_file(), f"missing {yolo}"
    assert screen.is_file(), f"missing {screen}"
    rfdetr = list(
        (PROJECT_ROOT / "data/models/face_detection_candidates/rfdetr_hf_cache").rglob("*.pth")
    )
    assert rfdetr, "RF-DETR checkpoint missing under rfdetr_hf_cache"


def test_every_face_anonymiser_is_registered_with_availability_signal() -> None:
    """Availability check only - does not call anonymise() (no GPU weight load)."""
    configure_app_runtime(force=True)
    from src.anonymisation.registry import build_anonymiser_registry

    reg = build_anonymiser_registry()
    required = {
        "blur",
        "pixelate",
        "nullface",
        "stylegan",
        "reverse_personalization",
        "diffusion",
        "fams",
        "riddle",
        "falco",
    }
    assert required.issubset(set(reg.keys()))
    report = {}
    for mid in sorted(required):
        anon = reg[mid]
        reason = getattr(anon, "reason", "") or ""
        # Visual-safe built-ins should have no preflight block
        if mid in {"blur", "pixelate"}:
            assert not reason, f"{mid} should be available without reason: {reason}"
        report[mid] = "available" if not reason else f"unavailable: {reason[:100]}"
    # Print for operator visibility when run with -s
    print("\n=== MODEL AVAILABILITY (registry preflight, no GPU load) ===")
    for mid, status in report.items():
        print(f"  {mid}: {status}")
    # Research methods must at least be constructible
    for mid in RESEARCH_FACE_METHOD_IDS:
        assert mid in reg


def test_riddle_falco_source_and_assets_present() -> None:
    configure_app_runtime(force=True)
    from src.anonymisation.riddle_anonymiser import RiddleAnonymiser
    from src.anonymisation.falco_anonymiser import FalcoAnonymiser

    r = RiddleAnonymiser()
    f = FalcoAnonymiser()
    # On this project tree we installed trees; preflight reason should be empty if CUDA ok
    assert r.source_root.is_dir()
    assert (r.asset_root / "e4e_ffhq_encode_256.pt").is_file() or (
        PROJECT_ROOT / "data/models/riddle_gdrive/e4e_ffhq_encode_256.pt"
    ).is_file()
    assert f.source_root.is_dir()
    assert (f.source_root / "models/pretrained/e4e/e4e_ffhq_encode.pt").is_file()
    # reason may still mention CUDA if no GPU - that is fine
    print(f"\nriddle.reason={r.reason!r}\nfalco.reason={f.reason!r}")


@pytest.mark.parametrize("profile_name", list(PROFILES.keys()))
def test_compute_profile_badges_and_auto_defaults(profile_name: str) -> None:
    env = PROFILES[profile_name]
    policy = select_runtime_policy(env)
    defaults = resolve_defaults_for_profile("privacy", env)

    # Policy contracts
    if not env["cuda_available"]:
        assert policy.policy_id == "portable_cpu"
        assert defaults["face_detection"] == "yolo11s_face"
        assert "640" in defaults["multimodal_detection"]
    elif env["vram_total_mb"] >= 12 * 1024:
        assert policy.policy_id == "accelerated_full"
        assert defaults["face_detection"] == "runtime_3_source_all_raw_rf_approximation"
    elif env["vram_total_mb"] >= 6 * 1024:
        assert policy.policy_id == "accelerated_efficient"
    else:
        assert policy.policy_id in {"accelerated_compact", "accelerated_efficient", "accelerated_full"}

    # Badge = compute only for every catalog option
    for stage_opts in (FACE_DETECTION_OPTIONS, MULTIMODAL_DETECTION_OPTIONS, FACE_ANON_OPTIONS):
        for opt in stage_opts:
            badge = compute_badge(opt, env)
            expected = "Recommended" if fits_compute(opt, env) else "Not recommended"
            assert badge == expected, f"{profile_name} {opt.method_id}: {badge} != {expected}"

    # Auto face ops stay visual-safe
    assert defaults["face_anonymisation"] not in RESEARCH_FACE_METHOD_IDS


def test_powerful_profiles_recommend_riddle_not_global_not_recommended() -> None:
    for name in ("H100_80GB", "A100_40GB", "RTX4090_24GB"):
        env = PROFILES[name]
        assert compute_badge(get_option("face_anonymisation", "riddle"), env) == "Recommended"
        assert compute_badge(get_option("face_anonymisation", "falco"), env) == "Recommended"
        assert compute_badge(get_option("face_detection", "fusion_rfdetr_yolo11s_scrfd10g"), env) == "Recommended"


def test_weak_profiles_mark_riddle_not_recommended() -> None:
    for name in ("RTX4060_8GB", "RTX3060_6GB", "CPU_ONLY"):
        env = PROFILES[name]
        assert compute_badge(get_option("face_anonymisation", "riddle"), env) == "Not recommended"


def test_wizard_invokes_selected_method_id_for_each_research_option() -> None:
    """Plan → _select_method returns the chosen research method (real wizard code)."""
    env = PROFILES["A100_40GB"]
    for mid in sorted(RESEARCH_FACE_METHOD_IDS):
        defaults = resolve_defaults_for_profile("balanced", env)
        selections = {**defaults, "face_anonymisation": mid}
        plan = apply_selections_to_plan(
            {"title": "Balanced", "stages": [], "runtime_policy": {}},
            selections,
            env,
        )
        selected, reason = _select_method(
            "objective_profile",
            "layered",
            "utility_under_privacy_floor",
            face_count=1,
            text_count=0,
            screen_count=0,
            plan=plan,
        )
        assert selected == mid, f"expected {mid}, got {selected} ({reason})"


def test_light_methods_apply_through_real_app_path() -> None:
    """Real apply for visual-safe methods only (safe on constrained GPU)."""
    image = Image.new("RGB", (64, 64), (12, 24, 36))
    boxes = [(8, 8, 40, 40)]
    for mid in ("solid_mask", "layered", "blur", "pixelate"):
        res = _apply_method(image, boxes, mid, fallback_method="solid_mask")
        assert res.status == "ok"
        assert res.selected_method == mid
        assert res.applied_method == mid


def test_research_apply_routes_to_named_anonymiser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove App invokes the correct registry entry for each research method.

    Uses lightweight stubs so we never load multi‑GB weights (no OOM).
    """
    from dataclasses import dataclass

    @dataclass
    class _R:
        image: Image.Image

    class _Spy:
        def __init__(self, name: str) -> None:
            self.method_name = name
            self.reason = ""
            self.calls = 0

        def anonymise(self, image, boxes):  # noqa: ANN001
            self.calls += 1
            return _R(image=image.copy())

    spies = {mid: _Spy(mid) for mid in RESEARCH_FACE_METHOD_IDS}
    monkeypatch.setattr(
        "src.anonymisation.registry.build_anonymiser_registry",
        lambda: dict(spies),
    )
    image = Image.new("RGB", (32, 32), (1, 2, 3))
    boxes = [(2, 2, 20, 20)]
    for mid in sorted(RESEARCH_FACE_METHOD_IDS):
        res = _apply_method(image, boxes, mid, fallback_method="solid_mask")
        assert res.status == "ok"
        assert res.selected_method == res.applied_method == mid
        assert spies[mid].calls == 1


def test_ten_compute_profiles_flow_table(capsys: pytest.CaptureFixture[str]) -> None:
    """Ten documented cases: fake hardware → real policy/badge/select (no live heavy GPU)."""
    cases = [
        (1, "H100 Privacy auto", PROFILES["H100_80GB"], "privacy", None),
        (2, "H100 user RiDDLE", PROFILES["H100_80GB"], "balanced", "riddle"),
        (3, "A100 user FALCO", PROFILES["A100_40GB"], "utility", "falco"),
        (4, "A100 user NullFace", PROFILES["A100_40GB"], "privacy", "nullface"),
        (5, "A100 user Diffusion", PROFILES["A100_40GB"], "balanced", "diffusion"),
        (6, "24GB Privacy auto", PROFILES["RTX4090_24GB"], "privacy", None),
        (7, "8GB force RiDDLE", PROFILES["RTX4060_8GB"], "balanced", "riddle"),
        (8, "CPU Privacy auto", PROFILES["CPU_ONLY"], "privacy", None),
        (9, "CPU force FALCO", PROFILES["CPU_ONLY"], "utility", "falco"),
        (10, "H100 StyleGAN", PROFILES["H100_80GB"], "balanced", "stylegan"),
    ]
    print("\n=== TEN COMPUTE-PROFILE CASES (real catalog/wizard; no heavy GPU load) ===")
    for num, label, env, focus, user_face in cases:
        defaults = resolve_defaults_for_profile(focus, env)
        face = user_face or defaults["face_anonymisation"]
        badge = compute_badge(get_option("face_anonymisation", face), env)
        policy = select_runtime_policy(env)
        plan = apply_selections_to_plan(
            {"title": focus, "stages": [], "runtime_policy": policy.to_dict()},
            {**defaults, "face_anonymisation": face},
            env,
        )
        selected, _ = _select_method(
            "objective_profile",
            defaults["face_anonymisation"],
            "utility_under_privacy_floor",
            1,
            0,
            0,
            plan=plan,
        )
        assert selected == face
        # Light path always applies
        if face not in RESEARCH_FACE_METHOD_IDS:
            res = _apply_method(Image.new("RGB", (24, 24)), [(2, 2, 12, 12)], face)
            assert res.applied_method == face and res.status == "ok"
            apply_note = f"applied={res.applied_method} status={res.status}"
        else:
            apply_note = "research_selected_via_wizard (apply covered by spy + optional live tests)"
        print(
            f"Case {num}: {label}\n"
            f"  tier={policy.policy_id} det={defaults['face_detection']} mm={defaults['multimodal_detection']}\n"
            f"  face={face} badge={badge} selected={selected}\n"
            f"  {apply_note}\n"
        )
