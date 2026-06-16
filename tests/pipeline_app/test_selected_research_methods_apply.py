"""When a research method is selected and available, App applies that method (not silent fallback)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from PIL import Image

from privacy_pipeline_app.method_catalog import RESEARCH_FACE_METHOD_IDS
from privacy_pipeline_app.wizard_workflow import _apply_method


@dataclass
class _OkResult:
    image: Image.Image


class _WorkingAnonymiser:
    """Stand-in for a fully installed research backend that succeeds."""

    def __init__(self, name: str) -> None:
        self.method_name = name
        self.reason = ""
        self.calls = 0

    def anonymise(self, image, boxes):  # noqa: ANN001
        self.calls += 1
        out = image.copy()
        # Mark pixels so tests can prove the method ran
        out.putpixel((0, 0), (min(255, 10 + self.calls), 20, 30))
        return _OkResult(image=out)


class _BrokenAnonymiser:
    def __init__(self, name: str, reason: str = "") -> None:
        self.method_name = name
        self.reason = reason

    def anonymise(self, image, boxes):  # noqa: ANN001
        raise RuntimeError(f"{self.method_name} simulated failure")


@pytest.fixture()
def working_registry(monkeypatch: pytest.MonkeyPatch):
    workers = {mid: _WorkingAnonymiser(mid) for mid in RESEARCH_FACE_METHOD_IDS}

    def _build():
        return dict(workers)

    monkeypatch.setattr(
        "src.anonymisation.registry.build_anonymiser_registry",
        _build,
    )
    return workers


def test_each_research_method_runs_when_selected_on_powerful_path(working_registry) -> None:
    image = Image.new("RGB", (64, 64), (1, 2, 3))
    boxes = [(8, 8, 40, 40)]
    for method_id in sorted(RESEARCH_FACE_METHOD_IDS):
        result = _apply_method(image, boxes, method_id, fallback_method="solid_mask")
        assert result.status == "ok", f"{method_id} should run: {result.error}"
        assert result.selected_method == method_id
        assert result.applied_method == method_id
        assert working_registry[method_id].calls == 1
        assert result.error == ""


def test_riddle_and_falco_selected_equal_applied_when_available(working_registry) -> None:
    image = Image.new("RGB", (48, 48), (0, 0, 0))
    for mid in ("riddle", "falco"):
        result = _apply_method(image, [(2, 2, 20, 20)], mid)
        assert result.status == "ok"
        assert result.selected_method == result.applied_method == mid


def test_unavailable_research_method_falls_back_honestly(monkeypatch: pytest.MonkeyPatch) -> None:
    """If assets missing (reason set), App must not claim generative success."""
    monkeypatch.setattr(
        "src.anonymisation.registry.build_anonymiser_registry",
        lambda: {"riddle": _BrokenAnonymiser("riddle", reason="riddle assets missing")},
    )
    image = Image.new("RGB", (40, 40), (5, 5, 5))
    result = _apply_method(image, [(1, 1, 15, 15)], "riddle", fallback_method="solid_mask")
    assert result.status == "fallback"
    assert result.selected_method == "riddle"
    assert result.applied_method == "solid_mask"
    assert "assets missing" in result.error or "FALLBACK" in result.reason_note


def test_ten_selection_flow_cases_documented(working_registry, capsys: pytest.CaptureFixture[str]) -> None:
    """Ten App flow cases: hardware × method → recommendation + apply outcome."""
    from privacy_pipeline_app.method_catalog import compute_badge, get_option, resolve_defaults_for_profile
    from privacy_pipeline_app.runtime_policy import select_runtime_policy

    cases = [
        {
            "case": 1,
            "label": "H100 + Privacy auto defaults",
            "env": {"cuda_available": True, "vram_total_mb": 80 * 1024, "gpu_name": "H100"},
            "focus": "privacy",
            "user_face": None,
        },
        {
            "case": 2,
            "label": "H100 + user selects RiDDLE",
            "env": {"cuda_available": True, "vram_total_mb": 80 * 1024, "gpu_name": "H100"},
            "focus": "balanced",
            "user_face": "riddle",
        },
        {
            "case": 3,
            "label": "A100 + user selects FALCO",
            "env": {"cuda_available": True, "vram_total_mb": 40 * 1024, "gpu_name": "A100"},
            "focus": "utility",
            "user_face": "falco",
        },
        {
            "case": 4,
            "label": "A100 + user selects NullFace",
            "env": {"cuda_available": True, "vram_total_mb": 40 * 1024, "gpu_name": "A100"},
            "focus": "privacy",
            "user_face": "nullface",
        },
        {
            "case": 5,
            "label": "A100 + user selects Diffusion",
            "env": {"cuda_available": True, "vram_total_mb": 40 * 1024, "gpu_name": "A100"},
            "focus": "balanced",
            "user_face": "diffusion",
        },
        {
            "case": 6,
            "label": "16GB GPU + full fusion recommended",
            "env": {"cuda_available": True, "vram_total_mb": 16 * 1024, "gpu_name": "RTX 4090"},
            "focus": "privacy",
            "user_face": None,
        },
        {
            "case": 7,
            "label": "8GB GPU + RiDDLE not recommended but user force",
            "env": {"cuda_available": True, "vram_total_mb": 8 * 1024, "gpu_name": "RTX 4060"},
            "focus": "balanced",
            "user_face": "riddle",
        },
        {
            "case": 8,
            "label": "CPU only + auto defaults",
            "env": {"cuda_available": False, "vram_total_mb": 0, "gpu_name": "none"},
            "focus": "privacy",
            "user_face": None,
        },
        {
            "case": 9,
            "label": "CPU only + user tries FALCO",
            "env": {"cuda_available": False, "vram_total_mb": 0, "gpu_name": "none"},
            "focus": "utility",
            "user_face": "falco",
        },
        {
            "case": 10,
            "label": "H100 + user selects StyleGAN branch",
            "env": {"cuda_available": True, "vram_total_mb": 80 * 1024, "gpu_name": "H100"},
            "focus": "balanced",
            "user_face": "stylegan",
        },
    ]

    image = Image.new("RGB", (32, 32), (7, 7, 7))
    boxes = [(2, 2, 18, 18)]
    print("\n=== TEN APP FLOW CASES (selection + apply) ===")
    for c in cases:
        env = c["env"]
        defaults = resolve_defaults_for_profile(c["focus"], env)
        face_id = c["user_face"] or defaults["face_anonymisation"]
        opt = get_option("face_anonymisation", face_id)
        badge = compute_badge(opt, env) if opt else "n/a"
        runtime = select_runtime_policy(env)
        # Apply face method
        apply = _apply_method(image, boxes, face_id, fallback_method="solid_mask")
        line = (
            f"Case {c['case']}: {c['label']}\n"
            f"  focus={c['focus']} runtime_tier={runtime.policy_id}\n"
            f"  auto_face_det={defaults['face_detection']}\n"
            f"  auto_mm={defaults['multimodal_detection']}\n"
            f"  face_method={face_id} badge={badge}\n"
            f"  selected={apply.selected_method} applied={apply.applied_method} "
            f"status={apply.status}\n"
        )
        print(line)
        # Contracts
        if face_id in RESEARCH_FACE_METHOD_IDS and env.get("cuda_available") and (
            int(env.get("vram_total_mb") or 0) >= (opt.min_vram_mb if opt else 0)
        ):
            # Apply the selected method from the registry fixture.
            assert apply.status == "ok"
            assert apply.applied_method == face_id
        if not env.get("cuda_available") and c["user_face"] is None:
            assert defaults["face_detection"] == "yolo11s_face"
            assert apply.applied_method in {"solid_mask", "layered", "blur", "pixelate", "copy"}
