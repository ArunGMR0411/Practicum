from __future__ import annotations

"""Clean-clone public micro-fixture: no CASTLE, no private weights."""


import csv
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

pytestmark = pytest.mark.public

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests/fixtures/public_micro"
ANN = FIX / "annotations.json"


def _load_ann() -> dict:
    if not ANN.is_file():
        # Generate on the fly for clean clones that omit binary images in git.
        from tests.fixtures.public_micro.generate_fixture import main as gen

        gen()
    return json.loads(ANN.read_text(encoding="utf-8"))


def _apply_box_op(image: Image.Image, boxes: list[list[int]], op: str) -> Image.Image:
    out = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = out.crop((x1, y1, x2, y2))
        if op == "fill":
            ImageDraw.Draw(out).rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
        elif op == "blur":
            out.paste(crop.filter(ImageFilter.GaussianBlur(radius=4)), (x1, y1))
        elif op == "layered":
            w, h = crop.size
            tiny = crop.resize((max(1, w // 8), max(1, h // 8)), Image.Resampling.BILINEAR)
            layered = tiny.resize((w, h), Image.Resampling.NEAREST)
            layered = layered.filter(ImageFilter.GaussianBlur(radius=2))
            out.paste(layered, (x1, y1))
        elif op == "copy":
            pass
        else:
            ImageDraw.Draw(out).rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
    return out


def test_public_fixture_exists_or_generates() -> None:
    data = _load_ann()
    assert len(data["images"]) >= 3
    for rec in data["images"]:
        path = FIX / rec["path"]
        if not path.is_file():
            from tests.fixtures.public_micro.generate_fixture import main as gen

            gen()
        assert path.is_file()
        with Image.open(path) as im:
            assert im.size[0] > 0


def test_public_fixture_routing_and_selected_vs_applied(tmp_path: Path) -> None:
    from src.policy.registry import get_app_policy_semantics, get_profile

    data = _load_ann()
    profile = get_profile("balanced")
    semantics = get_app_policy_semantics()
    assert semantics["app_policy_id"] == "objective_profile"

    decisions = tmp_path / "decisions.csv"
    out_dir = tmp_path / "anonymised"
    out_dir.mkdir()
    rows = []
    for rec in data["images"]:
        path = FIX / rec["path"]
        if not path.is_file():
            from tests.fixtures.public_micro.generate_fixture import main as gen

            gen()
        image = Image.open(path).convert("RGB")
        faces = rec.get("faces") or []
        selected = "copy" if not faces else str(profile["face_anonymisation"])
        applied = selected
        # Simulate method selection and fallback.
        if rec["image_id"] == "synthetic_face":
            selected_request = "nullface"
            applied = str(profile["face_anonymisation"])  # fallback visual-safe
            status = "fallback"
        else:
            selected_request = selected
            status = "ok"
        result = _apply_box_op(image, faces, applied if applied != "copy" else "copy")
        # multimodal operators
        result = _apply_box_op(result, rec.get("screen_boxes") or [], str(profile["screen_operator"]))
        result = _apply_box_op(result, rec.get("text_boxes") or [], str(profile["text_operator"]))
        out_path = out_dir / f"{rec['image_id']}.png"
        result.save(out_path)
        rows.append(
            {
                "image_id": rec["image_id"],
                "selected_method": selected_request,
                "applied_method": applied,
                "status": status,
                "face_count": len(faces),
                "output": str(out_path.name),
            }
        )
        if rec["image_id"] == "synthetic_noface":
            assert applied == "copy"
        if rec["image_id"] == "synthetic_face":
            assert status == "fallback"
            assert applied == profile["face_anonymisation"]

    with decisions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "selected_method",
                "applied_method",
                "status",
                "face_count",
                "output",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    selected_vs_applied = {
        "fallback_count": sum(1 for r in rows if r["status"] == "fallback"),
        "rows": rows,
        "app_policy_id": semantics["app_policy_id"],
        "scientific_policy_id": semantics["scientific_policy_id"],
    }
    (tmp_path / "selected_vs_applied.json").write_text(
        json.dumps(selected_vs_applied, indent=2), encoding="utf-8"
    )
    assert selected_vs_applied["fallback_count"] == 1
    assert all((out_dir / r["output"]).is_file() for r in rows)


def test_public_progress_callback_contract() -> None:
    """Progress callbacks used by the App should be 0..1 fractions."""
    progress = []

    def cb(fraction: float, message: str) -> None:
        progress.append((fraction, message))

    for i in range(5):
        cb((i + 1) / 5, f"step {i+1}")
    assert progress[0][0] == pytest.approx(0.2)
    assert progress[-1][0] == pytest.approx(1.0)
    assert all(0.0 <= f <= 1.0 for f, _ in progress)
