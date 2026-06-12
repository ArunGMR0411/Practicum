#!/usr/bin/env python3
"""Public smoke protocol via real App anonymisation + objective_profile routing.

No CASTLE and no private detector weights required. Uses synthetic
smoke_protocol_public frames with manifest-provided boxes (stand-in for live
detectors) and calls App ``pipeline_demo`` / ``wizard_workflow`` operators.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "app" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "app" / "src"))

SMOKE = ROOT / "outputs/01_protocol/smoke_protocol_public"
MANIFEST = SMOKE / "smoke_manifest_24.csv"
OUT = SMOKE / "runs" / "latest"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing {MANIFEST}; generate public micro fixture first.")

    from privacy_pipeline_app.pipeline_demo import METHODS, apply_solid
    from privacy_pipeline_app.wizard_workflow import _apply_method
    from src.policy.registry import get_app_policy_semantics, get_profile

    profile = get_profile("balanced")
    semantics = get_app_policy_semantics()
    face_op = str(profile["face_anonymisation"])
    screen_op = str(profile["screen_operator"])
    text_op = str(profile["text_operator"])

    # Map registry operator names onto App METHODS keys.
    screen_method = {
        "fill": "solid_mask",
        "solid_mask": "solid_mask",
        "blur": "blur",
        "pixelate": "pixelate",
        "layered": "layered",
    }.get(screen_op, "solid_mask")
    text_method = {
        "blur": "blur",
        "fill": "solid_mask",
        "solid_mask": "solid_mask",
        "pixelate": "pixelate",
        "layered": "layered",
    }.get(text_op, "blur")

    (OUT / "anonymised").mkdir(parents=True, exist_ok=True)
    (OUT / "side_by_side").mkdir(parents=True, exist_ok=True)
    (OUT / "metadata").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    decisions = []
    t0 = time.perf_counter()
    for row in rows:
        path = SMOKE / row["relative_path"]
        image = Image.open(path).convert("RGB")
        faces = [tuple(b) for b in json.loads(row["face_boxes_json"] or "[]")]
        screens = [tuple(b) for b in json.loads(row["screen_boxes_json"] or "[]")]
        texts = [tuple(b) for b in json.loads(row["text_boxes_json"] or "[]")]

        # objective_profile balanced routing (App path): copy if no faces else profile face op
        selected = "copy" if not faces else face_op
        result = _apply_method(image, faces, selected, fallback_method="solid_mask")
        out = result.image
        applied = result.applied_method

        # Multimodal operators via the same App METHODS table as pipeline_demo
        if screens:
            if screen_method == "solid_mask":
                out = apply_solid(out, screens)
            else:
                out = METHODS.get(screen_method, apply_solid)(out, screens)
        if texts:
            out = METHODS.get(text_method, METHODS["blur"])(out, texts)

        anon_path = OUT / "anonymised" / f"{row['image_id']}.jpg"
        out.save(anon_path, quality=90)
        w, h = image.size
        sbs = Image.new("RGB", (w * 2 + 8, h), (20, 20, 20))
        sbs.paste(image, (0, 0))
        sbs.paste(out, (w + 8, 0))
        sbs_path = OUT / "side_by_side" / f"{row['image_id']}.jpg"
        sbs.save(sbs_path, quality=90)
        decisions.append(
            {
                "protocol_id": row["protocol_id"],
                "image_id": row["image_id"],
                "n_faces_detected": len(faces),
                "selected_method": selected,
                "applied_method": applied,
                "screen_operator": screen_op,
                "text_operator": text_op,
                "app_module_face": "wizard_workflow._apply_method",
                "app_module_multimodal": "pipeline_demo.METHODS",
                "output_sha256": sha256_file(anon_path),
                "status": result.status,
            }
        )

    elapsed = time.perf_counter() - t0
    with (OUT / "metadata" / "decisions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(decisions[0].keys()))
        w.writeheader()
        w.writerows(decisions)
    summary = {
        "protocol": "public_smoke_24",
        "n_images": len(decisions),
        "app_policy_id": semantics["app_policy_id"],
        "scientific_policy_id": semantics["scientific_policy_id"],
        "profile": profile,
        "runtime_seconds": round(elapsed, 3),
        "method_counts": {
            m: sum(1 for d in decisions if d["applied_method"] == m)
            for m in sorted({d["applied_method"] for d in decisions})
        },
        "castle_required": False,
        "detector_weights_required": False,
        "uses_app_anonymisation_modules": True,
        "note": (
            "Manifest-provided boxes stand in for live detectors so a fresh clone can run "
            "without CASTLE/weights. Face ops call wizard_workflow._apply_method; "
            "screen/text ops call pipeline_demo.METHODS."
        ),
    }
    (OUT / "metadata" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "metadata" / "policy_semantics.json").write_text(
        json.dumps(semantics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
