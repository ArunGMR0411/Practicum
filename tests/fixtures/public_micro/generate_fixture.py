#!/usr/bin/env python3
"""Generate synthetic public micro-fixture images + annotation JSON."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
IMG = OUT / "images"
ANN = OUT / "annotations.json"


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    records = []

    # 1) synthetic face-like oval
    face = Image.new("RGB", (320, 240), (40, 80, 40))
    d = ImageDraw.Draw(face)
    d.ellipse((110, 50, 210, 180), fill=(220, 180, 150), outline=(20, 20, 20), width=2)
    d.ellipse((130, 90, 150, 110), fill=(30, 30, 30))
    d.ellipse((170, 90, 190, 110), fill=(30, 30, 30))
    d.arc((140, 120, 180, 155), 20, 160, fill=(80, 40, 40), width=2)
    face_path = IMG / "synthetic_face.png"
    face.save(face_path)
    records.append(
        {
            "image_id": "synthetic_face",
            "path": "images/synthetic_face.png",
            "faces": [[110, 50, 210, 180]],
            "text_boxes": [],
            "screen_boxes": [],
            "expected_face_action_if_balanced": "layered",
        }
    )

    # 2) no-face scene
    empty = Image.new("RGB", (320, 240), (90, 120, 160))
    d = ImageDraw.Draw(empty)
    d.rectangle((0, 180, 320, 240), fill=(60, 90, 50))
    empty_path = IMG / "synthetic_noface.png"
    empty.save(empty_path)
    records.append(
        {
            "image_id": "synthetic_noface",
            "path": "images/synthetic_noface.png",
            "faces": [],
            "text_boxes": [],
            "screen_boxes": [],
            "expected_face_action_if_balanced": "copy",
        }
    )

    # 3) screen + text-like blocks
    screen = Image.new("RGB", (320, 240), (30, 30, 30))
    d = ImageDraw.Draw(screen)
    d.rectangle((40, 40, 280, 180), fill=(20, 40, 80), outline=(200, 200, 200), width=3)
    d.rectangle((60, 60, 260, 90), fill=(230, 230, 230))
    d.rectangle((60, 100, 200, 120), fill=(200, 200, 200))
    screen_path = IMG / "synthetic_screen_text.png"
    screen.save(screen_path)
    records.append(
        {
            "image_id": "synthetic_screen_text",
            "path": "images/synthetic_screen_text.png",
            "faces": [],
            "text_boxes": [[60, 60, 260, 90], [60, 100, 200, 120]],
            "screen_boxes": [[40, 40, 280, 180]],
            "expected_face_action_if_balanced": "copy",
            "expected_screen_operator": "fill",
            "expected_text_operator": "blur",
        }
    )

    ANN.write_text(json.dumps({"schema_version": "1.0", "images": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} images under {IMG} and {ANN}")


if __name__ == "__main__":
    main()
