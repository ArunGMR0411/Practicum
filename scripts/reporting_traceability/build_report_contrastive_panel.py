#!/usr/bin/env python3

"""Build a report-facing contrastive panel from defended review panels."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "outputs" / "runs" / "compact_benchmark" / "review_panels"
OUTPUT_IMAGE = PROJECT_ROOT / "outputs" / "runs" / "figures" / "report_contrastive_utility_panel.png"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "runs" / "figures" / "report_contrastive_utility_panel.json"

SELECTED = [
    "day1__members__allie__09_0380.webp.png",
    "day1__members__bjorn__08_0558.webp.png",
    "day4__members__florian__14_0144.webp.png",
]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def slug_to_label(name: str) -> str:
    stem = name.replace(".webp.png", "")
    parts = stem.split("__")
    if len(parts) >= 4:
        return f"{parts[0]} / {parts[2]} / {parts[3].replace('.webp', '')}"
    return stem


def main() -> None:
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    title_font = load_font(28)
    label_font = load_font(18)

    images = []
    labels = []
    for filename in SELECTED:
        path = SOURCE_DIR / filename
        images.append(Image.open(path).convert("RGB"))
        labels.append(slug_to_label(filename))

    target_width = max(image.width for image in images)
    resized = []
    for image in images:
        if image.width == target_width:
            resized.append(image)
            continue
        scale = target_width / image.width
        target_height = int(round(image.height * scale))
        resized.append(image.resize((target_width, target_height), Image.Resampling.LANCZOS))

    margin = 24
    gutter = 18
    label_height = 28
    title_height = 44
    total_height = (
        margin
        + title_height
        + gutter
        + sum(image.height + label_height + gutter for image in resized)
        + margin
    )

    canvas = Image.new("RGB", (target_width + margin * 2, total_height), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, margin),
        "Contrastive Utility Panel: Retained Methods vs Bounded Negative Evidence",
        fill=(28, 28, 28),
        font=title_font,
    )

    y = margin + title_height + gutter
    for image, label in zip(resized, labels):
        canvas.paste(image, (margin, y))
        y += image.height + 6
        draw.text((margin, y), label, fill=(50, 50, 50), font=label_font)
        y += label_height + gutter

    canvas.save(OUTPUT_IMAGE)

    payload = {
        "version": "report_contrastive_utility_panel",
        "source_dir": str(SOURCE_DIR.relative_to(PROJECT_ROOT)),
        "selected_panels": [str((SOURCE_DIR / name).relative_to(PROJECT_ROOT)) for name in SELECTED],
        "output_image": str(OUTPUT_IMAGE.relative_to(PROJECT_ROOT)),
        "purpose": (
            "Report-facing contrastive panel showing retained reviewed anonymiser outputs "
            "and bounded negative evidence on the same egocentric examples."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
