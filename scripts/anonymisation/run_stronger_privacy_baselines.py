#!/usr/bin/env python3
"""Generate stronger classical privacy baselines from existing detections.

This script is provided for a future controlled run. Task 05 creates it but
does not execute it, so no image anonymisation is performed in Task 05.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_boxes(path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["image_id"]].append(
                (int(float(row["x1"])), int(float(row["y1"])), int(float(row["x2"])), int(float(row["y2"])))
            )
    return grouped


def load_manifest_paths(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["relative_path"] for row in reader if row.get("relative_path")]


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    left = max(0, min(x1, width))
    top = max(0, min(y1, height))
    right = max(0, min(x2, width))
    bottom = max(0, min(y2, height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def apply_solid_mask_black(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        valid = clamp_box(box, *out.size)
        if valid:
            draw.rectangle(valid, fill=(0, 0, 0))
    return out


def apply_solid_mask_mean_color(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        valid = clamp_box(box, *out.size)
        if not valid:
            continue
        region = out.crop(valid)
        pixels = list(region.resize((1, 1), Image.Resampling.BILINEAR).getdata())
        draw.rectangle(valid, fill=pixels[0])
    return out


def apply_layered_blur_downscale_noise(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    out = image.copy()
    for box in boxes:
        valid = clamp_box(box, *out.size)
        if not valid:
            continue
        left, top, right, bottom = valid
        region = out.crop(valid).convert("RGB")
        width, height = region.size
        tiny = region.resize((max(1, width // 24), max(1, height // 24)), Image.Resampling.BILINEAR)
        layered = tiny.resize((width, height), Image.Resampling.NEAREST)
        layered = layered.filter(ImageFilter.GaussianBlur(radius=max(4, min(width, height) / 18)))
        # Deterministic checker-noise avoids randomness while adding reconstruction resistance.
        pixels = layered.load()
        for y in range(height):
            for x in range(width):
                if (x + y) % 7 == 0:
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (max(0, r - 16), max(0, g - 16), max(0, b - 16))
        out.paste(layered, (left, top))
    return out


METHODS = {
    "solid_mask_black": apply_solid_mask_black,
    "solid_mask_mean_color": apply_solid_mask_mean_color,
    "layered_blur_downscale_noise": apply_layered_blur_downscale_noise,
}


def atomic_write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["relative_path", "method", "output_path", "boxes_processed", "runtime_seconds", "status"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--raw-root", default="data/castle2024/raw")
    parser.add_argument("--methods", nargs="+", default=["solid_mask_black", "layered_blur_downscale_noise"])
    parser.add_argument("--output-root", default="outputs/runs/stronger_baselines/outputs")
    parser.add_argument("--output-manifest", default="outputs/runs/stronger_baselines/manifest.csv")
    args = parser.parse_args()

    boxes_by_image = load_boxes(PROJECT_ROOT / args.detections)
    relative_paths = load_manifest_paths(PROJECT_ROOT / args.manifest)
    raw_root = PROJECT_ROOT / args.raw_root
    output_root = PROJECT_ROOT / args.output_root

    rows: list[dict[str, object]] = []
    for method in args.methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        for relative_path in relative_paths:
            input_path = raw_root / relative_path
            output_path = output_root / method / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            image = Image.open(input_path).convert("RGB")
            boxes = boxes_by_image.get(relative_path, [])
            output = METHODS[method](image, boxes)
            output.save(output_path)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "relative_path": relative_path,
                    "method": method,
                    "output_path": str(output_path.relative_to(PROJECT_ROOT)),
                    "boxes_processed": len(boxes),
                    "runtime_seconds": round(elapsed, 6),
                    "status": "ok",
                }
            )
    atomic_write_manifest(PROJECT_ROOT / args.output_manifest, rows)


if __name__ == "__main__":
    main()
