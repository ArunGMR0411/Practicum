#!/usr/bin/env python3

"""Run a controlled diffusion tuning loop on the existing smoke subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.diffusion_anonymiser import DiffusionAnonymiser
from src.evaluation.perceptual_metrics import evaluate_manifest


def load_boxes_by_image(path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int, int]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped.setdefault(row["image_id"], []).append(
                (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            )
    return grouped


def load_relative_paths(manifest_path: Path) -> list[str]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["relative_path"] for row in reader]


def save_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "method", "output_path", "boxes_processed", "tiling_required"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_contact_sheet(
    relative_paths: list[str],
    raw_root: Path,
    variants_root: Path,
    variant_names: list[str],
    output_path: Path,
) -> None:
    thumb_w = 320
    thumb_h = 180
    label_h = 28
    cols = len(variant_names) + 1
    rows = len(relative_paths)
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for row_idx, rel_path in enumerate(relative_paths):
        images = [raw_root / rel_path]
        images.extend(variants_root / name / rel_path for name in variant_names)
        for col_idx, img_path in enumerate(images):
            if not img_path.exists():
                continue
            with Image.open(img_path) as image:
                thumb = image.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = col_idx * thumb_w
            y = row_idx * (thumb_h + label_h)
            canvas.paste(thumb, (x, y))
            label = "raw" if col_idx == 0 else variant_names[col_idx - 1]
            draw.text((x + 8, y + thumb_h + 6), label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/01_protocol/supporting_protocols/01_development_300.csv")
    parser.add_argument("--detections", default="outputs/runs/detection/dev_set_detections_mtcnn.csv")
    parser.add_argument("--raw-root", default="data/castle2024/raw")
    parser.add_argument("--output-root", default="outputs/diffusion_tuning_probe")
    parser.add_argument("--eval-scale", type=float, default=0.25)
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    detections_path = PROJECT_ROOT / args.detections
    raw_root = PROJECT_ROOT / args.raw_root
    output_root = PROJECT_ROOT / args.output_root

    output_root.mkdir(parents=True, exist_ok=True)

    relative_paths = load_relative_paths(manifest_path)
    boxes_by_image = load_boxes_by_image(detections_path)

    variants: list[dict[str, Any]] = [
        {
            "name": "legacy_control",
            "prompt": "portrait photo of a different anonymous person, natural face, preserve lighting and pose",
            "negative_prompt": "blurry, distorted, deformed, duplicate face, extra eyes, extra mouth, low quality, artifacts",
            "inference_steps": 5,
            "guidance_scale": 7.0,
            "strength_padding_ratio": 0.75,
            "mask_expansion_ratio": 0.0,
            "mask_feather_px": 0,
            "target_resolution": 512,
        },
        {
            "name": "mild_feather",
            "prompt": "portrait photo of an anonymous adult, realistic skin, preserve lighting and pose, seamless natural face",
            "negative_prompt": "blurry, distorted, deformed, duplicate face, extra eyes, extra mouth, cartoon, mask, white patch, low quality, artifacts",
            "inference_steps": 5,
            "guidance_scale": 6.8,
            "strength_padding_ratio": 0.85,
            "mask_expansion_ratio": 0.08,
            "mask_feather_px": 6,
            "target_resolution": 512,
        },
        {
            "name": "mild_context",
            "prompt": "realistic candid face, anonymous identity, preserve scene lighting, preserve camera viewpoint, seamless natural face",
            "negative_prompt": "blurry, distorted, deformed, duplicate face, extra eyes, extra mouth, cartoon, painting, waxy skin, white patch, low quality, artifacts",
            "inference_steps": 6,
            "guidance_scale": 6.5,
            "strength_padding_ratio": 1.0,
            "mask_expansion_ratio": 0.12,
            "mask_feather_px": 10,
            "target_resolution": 512,
        },
    ]

    summary: dict[str, Any] = {"variants": []}
    variant_names: list[str] = []

    for config in variants:
        variant_name = config["name"]
        variant_names.append(variant_name)
        variant_dir = output_root / variant_name
        manifest_rows: list[dict[str, Any]] = []
        anonymiser = DiffusionAnonymiser(
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"],
            inference_steps=config["inference_steps"],
            guidance_scale=config["guidance_scale"],
            strength_padding_ratio=config["strength_padding_ratio"],
            mask_expansion_ratio=config["mask_expansion_ratio"],
            mask_feather_px=config["mask_feather_px"],
            target_resolution=config["target_resolution"],
            device="cuda",
        )

        for rel_path in relative_paths:
            with Image.open(raw_root / rel_path) as image:
                result = anonymiser.anonymise(image.convert("RGB"), boxes_by_image.get(rel_path, []))
                output_path = variant_dir / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.image.save(output_path)
                manifest_rows.append(
                    {
                        "relative_path": rel_path,
                        "method": variant_name,
                        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
                        "boxes_processed": result.metadata.get("boxes_processed", 0),
                        "tiling_required": result.metadata.get("tiling_required", False),
                    }
                )

        manifest_out = output_root / f"{variant_name}_manifest.csv"
        save_manifest(manifest_rows, manifest_out)
        perceptual_out = output_root / f"{variant_name}_perceptual.json"
        perceptual_payload = evaluate_manifest(
            manifest_csv=manifest_out,
            raw_root=raw_root,
            output_json=perceptual_out,
            eval_scale=args.eval_scale,
        )
        metrics = perceptual_payload["summary"][variant_name]
        proxy_utility = metrics["ssim_mean"] * (1.0 - metrics["lpips_mean"])
        summary["variants"].append(
            {
                "name": variant_name,
                "config": config,
                "perceptual_summary": metrics,
                "proxy_utility_score": proxy_utility,
            }
        )

    summary["variants"].sort(key=lambda item: item["proxy_utility_score"], reverse=True)
    output_json = output_root / "summary.json"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    contact_sheet_path = output_root / "contact_sheet.webp"
    build_contact_sheet(relative_paths, raw_root, output_root, variant_names, contact_sheet_path)

    print(json.dumps(summary, indent=2))
    print(f"Contact sheet saved to {contact_sheet_path}")


if __name__ == "__main__":
    main()
