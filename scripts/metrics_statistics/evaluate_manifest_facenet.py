#!/usr/bin/env python3
"""Evaluate manifest-backed outputs with an independent FaceNet attacker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def embeddings(crops: list[np.ndarray], model: InceptionResnetV1, batch_size: int) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            array = np.stack(crops[start : start + batch_size])
            tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float().cuda()
            tensor = torch.nn.functional.interpolate(
                tensor, size=(160, 160), mode="bilinear", align_corners=False
            )
            tensor = fixed_image_standardization(tensor)
            values.append(torch.nn.functional.normalize(model(tensor), dim=1).cpu().numpy())
    return np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--details-json", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/castle2024/raw"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")

    manifest = pd.read_csv(ROOT / args.manifest)
    manifest = manifest[manifest.method.eq(args.method)] if "method" in manifest else manifest
    boxes = pd.read_csv(ROOT / args.detections).groupby("image_id", sort=False)
    gallery_crops: list[np.ndarray] = []
    query_crops: list[np.ndarray] = []
    metadata = []
    for item in manifest.itertuples(index=False):
        if item.relative_path not in boxes.groups:
            continue
        source_path = ROOT / args.raw_root / item.relative_path
        output_path = ROOT / item.output_path
        with Image.open(source_path) as source, Image.open(output_path) as output:
            source = source.convert("RGB")
            output = output.convert("RGB")
            for box_index, box in enumerate(boxes.get_group(item.relative_path).itertuples(index=False)):
                x1 = max(0, min(int(box.x1), source.width))
                y1 = max(0, min(int(box.y1), source.height))
                x2 = max(0, min(int(box.x2), source.width))
                y2 = max(0, min(int(box.y2), source.height))
                if x2 <= x1 or y2 <= y1:
                    continue
                gallery_crops.append(np.asarray(source.crop((x1, y1, x2, y2)).resize((256, 256))).copy())
                query_crops.append(np.asarray(output.crop((x1, y1, x2, y2)).resize((256, 256))).copy())
                metadata.append({"image_id": item.relative_path, "box_index": box_index})

    model = InceptionResnetV1(pretrained="vggface2").eval().cuda()
    gallery = embeddings(gallery_crops, model, args.batch_size)
    query = embeddings(query_crops, model, args.batch_size)
    cosine = np.sum(gallery * query, axis=1)
    details = []
    for item, value in zip(metadata, cosine, strict=True):
        details.append({
            **item,
            "method": args.method,
            "facenet_cosine_similarity": float(value),
            "hit_at_050": bool(value >= 0.50),
            "hit_at_060": bool(value >= 0.60),
            "hit_at_070": bool(value >= 0.70),
        })
    summary = {
        "method": args.method,
        "face_crop_count": len(details),
        "facenet_cosine_similarity_mean": float(np.mean(cosine)),
        "facenet_reid_rate_050": float(np.mean(cosine >= 0.50)),
        "facenet_reid_rate_060": float(np.mean(cosine >= 0.60)),
        "facenet_reid_rate_070": float(np.mean(cosine >= 0.70)),
        "device": "cuda",
        "batch_size": args.batch_size,
    }
    summary_path = ROOT / args.summary_json
    details_path = ROOT / args.details_json
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    details_path.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
