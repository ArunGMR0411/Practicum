#!/usr/bin/env python3

"""Validate FaceNet cosine similarity as a proxy for AdaFace on dev anonymised crops."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.castle_loader import CASTLEDataset


def load_boxes_by_image(path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["image_id"]].append(
                (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            )
    return grouped


def load_adaface_reference(path: Path) -> dict[str, dict[tuple[str, int], float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_method: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)
    for row in data:
        by_method[row["method"]][(row["image_id"], int(row["box_idx"]))] = float(
            row["adaface_cosine_sim"]
        )
    return by_method


class FaceNetEvaluator:
    def __init__(self, device: str = "cpu") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @torch.no_grad()
    def extract_embeddings(
        self, crops: list[Image.Image], batch_size: int = 64
    ) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)

        outputs: list[np.ndarray] = []
        for start in range(0, len(crops), batch_size):
            batch_crops = crops[start : start + batch_size]
            tensors = []
            for crop in batch_crops:
                resized = crop.resize((160, 160), Image.Resampling.BILINEAR).convert(
                    "RGB"
                )
                tensor = TF.to_tensor(resized) * 255.0
                tensor = fixed_image_standardization(tensor)
                tensors.append(tensor)
            batch = torch.stack(tensors).to(self.device)
            embeddings = self.model(batch)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            outputs.append(embeddings.cpu().numpy())

        return np.concatenate(outputs, axis=0)


def build_crops(
    manifest_path: Path,
    detections_path: Path,
    output_root: Path,
    methods: list[str],
) -> tuple[
    list[Image.Image],
    dict[str, list[Image.Image]],
    list[tuple[str, int]],
]:
    dataset = CASTLEDataset(str(manifest_path), return_format="pil", filters={})
    boxes_by_image = load_boxes_by_image(detections_path)

    gallery_crops: list[Image.Image] = []
    query_crops_by_method: dict[str, list[Image.Image]] = defaultdict(list)
    crop_keys: list[tuple[str, int]] = []

    for item in dataset:
        image = item["image"]
        relative_path = item["metadata"]["relative_path"]
        boxes = boxes_by_image.get(relative_path, [])

        anon_imgs: dict[str, Image.Image] = {}
        for method in methods:
            anon_path = output_root / method / relative_path
            if anon_path.exists():
                with Image.open(anon_path) as anon_img:
                    anon_imgs[method] = anon_img.copy()

        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            if x2 <= x1 or y2 <= y1:
                continue
            gallery_crops.append(image.crop((x1, y1, x2, y2)))
            crop_keys.append((relative_path, idx))
            for method in methods:
                anon_img = anon_imgs.get(method)
                if anon_img is not None:
                    query_crops_by_method[method].append(anon_img.crop((x1, y1, x2, y2)))

    return gallery_crops, query_crops_by_method, crop_keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/01_protocol/supporting_protocols/01_development_300.csv")
    parser.add_argument(
        "--detections", default="outputs/02_face_detection/01_yolo_predictions_run.csv"
    )
    parser.add_argument("--methods", nargs="+", default=["blur", "pixelate"])
    parser.add_argument("--output-root", default="outputs/dev_anonymised")
    parser.add_argument("--reid-results", default="outputs/experimental_runs/classical_baselines/reid_results.json")
    parser.add_argument("--output-json", default="outputs/proxy_validation.json")
    parser.add_argument("--output-plot", default="outputs/proxy_validation.png")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    detections_path = PROJECT_ROOT / args.detections
    output_root = PROJECT_ROOT / args.output_root
    reid_results_path = PROJECT_ROOT / args.reid_results
    output_json_path = PROJECT_ROOT / args.output_json
    output_plot_path = PROJECT_ROOT / args.output_plot

    gallery_crops, query_crops_by_method, crop_keys = build_crops(
        manifest_path=manifest_path,
        detections_path=detections_path,
        output_root=output_root,
        methods=args.methods,
    )
    if not gallery_crops:
        raise ValueError("No face crops available for proxy validation.")

    adaface_reference = load_adaface_reference(reid_results_path)
    evaluator = FaceNetEvaluator(device=args.device)
    gallery_embeddings = evaluator.extract_embeddings(gallery_crops)

    fig, axes = plt.subplots(1, len(args.methods), figsize=(7 * len(args.methods), 6))
    if len(args.methods) == 1:
        axes = [axes]

    payload: dict[str, object] = {
        "version": "1.0",
        "detector_conditioning": "yolo_run",
        "crop_count": len(crop_keys),
        "methods": {},
    }

    for axis, method in zip(axes, args.methods, strict=True):
        query_crops = query_crops_by_method.get(method, [])
        if len(query_crops) != len(crop_keys):
            raise ValueError(
                f"Method {method} has {len(query_crops)} crops, expected {len(crop_keys)}."
            )
        query_embeddings = evaluator.extract_embeddings(query_crops)
        facenet_cos = np.sum(query_embeddings * gallery_embeddings, axis=1)

        adaface_scores = []
        facenet_scores = []
        missing = 0
        for idx, key in enumerate(crop_keys):
            score = adaface_reference.get(method, {}).get(key)
            if score is None:
                missing += 1
                continue
            adaface_scores.append(score)
            facenet_scores.append(float(facenet_cos[idx]))

        if len(adaface_scores) < 2:
            raise ValueError(f"Not enough paired scores for method {method}.")

        r_value, p_value = pearsonr(facenet_scores, adaface_scores)
        axis.scatter(facenet_scores, adaface_scores, alpha=0.35, s=14)
        axis.set_title(f"{method}: r={r_value:.3f}")
        axis.set_xlabel("FaceNet cosine similarity")
        axis.set_ylabel("AdaFace cosine similarity")
        axis.grid(alpha=0.2)

        payload["methods"][method] = {
            "paired_crop_count": len(adaface_scores),
            "missing_reference_scores": missing,
            "pearson_r": float(r_value),
            "pearson_p_value": float(p_value),
            "facenet_cosine_mean": float(np.mean(facenet_scores)),
            "adaface_cosine_mean": float(np.mean(adaface_scores)),
            "proxy_acceptable": bool(r_value >= 0.80),
        }

    fig.suptitle("FaceNet vs AdaFace Cosine Similarity on Dev Anonymised Crops")
    fig.tight_layout()
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    payload["overall_proxy_acceptable"] = all(
        method_data["proxy_acceptable"]
        for method_data in payload["methods"].values()
    )
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved plot to {output_plot_path}")


if __name__ == "__main__":
    main()
