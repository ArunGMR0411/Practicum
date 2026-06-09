"""Perceptual utility metrics (SSIM and LPIPS) for CASTLE face anonymisation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
import lpips
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """Compute structural similarity index (SSIM) between two PIL images.

    Both images are converted to RGB numpy arrays.
    """
    img1_rgb = img1.convert("RGB")
    img2_rgb = img2.convert("RGB")
    if img1_rgb.size != img2_rgb.size:
        img2_rgb = img2_rgb.resize(img1_rgb.size, Image.Resampling.LANCZOS)

    np1 = np.array(img1_rgb)
    np2 = np.array(img2_rgb)

    min_dim = min(np1.shape[0], np1.shape[1], np2.shape[0], np2.shape[1])
    if min_dim < 3:
        return 1.0 if np.array_equal(np1, np2) else 0.0

    win_size = 7
    if min_dim < win_size:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1

    # Use the largest valid odd window for very small images.
    score = structural_similarity(
        np1,
        np2,
        channel_axis=-1,
        data_range=255,
        win_size=win_size,
    )
    return float(score)


class LPIPSEvaluator:
    """Helper to load and evaluate LPIPS scores on PIL images."""

    def __init__(self, net: str = "alex", use_gpu: bool = True) -> None:
        """Initialise the LPIPS model."""
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        # Initialize the model; this might download weights on first load.
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute LPIPS distance between two PIL images."""
        img1_rgb = img1.convert("RGB")
        img2_rgb = img2.convert("RGB")
        if img1_rgb.size != img2_rgb.size:
            img2_rgb = img2_rgb.resize(img1_rgb.size, Image.Resampling.LANCZOS)

        np1 = np.array(img1_rgb)
        np2 = np.array(img2_rgb)

        # Convert to tensors normalized to [-1, 1] using LPIPS helper
        t1 = lpips.im2tensor(np1).to(self.device)
        t2 = lpips.im2tensor(np2).to(self.device)

        dist = self.model(t1, t2)
        return float(dist.item())


def evaluate_manifest(
    manifest_csv: str | Path,
    raw_root: str | Path,
    output_json: str | Path | None = None,
    net: str = "alex",
    use_gpu: bool = True,
    eval_scale: float | None = None,
) -> dict[str, Any]:
    """Load anonymised manifest, match with raw originals, and compute SSIM/LPIPS."""
    manifest_path = Path(manifest_csv)
    raw_path = Path(raw_root)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Load rows from manifest
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} entries from {manifest_path}")

    # Initialize evaluator
    evaluator = LPIPSEvaluator(net=net, use_gpu=use_gpu)

    # To group individual scores for summarization
    ssim_scores_by_method: dict[str, list[float]] = {}
    lpips_scores_by_method: dict[str, list[float]] = {}
    detailed_results: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="Evaluating frames"):
        rel_path = row["relative_path"]
        method = row["method"]
        anon_path_str = row["output_path"]

        # Resolve paths
        anon_img_path = Path(anon_path_str)
        if not anon_img_path.is_absolute():
            anon_img_path = PROJECT_ROOT / anon_img_path

        orig_img_path = raw_path / rel_path

        if not anon_img_path.exists():
            print(f"Warning: Anonymised image not found at {anon_img_path}. Skipping.")
            continue
        if not orig_img_path.exists():
            print(f"Warning: Original image not found at {orig_img_path}. Skipping.")
            continue

        try:
            # Load images
            with Image.open(orig_img_path) as orig_img, Image.open(anon_img_path) as anon_img:
                if eval_scale is not None and eval_scale != 1.0:
                    w = max(1, int(round(orig_img.width * eval_scale)))
                    h = max(1, int(round(orig_img.height * eval_scale)))
                    orig_img = orig_img.resize((w, h), Image.Resampling.LANCZOS)
                    anon_img = anon_img.resize((w, h), Image.Resampling.LANCZOS)

                ssim_score = compute_ssim(orig_img, anon_img)
                lpips_score = evaluator.compute_lpips(orig_img, anon_img)

            # Store scores
            ssim_scores_by_method.setdefault(method, []).append(ssim_score)
            lpips_scores_by_method.setdefault(method, []).append(lpips_score)

            detailed_results.append(
                {
                    "relative_path": rel_path,
                    "method": method,
                    "ssim": ssim_score,
                    "lpips": lpips_score,
                }
            )
        except Exception as exc:
            print(f"Error processing {rel_path} with method {method}: {exc}")

    # Compute summary statistics
    summary: dict[str, dict[str, Any]] = {}
    for method in ssim_scores_by_method:
        ssims = ssim_scores_by_method[method]
        lpipss = lpips_scores_by_method[method]

        summary[method] = {
            "ssim_mean": float(np.mean(ssims)),
            "ssim_std": float(np.std(ssims)),
            "lpips_mean": float(np.mean(lpipss)),
            "lpips_std": float(np.std(lpipss)),
            "sample_count": len(ssims),
        }

    results = {
        "summary": summary,
        "detailed": detailed_results,
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved results to {out_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SSIM and LPIPS perceptual metrics on anonymised frames.")
    parser.add_argument("--manifest", default="outputs/dev_anonymised_manifest.csv", help="Path to anonymised output manifest CSV.")
    parser.add_argument("--raw-root", default="data/castle2024/raw", help="Path to raw/original WebP frames directory.")
    parser.add_argument("--output", default="outputs/anonymisation_perceptual_results.json", help="Path to save the JSON results.")
    parser.add_argument("--net", default="alex", choices=["alex", "vgg"], help="Network architecture for LPIPS.")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU for LPIPS evaluation.")
    parser.add_argument("--eval-scale", type=float, default=1.0, help="Scaling factor to resize images before evaluation.")
    args = parser.parse_args()

    # Resolve paths relative to project root if they are relative
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    raw_path = Path(args.raw_root)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    evaluate_manifest(
        manifest_csv=manifest_path,
        raw_root=raw_path,
        output_json=output_path,
        net=args.net,
        use_gpu=not args.no_gpu,
        eval_scale=args.eval_scale,
    )


if __name__ == "__main__":
    main()
