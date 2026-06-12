#!/usr/bin/env python3
"""Evaluate full-resolution SSIM and LPIPS with parallel CPU and batched CUDA work."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/castle2024/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def ssim_job(item: tuple[str, str]) -> float:
    raw_path, output_path = item
    with Image.open(raw_path).convert("RGB") as original, Image.open(output_path).convert("RGB") as anonymised:
        before = np.asarray(original)
        after = np.asarray(anonymised)
    return float(structural_similarity(before, after, channel_axis=2, data_range=255))


class ImagePairs(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        raw_path, output_path = self.pairs[index]
        with Image.open(raw_path).convert("RGB") as original, Image.open(output_path).convert("RGB") as anonymised:
            before = torch.from_numpy(np.asarray(original).copy()).permute(2, 0, 1)
            after = torch.from_numpy(np.asarray(anonymised).copy()).permute(2, 0, 1)
        return before, after, index


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    manifest_path = resolve(args.manifest)
    raw_root = resolve(args.raw_root)
    output_path = resolve(args.output)
    manifest = pd.read_csv(manifest_path)
    pairs = [
        (raw_root / str(row.relative_path), resolve(Path(str(row.output_path))))
        for row in manifest.itertuples(index=False)
    ]
    missing = [(str(a), str(b)) for a, b in pairs if not a.exists() or not b.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} image pairs; first={missing[0]}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        ssim_scores = list(
            tqdm(
                executor.map(ssim_job, [(str(a), str(b)) for a, b in pairs], chunksize=2),
                total=len(pairs),
                desc="SSIM",
            )
        )

    model = lpips.LPIPS(net="alex").eval().cuda()
    loader = DataLoader(
        ImagePairs(pairs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    lpips_scores = np.zeros(len(pairs), dtype=np.float64)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for before, after, indices in tqdm(loader, desc="LPIPS", total=len(loader)):
            before = before.cuda(non_blocking=True).float().div_(127.5).sub_(1.0)
            after = after.cuda(non_blocking=True).float().div_(127.5).sub_(1.0)
            values = model(before, after).reshape(-1).cpu().numpy()
            lpips_scores[indices.numpy()] = values

    method = str(manifest.method.iloc[0])
    detailed = [
        {
            "relative_path": str(row.relative_path),
            "method": method,
            "ssim": float(ssim_scores[index]),
            "lpips": float(lpips_scores[index]),
        }
        for index, row in enumerate(manifest.itertuples(index=False))
    ]
    payload = {
        "summary": {
            method: {
                "ssim_mean": float(np.mean(ssim_scores)),
                "ssim_std": float(np.std(ssim_scores)),
                "lpips_mean": float(np.mean(lpips_scores)),
                "lpips_std": float(np.std(lpips_scores)),
                "sample_count": len(pairs),
                "evaluation_resolution": "3840x2160",
                "lpips_device": "cuda",
                "lpips_batch_size": args.batch_size,
                "lpips_peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        },
        "detailed": detailed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
