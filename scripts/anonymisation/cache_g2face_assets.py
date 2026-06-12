#!/usr/bin/env python3

"""Download the official G2Face pretrained asset folder into third_party/g2face."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
G2FACE_ROOT = PROJECT_ROOT / "third_party" / "g2face"
G2FACE_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1dp4RyL5Z_28rxzyPCll_6gcIgf-J-feV"

EXPECTED_PATHS = (
    G2FACE_ROOT / "weights" / "G2Face.pth",
    G2FACE_ROOT / "weights" / "epoch_20.pth",
    G2FACE_ROOT / "pretrain" / "ms1mv3_arcface_r50.pth",
    G2FACE_ROOT / "model" / "d3dfr" / "BFM" / "BFM_model_front.mat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def existing_summary() -> list[dict[str, object]]:
    return [
        {"path": str(path), "status": "present" if path.exists() else "missing"}
        for path in EXPECTED_PATHS
    ]


def place_downloaded_file(download_root: Path, name: str, destination: Path) -> None:
    matches = list(download_root.rglob(name))
    if not matches:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(matches[0], destination)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(existing_summary(), indent=2))
        if any(item["status"] == "missing" for item in existing_summary()):
            raise SystemExit(1)
        return

    if all(path.exists() for path in EXPECTED_PATHS) and not args.force:
        print(json.dumps(existing_summary(), indent=2))
        return

    import gdown

    download_root = G2FACE_ROOT / "_downloaded_weights"
    download_root.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(G2FACE_DRIVE_FOLDER, output=str(download_root), quiet=False, use_cookies=False)

    placements = {
        "G2Face.pth": G2FACE_ROOT / "weights" / "G2Face.pth",
        "epoch_20.pth": G2FACE_ROOT / "weights" / "epoch_20.pth",
        "ms1mv3_arcface_r50.pth": G2FACE_ROOT / "pretrain" / "ms1mv3_arcface_r50.pth",
        "BFM_model_front.mat": G2FACE_ROOT / "model" / "d3dfr" / "BFM" / "BFM_model_front.mat",
    }
    for filename, destination in placements.items():
        place_downloaded_file(download_root, filename, destination)

    summary = existing_summary()
    shutil.rmtree(download_root, ignore_errors=True)
    print(json.dumps(summary, indent=2))
    if any(item["status"] == "missing" for item in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
