#!/usr/bin/env python3

"""Download the official StyleID pretrained assets into third_party/styleid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STYLEID_ROOT = PROJECT_ROOT / "third_party" / "styleid"

ASSETS = {
    "stylegan2_ffhq": {
        "url": "https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT",
        "path": STYLEID_ROOT / "pretrained_models" / "stylegan2-ffhq-config-f.pt",
        "required": True,
    },
    "psp_seg": {
        "url": "https://drive.google.com/uc?id=1VpEKc6E6yG3xhYuZ0cq8D2_1CbT0Dstz",
        "path": STYLEID_ROOT / "pretrained_models" / "psp_celebs_seg_to_face.pt",
        "required": True,
    },
    "curricular_face": {
        "url": "https://drive.google.com/uc?id=1f4IwVa2-Bn9vWLwB-bUwm53U_MlvinAj",
        "path": STYLEID_ROOT / "pretrained_models" / "CurricularFace_Backbone.pth",
        "required": True,
    },
    "attr_net": {
        "url": "https://drive.google.com/uc?id=1wjRJM8O7RYOKJEN2X-4Dtg5XDkXhp9dh",
        "path": STYLEID_ROOT / "pretrained_models" / "mobilenet_celeba.pth",
        "required": True,
    },
    "unet": {
        "url": "https://drive.google.com/uc?id=112SilQfnCM3_-Zugik6fu_EuXlmDP4Vi",
        "path": STYLEID_ROOT / "pretrained_models" / "unet_model.pth",
        "required": True,
    },
    "ir_se50": {
        "url": "https://drive.google.com/uc?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn",
        "path": STYLEID_ROOT / "pretrained_models" / "model_ir_se50.pth",
        "required": False,
    },
    "shape_predictor": {
        "url": "https://drive.google.com/uc?id=1sePXzGZBzm1PAKvbEhcQ07LLojS2AkMu",
        "path": STYLEID_ROOT / "pretrained_models" / "shape_predictor_68_face_landmarks.dat",
        "required": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary: list[dict[str, object]] = []
    for name, spec in ASSETS.items():
        required = bool(spec["required"])
        if not required and not args.include_optional:
            continue
        output_path = Path(spec["path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exists_before = output_path.is_file()
        if args.dry_run or (exists_before and not args.force):
            summary.append(
                {
                    "name": name,
                    "path": str(output_path),
                    "required": required,
                    "status": "present" if exists_before else "missing",
                }
            )
            continue
        import gdown

        result = gdown.download(str(spec["url"]), str(output_path), quiet=False, fuzzy=True)
        summary.append(
            {
                "name": name,
                "path": str(output_path),
                "required": required,
                "status": "downloaded" if result and output_path.is_file() else "failed",
            }
        )
    print(json.dumps(summary, indent=2))
    if any(item["required"] and item["status"] in {"missing", "failed"} for item in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
