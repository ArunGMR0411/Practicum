#!/usr/bin/env python3

"""Generate SCRFD-assisted face-box proposals for the CASTLE annotation pack."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

try:
    from insightface.model_zoo import get_model
except ModuleNotFoundError as exc:  # pragma: no cover - env-specific
    raise SystemExit("insightface is required for SCRFD proposals") from exc


PACK_ROOT = PROJECT_ROOT / "data" / "castle2024" / "annotations" / "face_detection" / "02_egocentric_stress_500"
TASKS_CSV_PATH = PROJECT_ROOT / "data" / "thesis_manifests" / "final_face_detection_500.csv"
PROPOSALS_PATH = PACK_ROOT / "manifest.csv"
RAW_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"
SCRFD_MODEL_PATH = Path("/home/arun-gmr/.insightface/models/buffalo_l/det_10g.onnx")


def save_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    """Atomically save proposal rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
        "annotator_id",
        "annotation_round",
        "condition_label",
        "notes",
    ]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    """Run SCRFD over the annotation pack and emit proposed face boxes."""
    logger = get_logger("scrfd_annotation_assist")
    logger.info("Initialising SCRFD-assisted annotation workflow")
    detector = get_model(str(SCRFD_MODEL_PATH), providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=0, input_size=(640, 640))

    proposal_rows: list[dict[str, object]] = []
    image_count = 0
    proposal_count = 0

    with TASKS_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_count += 1
            image_path = PROJECT_ROOT / row.get("image_path", "")
            if not image_path.is_file():
                image_path = RAW_ROOT / row["relative_path"]
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                logger.warning("Skipping unreadable image %s", image_path)
                continue
            height, width = image_bgr.shape[:2]
            faces, _ = detector.detect(image_bgr, input_size=(640, 640), max_num=0, metric="default")
            if faces is None:
                continue
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face[:4]]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                if x2 <= x1 or y2 <= y1:
                    continue
                score = float(face[4]) if len(face) >= 5 else 1.0
                proposal_rows.append(
                    {
                        "image_id": row["relative_path"],
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": round(score, 6),
                        "annotator_id": "scrfd_assist",
                        "annotation_round": 0,
                        "condition_label": row.get("condition_label", ""),
                        "notes": "Proposed by SCRFD; human verification required.",
                    }
                )
                proposal_count += 1

    save_rows(proposal_rows, PROPOSALS_PATH)
    logger.info("Saved %s proposals across %s images to %s", proposal_count, image_count, PROPOSALS_PATH)
    print(f"Saved {proposal_count} SCRFD proposals across {image_count} images to {PROPOSALS_PATH}")


if __name__ == "__main__":
    main()
