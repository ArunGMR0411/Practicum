"""Detection reviewer: free-port server and JSONL rewrite contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from privacy_pipeline_app.detection_reviewer import _free_port, start_detection_reviewer
from privacy_pipeline_app.wizard_workflow import write_detection_artifacts


def test_free_port_returns_open_port() -> None:
    port = _free_port()
    assert isinstance(port, int)
    assert 1024 < port < 65536


def test_start_detection_reviewer_serves_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    det = run / "detections"
    # Minimal image referenced by detection record
    img = tmp_path / "frame.jpg"
    from PIL import Image

    Image.new("RGB", (32, 32), (0, 0, 0)).save(img)
    write_detection_artifacts(
        det,
        [
            {
                "image_id": img.name,
                "local_path": str(img),
                "detector": "test",
                "faces": [{"x1": 1, "y1": 1, "x2": 10, "y2": 10, "score": 0.9}],
                "screens": [],
                "texts": [],
                "screen_sources": [],
                "text_policy": "none",
            }
        ],
    )
    url = start_detection_reviewer(str(run), host="127.0.0.1")
    assert url.startswith("http://127.0.0.1:")
    # CSV still readable after start
    faces = list(csv.DictReader((det / "face_boxes.csv").open(encoding="utf-8")))
    assert len(faces) == 1
    line = (det / "detections.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["image_id"] == img.name
