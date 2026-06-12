#!/usr/bin/env python3

"""Log multimodal results and reviewed validation metrics to W&B or filesystem fallback."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.run_ids import build_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--easyocr-results", default="outputs/multimodal_dev_results.json")
    parser.add_argument("--east-results", default="outputs/multimodal_dev_east_results.json")
    parser.add_argument("--doctr-results", default="outputs/multimodal_dev_doctr_results.json")
    parser.add_argument("--text-review-json", default="outputs/text_validation_subset_evaluation.json")
    parser.add_argument("--screen-review-json", default="outputs/screen_validation_subset_evaluation.json")
    parser.add_argument("--status-json", default="outputs/multimodal_reviewed_status.json")
    parser.add_argument("--run-group", default="evaluation")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_metrics(prefix: str, payload: dict[str, Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            metrics.update(flatten_metrics(f"{prefix}{key}/", value))
        elif isinstance(value, (int, float)):
            metrics[f"{prefix}{key}"] = value
    return metrics


def resolve_wandb_mode(enable_wandb: bool) -> str:
    if not enable_wandb:
        return "disabled"
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    return "offline"


def main() -> int:
    args = parse_args()
    config = load_yaml(PROJECT_ROOT / args.config)
    easyocr_payload = load_json(PROJECT_ROOT / args.easyocr_results)
    east_payload = load_json(PROJECT_ROOT / args.east_results)
    doctr_payload = load_json(PROJECT_ROOT / args.doctr_results)
    text_review_payload = load_json(PROJECT_ROOT / args.text_review_json)
    screen_review_payload = load_json(PROJECT_ROOT / args.screen_review_json)
    status_payload = load_json(PROJECT_ROOT / args.status_json)

    outputs_dir = PROJECT_ROOT / config.get("paths", {}).get("outputs_dir", "outputs")
    logs_dir = PROJECT_ROOT / config.get("paths", {}).get("logs_dir", "logs")
    logger = get_logger("multimodal_metrics", log_dir=logs_dir)

    run_id = build_run_id("evaluation")
    enable_wandb = bool(config.get("evaluation_switches", {}).get("enable_wandb", True))
    wandb_mode = resolve_wandb_mode(enable_wandb)

    metrics: dict[str, float | int] = {}
    metrics.update(flatten_metrics("multimodal/easyocr/", easyocr_payload))
    metrics.update(flatten_metrics("multimodal/east/", east_payload))
    metrics.update(flatten_metrics("multimodal/doctr/", doctr_payload))
    metrics.update(flatten_metrics("multimodal/text_review/", text_review_payload))
    metrics.update(flatten_metrics("multimodal/screen_review/", screen_review_payload))
    metrics.update(flatten_metrics("multimodal/status/", status_payload))

    tracking_backend = "filesystem"
    try:
        import wandb

        run = wandb.init(
            project="castle-anonymisation",
            group=args.run_group,
            name=run_id,
            config={
                "source_easyocr_results_json": args.easyocr_results,
                "source_east_results_json": args.east_results,
                "source_doctr_results_json": args.doctr_results,
                "source_text_review_json": args.text_review_json,
                "source_screen_review_json": args.screen_review_json,
                "source_status_json": args.status_json,
            },
            mode=wandb_mode,
        )
        run.log(metrics)
        run.finish()
        tracking_backend = f"wandb_{wandb_mode}"
        logger.info("Logged multimodal metrics via %s", tracking_backend)
    except Exception as exc:  # pragma: no cover
        logger.warning("W&B logging unavailable, falling back to filesystem record: %s", exc)

    record = {
        "run_id": run_id,
        "run_group": args.run_group,
        "tracking_backend": tracking_backend,
        "sources": {
            "easyocr_results_json": args.easyocr_results,
            "east_results_json": args.east_results,
            "doctr_results_json": args.doctr_results,
            "text_review_json": args.text_review_json,
            "screen_review_json": args.screen_review_json,
            "status_json": args.status_json,
        },
        "metrics": metrics,
    }
    outputs_dir.mkdir(parents=True, exist_ok=True)
    local_path = outputs_dir / "multimodal_metrics_log.json"
    local_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
