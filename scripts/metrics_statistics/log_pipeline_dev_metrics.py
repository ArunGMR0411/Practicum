#!/usr/bin/env python3

"""Log full dev-run pipeline metrics to W&B when available, else offline/filesystem."""

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
    parser.add_argument("--run-json", default="outputs/pipeline_dev_run.json")
    parser.add_argument("--timing-json", default="outputs/timing.json")
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
    run_payload = load_json(PROJECT_ROOT / args.run_json)
    timing_payload = load_json(PROJECT_ROOT / args.timing_json)

    outputs_dir = PROJECT_ROOT / config.get("paths", {}).get("outputs_dir", "outputs")
    logs_dir = PROJECT_ROOT / config.get("paths", {}).get("logs_dir", "logs")
    logger = get_logger("pipeline_dev_metrics", log_dir=logs_dir)

    run_id = build_run_id("evaluation")
    enable_wandb = bool(config.get("evaluation_switches", {}).get("enable_wandb", True))
    wandb_mode = resolve_wandb_mode(enable_wandb)

    summary_metrics: dict[str, float | int] = {
        "pipeline/images_attempted": int(run_payload.get("images_attempted", 0)),
        "pipeline/images_succeeded": int(run_payload.get("images_succeeded", 0)),
        "pipeline/images_failed": int(run_payload.get("images_failed", 0)),
    }
    summary_metrics.update(flatten_metrics("pipeline/timing/", timing_payload))

    tracking_backend = "filesystem"
    try:
        import wandb

        run = wandb.init(
            project="castle-anonymisation",
            group=args.run_group,
            name=run_id,
            config={
                "source_run_json": args.run_json,
                "source_timing_json": args.timing_json,
                "images_attempted": run_payload.get("images_attempted", 0),
                "images_succeeded": run_payload.get("images_succeeded", 0),
                "images_failed": run_payload.get("images_failed", 0),
            },
            mode=wandb_mode,
        )
        run.log(summary_metrics)
        run.summary["source_run_json"] = args.run_json
        run.summary["source_timing_json"] = args.timing_json
        run.summary["images_attempted"] = int(run_payload.get("images_attempted", 0))
        run.summary["images_succeeded"] = int(run_payload.get("images_succeeded", 0))
        run.summary["images_failed"] = int(run_payload.get("images_failed", 0))
        run.summary["timing_stage_count"] = int(len(timing_payload.get("stage_summary", {})))
        run.finish()
        tracking_backend = f"wandb_{wandb_mode}"
        logger.info("Logged pipeline dev metrics via %s", tracking_backend)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("W&B logging unavailable, falling back to filesystem record: %s", exc)

    record = {
        "run_id": run_id,
        "run_group": args.run_group,
        "tracking_backend": tracking_backend,
        "source_run_json": args.run_json,
        "source_timing_json": args.timing_json,
        "metrics": summary_metrics,
    }
    outputs_dir.mkdir(parents=True, exist_ok=True)
    local_path = outputs_dir / "pipeline_dev_metrics_log.json"
    local_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
