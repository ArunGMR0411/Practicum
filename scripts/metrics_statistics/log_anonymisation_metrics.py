#!/usr/bin/env python3

"""Log full-scale anonymisation metrics to W&B when available, else filesystem JSON."""

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
    parser.add_argument(
        "--results-json",
        default="outputs/experimental_runs/classical_baselines/anonymisation_full_results_yolo_scrfd_fallback.json",
    )
    parser.add_argument("--run-group", default="anonymisation")
    parser.add_argument("--output-json", default="outputs/experimental_runs/classical_baselines/anonymisation_metrics_log.json")
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
    payload = load_json(PROJECT_ROOT / args.results_json)

    outputs_dir = PROJECT_ROOT / config.get("paths", {}).get("outputs_dir", "outputs")
    logs_dir = PROJECT_ROOT / config.get("paths", {}).get("logs_dir", "logs")
    logger = get_logger("anonymisation_metrics", log_dir=logs_dir)

    run_id = build_run_id("anonymisation")
    enable_wandb = bool(config.get("evaluation_switches", {}).get("enable_wandb", True))
    wandb_mode = resolve_wandb_mode(enable_wandb)
    metrics = flatten_metrics("anonymisation/full/", payload)

    tracking_backend = "filesystem"
    try:
        import wandb

        run = wandb.init(
            project="castle-anonymisation",
            group=args.run_group,
            name=run_id,
            config={"source_results_json": args.results_json},
            mode=wandb_mode,
        )
        run.log(metrics)
        run.summary["source_results_json"] = args.results_json
        run.summary["methods_evaluated"] = ",".join(payload.get("methods_evaluated", []))
        run.finish()
        tracking_backend = f"wandb_{wandb_mode}"
        logger.info("Logged anonymisation metrics via %s", tracking_backend)
    except Exception as exc:  # pragma: no cover
        logger.warning("W&B logging unavailable, falling back to filesystem record: %s", exc)

    record = {
        "run_id": run_id,
        "run_group": args.run_group,
        "tracking_backend": tracking_backend,
        "source_results_json": args.results_json,
        "metrics": metrics,
    }
    output_path = PROJECT_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
