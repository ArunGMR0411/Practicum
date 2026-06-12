#!/usr/bin/env python3

"""Log routing results and timing metrics to W&B when available, else offline/filesystem."""

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
    parser.add_argument("--routing-results-json", default="outputs/runs/routing_baseline/routing_dev_results.json")
    parser.add_argument("--runtime-json", default="outputs/runs/routing_baseline/routing_runtime_benchmark_dev.json")
    parser.add_argument("--run-group", default="routing")
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
    routing_payload = load_json(PROJECT_ROOT / args.routing_results_json)
    runtime_payload = load_json(PROJECT_ROOT / args.runtime_json)

    outputs_dir = PROJECT_ROOT / config.get("paths", {}).get("outputs_dir", "outputs")
    logs_dir = PROJECT_ROOT / config.get("paths", {}).get("logs_dir", "logs")
    logger = get_logger("routing_metrics", log_dir=logs_dir)

    run_id = build_run_id("routing")
    enable_wandb = bool(config.get("evaluation_switches", {}).get("enable_wandb", True))
    wandb_mode = resolve_wandb_mode(enable_wandb)

    summary_metrics: dict[str, float | int] = {}
    summary_metrics.update(flatten_metrics("routing/results/", routing_payload))
    summary_metrics.update(flatten_metrics("routing/runtime/", runtime_payload))

    tracking_backend = "filesystem"
    try:
        import wandb

        run = wandb.init(
            project="castle-anonymisation",
            group=args.run_group,
            name=run_id,
            config={
                "source_routing_results_json": args.routing_results_json,
                "source_runtime_json": args.runtime_json,
                "best_fixed_strategy": routing_payload.get("best_fixed_strategy", ""),
                "n_frames_evaluated": routing_payload.get("n_frames_evaluated", 0),
            },
            mode=wandb_mode,
        )
        run.log(summary_metrics)
        run.summary["source_routing_results_json"] = args.routing_results_json
        run.summary["source_runtime_json"] = args.runtime_json
        run.summary["best_fixed_strategy"] = routing_payload.get("best_fixed_strategy", "")
        run.summary["n_frames_evaluated"] = int(routing_payload.get("n_frames_evaluated", 0))
        run.finish()
        tracking_backend = f"wandb_{wandb_mode}"
        logger.info("Logged routing metrics via %s", tracking_backend)
    except Exception as exc:  # pragma: no cover
        logger.warning("W&B logging unavailable, falling back to filesystem record: %s", exc)

    record = {
        "run_id": run_id,
        "run_group": args.run_group,
        "tracking_backend": tracking_backend,
        "source_routing_results_json": args.routing_results_json,
        "source_runtime_json": args.runtime_json,
        "metrics": summary_metrics,
    }
    outputs_dir.mkdir(parents=True, exist_ok=True)
    local_path = outputs_dir / "routing_metrics_log.json"
    local_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
