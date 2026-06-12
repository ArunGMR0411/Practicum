#!/usr/bin/env python3

"""Log cross-experiment synthesis outputs to W&B when available, with a filesystem mirror."""

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


DEFAULT_ARTIFACTS = (
    "outputs/09_traceability/01_evidence_index.csv",
    "outputs/03_anonymisation/01_all_methods_comparison.csv",
    "outputs/04_multimodal_privacy/01_multimodal_250_evidence/11_rq3_final_summary.md",
    "outputs/04_multimodal_privacy/01_multimodal_250_evidence/07_redaction_method_comparison.csv",
    "outputs/05_oapr/12_oapr_full_metric_summary.csv",
    "outputs/08_figures/01_privacy_utility_pareto.png",
    "outputs/08_figures/03_failure_taxonomy_heatmap.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-group", default="synthesis")
    parser.add_argument("--output-json", default="outputs/runs/synthesis/metrics_log.json")
    parser.add_argument("--artifact", action="append", dest="artifacts", default=[])
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_metrics(prefix: str, payload: Any) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            metrics.update(flatten_metrics(f"{prefix}{key}/", value))
    elif isinstance(payload, list):
        metrics[f"{prefix.rstrip('/')}/count"] = len(payload)
    elif isinstance(payload, (int, float)):
        metrics[prefix.rstrip("/")] = payload
    return metrics


def resolve_wandb_mode(enable_wandb: bool) -> str:
    if not enable_wandb:
        return "disabled"
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    return "offline"


def existing_artifacts(paths: list[str]) -> list[Path]:
    artifacts: list[Path] = []
    for artifact in paths:
        path = PROJECT_ROOT / artifact
        if path.exists():
            artifacts.append(path)
    return artifacts


def build_metrics() -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    index_path = PROJECT_ROOT / "outputs" / "submission_evidence" / "09_traceability" / "01_evidence_index.csv"
    figures_dir = PROJECT_ROOT / "outputs" / "submission_evidence" / "08_figures"
    fid_path = PROJECT_ROOT / "outputs" / "submission_evidence" / "06_statistics_and_review" / "03_fid_webp_baseline.json"
    if index_path.exists():
        metrics["synthesis/evidence_index/available"] = 1
    if figures_dir.exists():
        metrics["synthesis/figures/count"] = sum(1 for path in figures_dir.iterdir() if path.is_file())
    if fid_path.exists():
        fid = load_json(fid_path)
        metrics.update(flatten_metrics("synthesis/fid_baseline/", fid))
    return metrics


def main() -> int:
    args = parse_args()
    config = load_yaml(PROJECT_ROOT / args.config)
    logs_dir = PROJECT_ROOT / config.get("paths", {}).get("logs_dir", "logs")
    logger = get_logger("synthesis_metrics", log_dir=logs_dir)

    run_id = build_run_id("synthesis")
    enable_wandb = bool(config.get("evaluation_switches", {}).get("enable_wandb", True))
    wandb_mode = resolve_wandb_mode(enable_wandb)
    artifacts = existing_artifacts(list(DEFAULT_ARTIFACTS) + args.artifacts)
    metrics = build_metrics()

    tracking_backend = "filesystem"
    try:
        import wandb

        wandb_dir = PROJECT_ROOT / "outputs" / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project="castle-anonymisation",
            group=args.run_group,
            name=run_id,
            config={"artifacts": [str(path.relative_to(PROJECT_ROOT)) for path in artifacts]},
            mode=wandb_mode,
            dir=str(wandb_dir),
        )
        run.log(metrics)
        for path in artifacts:
            wandb.save(str(path), base_path=str(PROJECT_ROOT), policy="now")
        run.summary["artifact_count"] = len(artifacts)
        run.summary["tracking_note"] = "Offline mode is expected when WANDB_API_KEY is unavailable."
        run.finish()
        tracking_backend = f"wandb_{wandb_mode}"
        logger.info("Logged synthesis outputs via %s", tracking_backend)
    except Exception as exc:  # pragma: no cover
        logger.warning("W&B synthesis logging unavailable, falling back to filesystem record: %s", exc)

    record = {
        "run_id": run_id,
        "run_group": args.run_group,
        "tracking_backend": tracking_backend,
        "artifacts": [str(path.relative_to(PROJECT_ROOT)) for path in artifacts],
        "metrics": metrics,
        "note": "Filesystem mirror is authoritative when W&B is offline or unavailable.",
    }
    output_path = PROJECT_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
