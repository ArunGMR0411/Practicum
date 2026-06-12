#!/usr/bin/env python3

"""Combine reviewed text-validation evaluations into one summary artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
    }


def merge_counts(payloads: list[dict], key_path: list[str]) -> dict[str, int]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for payload in payloads:
        node = payload
        for key in key_path:
            node = node[key]
        for count_key in counts:
            counts[count_key] += int(node[count_key])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "outputs/text_validation_subset_evaluation.json",
            "outputs/text_validation_extension_evaluation.json",
        ],
    )
    parser.add_argument("--output-json", default="outputs/text_validation_combined_summary.json")
    args = parser.parse_args()

    payloads = [load_json(PROJECT_ROOT / rel_path) for rel_path in args.inputs]

    combined = {
        "version": "1.0",
        "input_evaluations": args.inputs,
        "reviewed_rows_total": int(sum(payload["reviewed_rows"] for payload in payloads)),
        "manual_text_present_total": int(sum(payload["manual_text_present_count"] for payload in payloads)),
        "manual_text_absent_total": int(sum(payload["manual_text_absent_count"] for payload in payloads)),
        "manual_legible_text_total": int(sum(payload["manual_legible_text_count"] for payload in payloads)),
    }

    proxy_counts = merge_counts(payloads, ["proxy_vs_manual", "counts"])
    combined["proxy_vs_manual"] = {
        "counts": proxy_counts,
        "metrics": metric_from_counts(proxy_counts),
    }

    backends: dict[str, dict[str, object]] = {}
    backend_names = payloads[0]["backends"].keys()
    for backend_name in backend_names:
        backend_counts = merge_counts(payloads, ["backends", backend_name, "counts"])
        backends[backend_name] = {
            "counts": backend_counts,
            "metrics": metric_from_counts(backend_counts),
        }
    combined["backends"] = backends

    output_path = PROJECT_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
