#!/usr/bin/env python3

"""Summarise the current reviewed multimodal detector status into one artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-json", default="outputs/multimodal_dev_results.json")
    parser.add_argument("--text-reviewed-json", default="outputs/text_validation_subset_evaluation.json")
    parser.add_argument("--screen-reviewed-json", default="outputs/screen_validation_subset_evaluation.json")
    parser.add_argument("--output-json", default="outputs/multimodal_reviewed_status.json")
    args = parser.parse_args()

    multimodal = json.loads((PROJECT_ROOT / args.multimodal_json).read_text(encoding="utf-8"))
    text_eval = json.loads((PROJECT_ROOT / args.text_reviewed_json).read_text(encoding="utf-8"))
    screen_eval = json.loads((PROJECT_ROOT / args.screen_reviewed_json).read_text(encoding="utf-8"))

    best_text_name, best_text_payload = max(
        text_eval["backends"].items(),
        key=lambda item: item[1]["metrics"]["f1"],
    )
    screen_metrics = screen_eval["detector_vs_manual"]["metrics"]

    output = {
        "version": "1.0",
        "multimodal_dev_artifact": args.multimodal_json,
        "images_processed": multimodal["images_processed"],
        "ocr_suppression_fraction": multimodal["suppressed_region_fraction"],
        "provisional_active_stack": {
            "text_detector": {
                "name": best_text_name,
                "rationale": "best reviewed-slice F1 among implemented backends",
                "reviewed_metrics": best_text_payload["metrics"],
            },
            "screen_detector": {
                "name": multimodal["screen_detector"],
                "rationale": "strong reviewed-slice precision, recall, and specificity on the current fallback path",
                "reviewed_metrics": screen_metrics,
            },
        },
        "reviewed_evidence": {
            "text_reviewed_slice": {
                "artifact": args.text_reviewed_json,
                "reviewed_rows": text_eval["reviewed_rows"],
                "proxy_vs_manual": text_eval["proxy_vs_manual"]["metrics"],
            },
            "screen_reviewed_slice": {
                "artifact": args.screen_reviewed_json,
                "reviewed_rows": screen_eval["reviewed_rows"],
                "proxy_vs_manual": screen_eval["proxy_vs_manual"]["metrics"],
            },
        },
        "current_interpretation": {
            "text_status": "provisional",
            "screen_status": "credible_provisional",
            "text_note": "text detector choice is grounded by reviewed evidence but still limited by a small reviewed slice",
            "screen_note": "screen fallback is supported by stronger reviewed-slice evidence than the initial proxy audit suggested",
        },
    }

    output_path = PROJECT_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
