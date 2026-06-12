#!/usr/bin/env python3
"""Cause-driven multimodal detection fix campaign (max 10 configs).

Works from locked CRAFT/YOLO predictions plus optional lightweight YOLO
re-passes. Selects parameters on development; reports held-out only for ranking.
Does not promote automatically - writes a campaign report for human/system gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.screen_detector import ScreenDetector

ANNOTATIONS = ROOT / "outputs/01_protocol/annotations/multimodal_250/reviewed_multimodal_250_with_boxes.csv"
EVIDENCE = ROOT / "outputs/04_multimodal_privacy/01_multimodal_250_evidence"
BASE_PRED = EVIDENCE / "03_selected_localisation_predictions.csv"
RAW = ROOT / "data/castle2024/raw"
OUT_DIR = EVIDENCE / "14_detection_fix_campaign"
Box = tuple[int, int, int, int]


def parse_boxes(value: object) -> list[Box]:
    if value is None or value == "" or value == "[]" or (isinstance(value, float) and np.isnan(value)):
        return []
    return [
        (int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"]))
        for item in json.loads(str(value))
    ]


def box_json(boxes: list[Box]) -> str:
    return json.dumps(
        [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in boxes],
        separators=(",", ":"),
    )


def area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / max(1, area(a) + area(b) - inter)


def centre(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def centre_inside(inner: Box, outer: Box) -> bool:
    x, y = centre(inner)
    return outer[0] <= x <= outer[2] and outer[1] <= y <= outer[3]


def any_corner_inside(inner: Box, outer: Box) -> bool:
    return any(
        outer[0] <= x <= outer[2] and outer[1] <= y <= outer[3]
        for x, y in (
            (inner[0], inner[1]),
            (inner[2], inner[1]),
            (inner[0], inner[3]),
            (inner[2], inner[3]),
        )
    )


def clamp_box(box: Box, width: int, height: int) -> Box | None:
    x1 = max(0, min(width - 1, box[0]))
    y1 = max(0, min(height - 1, box[1]))
    x2 = max(0, min(width, box[2]))
    y2 = max(0, min(height, box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def expand_box(box: Box, width: int, height: int, ratio: float) -> Box | None:
    bw, bh = box[2] - box[0], box[3] - box[1]
    mx, my = int(bw * ratio), int(bh * ratio)
    return clamp_box((box[0] - mx, box[1] - my, box[2] + mx, box[3] + my), width, height)


def union_boxes(primary: list[Box], secondary: list[Box], thr: float = 0.50) -> list[Box]:
    merged = list(primary)
    for candidate in secondary:
        if not any(iou(candidate, existing) >= thr for existing in merged):
            merged.append(candidate)
    return merged


def hull(boxes: list[Box], width: int, height: int, margin_ratio: float = 0.08) -> Box | None:
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return expand_box((x1, y1, x2, y2), width, height, margin_ratio)


def cluster_text_boxes(
    boxes: list[Box],
    *,
    width: int,
    height: int,
    link_frac: float = 0.08,
) -> list[list[Box]]:
    """Greedy spatial clustering by centre distance."""
    if not boxes:
        return []
    link = link_frac * max(width, height)
    remaining = list(boxes)
    clusters: list[list[Box]] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            keep = []
            for box in remaining:
                cx, cy = centre(box)
                if any(
                    abs(cx - centre(member)[0]) <= link and abs(cy - centre(member)[1]) <= link
                    for member in cluster
                ):
                    cluster.append(box)
                    changed = True
                else:
                    keep.append(box)
            remaining = keep
        clusters.append(cluster)
    return clusters


def hypothesize_screens_from_text(
    text_boxes: list[Box],
    *,
    width: int,
    height: int,
    min_count: int,
    require_lower_half: bool,
    min_cluster_area_frac: float,
    margin_ratio: float,
    link_frac: float,
) -> list[Box]:
    hyps: list[Box] = []
    for cluster in cluster_text_boxes(
        text_boxes, width=width, height=height, link_frac=link_frac
    ):
        if len(cluster) < min_count:
            continue
        if require_lower_half:
            cys = [centre(b)[1] / height for b in cluster]
            if float(np.mean(cys)) < 0.45:
                continue
        box = hull(cluster, width, height, margin_ratio=margin_ratio)
        if box is None:
            continue
        if area(box) / float(width * height) < min_cluster_area_frac:
            continue
        # Cap absurd full-frame hypotheses
        if area(box) / float(width * height) > 0.45:
            continue
        hyps.append(box)
    return hyps


def strip_text_in_screens(text_boxes: list[Box], screens: list[Box]) -> list[Box]:
    if not screens:
        return text_boxes
    return [
        box
        for box in text_boxes
        if not any(any_corner_inside(box, screen) or centre_inside(box, screen) for screen in screens)
    ]


def risk_state(text: bool, screen: bool) -> str:
    if text and screen:
        return "text_and_screen_present"
    if text:
        return "text_present"
    if screen:
        return "screen_present"
    return "no_text_screen_risk"


def image_metrics(
    rows: pd.DataFrame,
    gt_text: dict[str, list[Box]],
    gt_screen: dict[str, list[Box]],
    pred_text: dict[str, list[Box]],
    pred_screen: dict[str, list[Box]],
) -> dict[str, float | int]:
    # Image-level combined risk
    tp = fp = fn = tn = 0
    screen_tp = screen_fp = screen_fn = screen_tn = 0
    text_tp = text_fp = text_fn = text_tn = 0
    screen_box_hit = screen_box_miss = 0
    miss_ids: list[str] = []
    for image_id in rows["image_id"]:
        gt_t = bool(gt_text[image_id])
        gt_s = bool(gt_screen[image_id])
        pr_t = bool(pred_text.get(image_id))
        pr_s = bool(pred_screen.get(image_id))
        gt_risk = gt_t or gt_s
        pr_risk = pr_t or pr_s
        tp += int(gt_risk and pr_risk)
        fp += int((not gt_risk) and pr_risk)
        fn += int(gt_risk and (not pr_risk))
        tn += int((not gt_risk) and (not pr_risk))
        screen_tp += int(gt_s and pr_s)
        screen_fp += int((not gt_s) and pr_s)
        screen_fn += int(gt_s and (not pr_s))
        screen_tn += int((not gt_s) and (not pr_s))
        text_tp += int(gt_t and pr_t)
        text_fp += int((not gt_t) and pr_t)
        text_fn += int(gt_t and (not pr_t))
        text_tn += int((not gt_t) and (not pr_t))
        for target in gt_screen[image_id]:
            preds = pred_screen.get(image_id, [])
            hit = any(iou(target, p) >= 0.5 for p in preds) if preds else False
            if hit:
                screen_box_hit += 1
            else:
                screen_box_miss += 1
                if gt_s and not pr_s:
                    miss_ids.append(image_id)
    def pr(a, b):
        return a / (a + b) if a + b else 0.0

    screen_image_fn = screen_fn
    recall = pr(tp, fn)
    precision = pr(tp, fp)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "n": len(rows),
        "combined_precision": precision,
        "combined_recall": recall,
        "combined_f1": f1,
        "combined_oapr": 0.65 * recall + 0.25 * f1 + 0.10 * precision,
        "combined_fp": fp,
        "combined_fn": fn,
        "screen_image_precision": pr(screen_tp, screen_fp),
        "screen_image_recall": pr(screen_tp, screen_fn),
        "screen_image_fn": screen_image_fn,
        "screen_box_hit": screen_box_hit,
        "screen_box_miss": screen_box_miss,
        "text_image_precision": pr(text_tp, text_fp),
        "text_image_recall": pr(text_tp, text_fn),
        "text_image_fp": text_fp,
        "text_image_fn": text_fn,
        "unique_screen_presence_miss_images": len(set(miss_ids)),
    }


@dataclass
class FixConfig:
    fix_id: int
    name: str
    description: str
    # Detection transforms
    use_text_screen_hypothesis: bool = False
    hyp_min_count: int = 6
    hyp_require_lower_half: bool = True
    hyp_min_area_frac: float = 0.008
    hyp_margin: float = 0.12
    hyp_link_frac: float = 0.07
    hyp_only_if_no_yolo_screen: bool = True
    strip_text_in_screens: bool = True
    text_presence_min_count: int = 1
    text_presence_min_area_frac: float = 0.0
    force_screen_if_hyp: bool = True
    # Optional YOLO re-pass keys applied before hypothesis
    extra_screen_source: str | None = None


@dataclass
class ImageState:
    width: int
    height: int
    base_text: list[Box]
    base_screen: list[Box]
    extra_screens: dict[str, list[Box]] = field(default_factory=dict)


def apply_fix(
    state: ImageState,
    config: FixConfig,
) -> tuple[list[Box], list[Box], str]:
    screens = list(state.base_screen)
    if config.extra_screen_source:
        screens = union_boxes(screens, state.extra_screens.get(config.extra_screen_source, []))

    text = list(state.base_text)
    # Hypothesis from raw base text (before strip) so UI-on-screen is visible
    hyps: list[Box] = []
    if config.use_text_screen_hypothesis:
        allow = (not screens) if config.hyp_only_if_no_yolo_screen else True
        if allow:
            hyps = hypothesize_screens_from_text(
                text,
                width=state.width,
                height=state.height,
                min_count=config.hyp_min_count,
                require_lower_half=config.hyp_require_lower_half,
                min_cluster_area_frac=config.hyp_min_area_frac,
                margin_ratio=config.hyp_margin,
                link_frac=config.hyp_link_frac,
            )
            screens = union_boxes(screens, hyps)

    if config.strip_text_in_screens:
        text = strip_text_in_screens(text, screens)

    # Presence gates for routing (boxes retained for redaction may still be filtered lightly)
    text_area = sum(area(b) for b in text) / float(state.width * state.height)
    text_present = (
        len(text) >= config.text_presence_min_count
        and text_area >= config.text_presence_min_area_frac
    )
    # If gate fails, drop text for routing AND redaction of text channel
    if not text_present:
        text = []

    screen_present = bool(screens)
    if config.force_screen_if_hyp and hyps:
        screen_present = True
        screens = screens or hyps

    state_name = risk_state(bool(text), screen_present)
    return text, screens, state_name


def load_frame() -> tuple[pd.DataFrame, dict[str, list[Box]], dict[str, list[Box]], dict[str, ImageState]]:
    ann = pd.read_csv(ANNOTATIONS, keep_default_na=False)
    pred = pd.read_csv(BASE_PRED, keep_default_na=False)
    frame = ann.merge(pred, on=["protocol_id", "image_id"], validate="one_to_one")
    gt_text = {r.image_id: parse_boxes(r.text_boxes_json) for r in frame.itertuples()}
    gt_screen = {r.image_id: parse_boxes(r.screen_boxes_json) for r in frame.itertuples()}
    states: dict[str, ImageState] = {}
    for r in frame.itertuples():
        with Image.open(RAW / r.image_id) as im:
            w, h = im.size
        states[r.image_id] = ImageState(
            width=w,
            height=h,
            base_text=parse_boxes(r.predicted_text_boxes_json),
            base_screen=parse_boxes(r.predicted_screen_boxes_json),
        )
    return frame, gt_text, gt_screen, states


def run_yolo_extra_passes(states: dict[str, ImageState], device: str = "0") -> None:
    """Lightweight extra screen passes used by later fixes."""
    configs = {
        "yolo11_1280_conf010": ("yolo11n.pt", 1280, 0.10),
        "yolo11_1280_conf015": ("yolo11n.pt", 1280, 0.15),
        "yolo11_960_conf010": ("yolo11n.pt", 960, 0.10),
    }
    for key, (model, size, conf) in configs.items():
        print(f"extra screen pass: {key}", flush=True)
        detector = ScreenDetector(
            model_path=str(ROOT / "data/models" / model),
            device=device,
            confidence_threshold=conf,
            iou_threshold=0.70,
            image_size=size,
            half_precision=True,
        )
        for index, (image_id, state) in enumerate(states.items(), 1):
            with Image.open(RAW / image_id) as image:
                result = detector.detect(image.convert("RGB"))
            boxes = [tuple(map(int, item.box)) for item in result.detections]
            state.extra_screens[key] = boxes
            if index % 50 == 0:
                print(f"  {key}: {index}/{len(states)}", flush=True)
        del detector

    # Bottom-half TTA at 1280 conf 0.10
    print("extra screen pass: bottom_half_1280_conf010", flush=True)
    detector = ScreenDetector(
        model_path=str(ROOT / "data/models" / "yolo11n.pt"),
        device=device,
        confidence_threshold=0.10,
        iou_threshold=0.70,
        image_size=1280,
        half_precision=True,
    )
    for index, (image_id, state) in enumerate(states.items(), 1):
        with Image.open(RAW / image_id) as image:
            rgb = image.convert("RGB")
            w, h = rgb.size
            y0 = int(0.40 * h)
            crop = rgb.crop((0, y0, w, h))
            result = detector.detect(crop)
            mapped: list[Box] = []
            for item in result.detections:
                x1, y1, x2, y2 = map(int, item.box)
                mapped_box = clamp_box((x1, y1 + y0, x2, y2 + y0), w, h)
                if mapped_box is not None:
                    mapped.append(mapped_box)
            state.extra_screens["bottom_half_1280_conf010"] = mapped
        if index % 50 == 0:
            print(f"  bottom_half: {index}/{len(states)}", flush=True)
    del detector


def evaluate_config(
    frame: pd.DataFrame,
    gt_text: dict[str, list[Box]],
    gt_screen: dict[str, list[Box]],
    states: dict[str, ImageState],
    config: FixConfig,
) -> tuple[dict[str, float | int], dict[str, float | int], pd.DataFrame]:
    pred_text: dict[str, list[Box]] = {}
    pred_screen: dict[str, list[Box]] = {}
    rows = []
    for r in frame.itertuples():
        text, screens, state_name = apply_fix(states[r.image_id], config)
        pred_text[r.image_id] = text
        pred_screen[r.image_id] = screens
        rows.append(
            {
                "protocol_id": r.protocol_id,
                "image_id": r.image_id,
                "evaluation_split": r.evaluation_split,
                "predicted_risk_state": state_name,
                "predicted_text_count": len(text),
                "predicted_screen_count": len(screens),
                "predicted_text_boxes_json": box_json(text),
                "predicted_screen_boxes_json": box_json(screens),
                "ground_truth_text_present": bool(gt_text[r.image_id]),
                "ground_truth_screen_present": bool(gt_screen[r.image_id]),
                "predicted_text_present": bool(text),
                "predicted_screen_present": bool(screens),
            }
        )
    pred_df = pd.DataFrame(rows)
    dev = frame[frame.evaluation_split.eq("development")]
    test = frame[frame.evaluation_split.eq("test")]
    dev_m = image_metrics(dev, gt_text, gt_screen, pred_text, pred_screen)
    test_m = image_metrics(test, gt_text, gt_screen, pred_text, pred_screen)
    return dev_m, test_m, pred_df


def build_campaign() -> list[FixConfig]:
    """At most 10 fix configs, ordered by causal priority."""
    baseline = FixConfig(
        fix_id=0,
        name="baseline_locked",
        description="Locked CRAFT4k + YOLO11 union predictions (current canonical).",
        use_text_screen_hypothesis=False,
        strip_text_in_screens=True,
        text_presence_min_count=1,
    )
    fixes = [
        baseline,
        FixConfig(
            fix_id=1,
            name="S_text_cluster_screen_hyp_strict",
            description="If no YOLO screen: dense lower-half text cluster (≥8) → hypothesized screen; strip text inside; force screen route.",
            use_text_screen_hypothesis=True,
            hyp_min_count=8,
            hyp_require_lower_half=True,
            hyp_min_area_frac=0.010,
            hyp_margin=0.15,
            hyp_only_if_no_yolo_screen=True,
        ),
        FixConfig(
            fix_id=2,
            name="S_text_cluster_screen_hyp_recall",
            description="Looser hypothesis: ≥5 text cluster, lower-half optional if cluster dense, only if no YOLO screen.",
            use_text_screen_hypothesis=True,
            hyp_min_count=5,
            hyp_require_lower_half=False,
            hyp_min_area_frac=0.006,
            hyp_margin=0.18,
            hyp_link_frac=0.09,
            hyp_only_if_no_yolo_screen=True,
        ),
        FixConfig(
            fix_id=3,
            name="S_hyp_always_dense_cluster",
            description="Add text-cluster hypotheses even when YOLO already has screens (union).",
            use_text_screen_hypothesis=True,
            hyp_min_count=6,
            hyp_require_lower_half=True,
            hyp_min_area_frac=0.008,
            hyp_margin=0.12,
            hyp_only_if_no_yolo_screen=False,
        ),
        FixConfig(
            fix_id=4,
            name="S_yolo_conf010_union",
            description="Union base screens with YOLO11-1280 conf0.10 pass.",
            extra_screen_source="yolo11_1280_conf010",
            use_text_screen_hypothesis=False,
        ),
        FixConfig(
            fix_id=5,
            name="S_bottom_half_tta",
            description="Union base screens with bottom-half crop YOLO11-1280 conf0.10.",
            extra_screen_source="bottom_half_1280_conf010",
            use_text_screen_hypothesis=False,
        ),
        FixConfig(
            fix_id=6,
            name="S_yolo010_plus_hyp",
            description="YOLO conf0.10 union + strict text-cluster hyp when still empty.",
            extra_screen_source="yolo11_1280_conf010",
            use_text_screen_hypothesis=True,
            hyp_min_count=6,
            hyp_require_lower_half=True,
            hyp_min_area_frac=0.008,
            hyp_only_if_no_yolo_screen=True,
        ),
        FixConfig(
            fix_id=7,
            name="S_bottom_tta_plus_hyp",
            description="Bottom-half TTA union + text-cluster hyp when still empty.",
            extra_screen_source="bottom_half_1280_conf010",
            use_text_screen_hypothesis=True,
            hyp_min_count=6,
            hyp_require_lower_half=True,
            hyp_min_area_frac=0.008,
            hyp_only_if_no_yolo_screen=True,
        ),
        FixConfig(
            fix_id=8,
            name="S_full_stack_screen",
            description="Base ∪ conf010 ∪ bottom TTA ∪ hyp-if-empty (screen stack).",
            extra_screen_source="stack_conf010_bottom",  # special handled below
            use_text_screen_hypothesis=True,
            hyp_min_count=5,
            hyp_require_lower_half=True,
            hyp_min_area_frac=0.006,
            hyp_only_if_no_yolo_screen=True,
        ),
        FixConfig(
            fix_id=9,
            name="T_text_presence_gate_count2",
            description="Text presence requires ≥2 boxes (routing/redaction text channel); keep base screens.",
            text_presence_min_count=2,
            use_text_screen_hypothesis=False,
        ),
        FixConfig(
            fix_id=10,
            name="ST_best_screen_stack_plus_text_gate",
            description="Full screen stack (fix 8 sources) + text presence min count 2.",
            extra_screen_source="stack_conf010_bottom",
            use_text_screen_hypothesis=True,
            hyp_min_count=5,
            hyp_require_lower_half=True,
            hyp_min_area_frac=0.006,
            hyp_only_if_no_yolo_screen=True,
            text_presence_min_count=2,
        ),
    ]
    # Limit the sweep to ten enhancements plus the baseline.
    return fixes[:11]


def materialize_stack_sources(states: dict[str, ImageState]) -> None:
    for state in states.values():
        stacked = union_boxes(
            state.extra_screens.get("yolo11_1280_conf010", []),
            state.extra_screens.get("bottom_half_1280_conf010", []),
        )
        state.extra_screens["stack_conf010_bottom"] = stacked


def score_for_selection(dev_m: dict[str, float | int]) -> float:
    """Development selection score: prioritise screen-miss reduction then combined OAPR."""
    # Lower screen FN is better; reward combined OAPR; penalise text FP lightly.
    return (
        float(dev_m["combined_oapr"])
        - 0.04 * float(dev_m["screen_image_fn"])
        - 0.005 * float(dev_m["text_image_fp"])
        + 0.01 * float(dev_m["screen_image_recall"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-yolo-extra", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame, gt_text, gt_screen, states = load_frame()
    if not args.skip_yolo_extra:
        run_yolo_extra_passes(states, device=args.device)
        materialize_stack_sources(states)
        # Persist extras for reuse
        extra_rows = []
        for image_id, state in states.items():
            extra_rows.append(
                {
                    "image_id": image_id,
                    **{
                        f"{key}_json": box_json(val)
                        for key, val in state.extra_screens.items()
                    },
                }
            )
        pd.DataFrame(extra_rows).to_csv(OUT_DIR / "extra_screen_passes.csv", index=False)
    else:
        extra_path = OUT_DIR / "extra_screen_passes.csv"
        if extra_path.exists():
            extras = pd.read_csv(extra_path, keep_default_na=False)
            for r in extras.itertuples():
                st = states[r.image_id]
                for col in extras.columns:
                    if col.endswith("_json"):
                        key = col[: -len("_json")]
                        st.extra_screens[key] = parse_boxes(getattr(r, col))
            materialize_stack_sources(states)

    campaign = build_campaign()
    summary_rows = []
    best_pred: pd.DataFrame | None = None
    best_name = "baseline_locked"
    best_sel = -1e9
    baseline_test: dict[str, float | int] | None = None

    for config in campaign:
        # Special-case stack source already materialized
        print(f"\n=== Fix {config.fix_id}: {config.name} ===", flush=True)
        print(config.description, flush=True)
        dev_m, test_m, pred_df = evaluate_config(frame, gt_text, gt_screen, states, config)
        sel = score_for_selection(dev_m)
        if config.fix_id == 0:
            baseline_test = test_m
        summary_rows.append(
            {
                "fix_id": config.fix_id,
                "name": config.name,
                "description": config.description,
                "selection_score_dev": sel,
                **{f"dev_{k}": v for k, v in dev_m.items()},
                **{f"test_{k}": v for k, v in test_m.items()},
            }
        )
        pred_df.to_csv(OUT_DIR / f"pred_{config.fix_id:02d}_{config.name}.csv", index=False)
        print(
            f"DEV  screen_fn={dev_m['screen_image_fn']} combined_oapr={dev_m['combined_oapr']:.4f} "
            f"text_fp={dev_m['text_image_fp']} sel={sel:.4f}",
            flush=True,
        )
        print(
            f"TEST screen_fn={test_m['screen_image_fn']} combined_oapr={test_m['combined_oapr']:.4f} "
            f"text_fp={test_m['text_image_fp']} text_fn={test_m['text_image_fn']}",
            flush=True,
        )
        # Prefer configs that improve development selection score; ties keep lower fix_id.
        if sel > best_sel + 1e-12:
            best_sel = sel
            best_name = config.name
            best_pred = pred_df

    summary = pd.DataFrame(summary_rows).sort_values("fix_id")
    summary.to_csv(OUT_DIR / "01_campaign_summary.csv", index=False)

    assert baseline_test is not None and best_pred is not None
    best_row = summary[summary.name.eq(best_name)].iloc[0]
    # Promotion gate on HELD-OUT relative to baseline
    promote = False
    reasons = []
    base_fn = int(baseline_test["screen_image_fn"])
    best_fn = int(best_row["test_screen_image_fn"])
    base_oapr = float(baseline_test["combined_oapr"])
    best_oapr = float(best_row["test_combined_oapr"])
    base_tfp = int(baseline_test["text_image_fp"])
    best_tfp = int(best_row["test_text_image_fp"])
    base_tfn = int(baseline_test["text_image_fn"])
    best_tfn = int(best_row["test_text_image_fn"])

    if best_fn < base_fn:
        reasons.append(f"screen presence FN {base_fn}→{best_fn}")
    if best_oapr + 1e-9 >= base_oapr - 0.01 and best_oapr >= 0.90:
        reasons.append(f"combined OAPR held ({best_oapr:.4f})")
    if best_tfn <= base_tfn + 1:
        reasons.append(f"text FN not regressed badly ({base_tfn}→{best_tfn})")
    # Strict promotion: must reduce screen FN and not collapse combined recall/OAPR
    promote = (
        best_fn <= max(0, base_fn - 1)
        and best_oapr >= base_oapr - 0.015
        and float(best_row["test_combined_recall"]) >= 0.96
        and best_tfn <= base_tfn + 2
        and best_name != "baseline_locked"
    )

    report = [
        "# Detection Fix Campaign Report",
        "",
        f"Configs evaluated: {len(campaign)} (baseline + up to 10 fixes).",
        f"Development-selected best by selection score: **{best_name}**.",
        "",
        "## Baseline held-out",
        f"- screen image FN: {base_fn}",
        f"- combined OAPR: {base_oapr:.4f}",
        f"- combined recall: {float(baseline_test['combined_recall']):.4f}",
        f"- text image FP/FN: {base_tfp}/{base_tfn}",
        "",
        "## Best config held-out",
        f"- name: {best_name}",
        f"- screen image FN: {best_fn}",
        f"- combined OAPR: {best_oapr:.4f}",
        f"- combined recall: {float(best_row['test_combined_recall']):.4f}",
        f"- text image FP/FN: {best_tfp}/{best_tfn}",
        "",
        f"## Promotion decision: {'PROMOTE_CANDIDATE' if promote else 'DO_NOT_PROMOTE'}",
        "",
        "Promotion requires held-out screen FN reduction (≥1), combined recall ≥ 0.96,",
        "combined OAPR within 0.015 of baseline, and limited text FN regression.",
        "",
        "Notes: " + ("; ".join(reasons) if reasons else "no material held-out gains."),
        "",
        "See `01_campaign_summary.csv` for all fixes.",
    ]
    (OUT_DIR / "02_campaign_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    best_pred.to_csv(OUT_DIR / "03_best_predictions.csv", index=False)
    print("\n".join(report), flush=True)
    print(f"\nWrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
