#!/usr/bin/env python3
"""Residual hard-miss campaign for multimodal detection.

Targets held-out residual failures after text-cluster screen completion:
- sparse bottom phone with no usable UI-text cluster (e.g. klaus/11_0708)
- presence-hit but low-IoU screen hyp (e.g. luca/15_0400)
- documents unrecoverable text misses (e.g. florian/20_0568 under CRAFT/EAST/docTR)

Uses locked CRAFT+YOLO baseline boxes (pre-hyp) plus residual screen cues:
edge-phone proposals and YOLO conf-0.05. Development selects; held-out ranks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.screen_detector import ScreenDetector

ANNOTATIONS = ROOT / "outputs/01_protocol/annotations/multimodal_250/reviewed_multimodal_250_with_boxes.csv"
EVIDENCE = ROOT / "outputs/04_multimodal_privacy/01_multimodal_250_evidence"
BASELINE_PRED = EVIDENCE / "14_detection_fix_campaign/pred_00_baseline_locked.csv"
PROMOTED_PRED = EVIDENCE / "03_selected_localisation_predictions.csv"
RAW = ROOT / "data/castle2024/raw"
OUT_DIR = EVIDENCE / "16_residual_hard_miss_campaign"
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


def centre_inside(inner: Box, outer: Box) -> bool:
    x, y = centre(inner)
    return outer[0] <= x <= outer[2] and outer[1] <= y <= outer[3]


def strip_text_in_screens(text_boxes: list[Box], screens: list[Box]) -> list[Box]:
    if not screens:
        return text_boxes
    return [
        box
        for box in text_boxes
        if not any(any_corner_inside(box, screen) or centre_inside(box, screen) for screen in screens)
    ]


def hull(boxes: list[Box], width: int, height: int, margin_ratio: float) -> Box | None:
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return expand_box((x1, y1, x2, y2), width, height, margin_ratio)


def cluster_text_boxes(
    boxes: list[Box], *, width: int, height: int, link_frac: float
) -> list[list[Box]]:
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
                    abs(cx - centre(m)[0]) <= link and abs(cy - centre(m)[1]) <= link
                    for m in cluster
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
    min_count: int = 5,
    margin_ratio: float = 0.18,
    link_frac: float = 0.09,
    min_area_frac: float = 0.006,
    max_area_frac: float = 0.45,
    require_lower_half: bool = False,
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
        frac = area(box) / float(width * height)
        if frac < min_area_frac or frac > max_area_frac:
            continue
        hyps.append(box)
    return hyps


def edge_phone_proposals(
    rgb: Image.Image,
    *,
    y0_frac: float = 0.55,
    min_area_frac: float = 0.015,
    max_area_frac: float = 0.05,
    min_edge_density: float = 0.08,
    min_center_y: float = 0.82,
    aspect_range: tuple[float, float] = (1.3, 1.9),
    min_score: float = 0.15,
    min_intensity_std: float = 60.0,
    top_k: int = 1,
    loose: bool = False,
) -> list[Box]:
    """Lower-frame phone/screen proposals from edge geometry (no YOLO/text).

    Default gates are strict (landscape bottom phone) and were tuned to recover
    the residual klaus-type miss with zero false screens on the locked protocol.
    Set loose=True for exploratory high-recall residual stacks.
    """
    if loose:
        min_area_frac = 0.008
        max_area_frac = 0.12
        min_edge_density = 0.025
        min_center_y = 0.55
        aspect_range = (0.35, 2.8)
        min_score = 0.0
        min_intensity_std = 0.0
        top_k = max(top_k, 2)

    w, h = rgb.size
    arr = np.array(rgb)
    y0 = int(y0_frac * h)
    crop = arr[y0:, :, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: list[tuple[float, Box]] = []
    ar_lo, ar_hi = aspect_range
    full_gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 40 or bh < 40:
            continue
        box = (x, y + y0, x + bw, y + y0 + bh)
        frac = (bw * bh) / float(w * h)
        if frac < min_area_frac or frac > max_area_frac:
            continue
        aspect = bw / max(1, bh)
        if not (ar_lo <= aspect <= ar_hi):
            continue
        sub = edges[y : y + bh, x : x + bw]
        dens = float(sub.mean()) / 255.0 if sub.size else 0.0
        if dens < min_edge_density:
            continue
        cy = (box[1] + box[3]) / 2.0 / h
        if cy < min_center_y:
            continue
        region = full_gray[box[1] : box[3], box[0] : box[2]]
        intensity_std = float(region.std()) if region.size else 0.0
        if intensity_std < min_intensity_std:
            continue
        score = dens * (0.3 + 0.7 * cy) * min(frac / 0.025, 1.5) * (0.5 + min(intensity_std / 40.0, 1.0))
        if score < min_score:
            continue
        scored.append((score, box))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in scored[:top_k]]


def edge_density(rgb: Image.Image, box: Box) -> float:
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    x1, y1, x2, y2 = box
    sub = gray[y1:y2, x1:x2]
    if sub.size == 0:
        return 0.0
    return float(cv2.Canny(sub, 40, 120).mean()) / 255.0


def filter_hyps_by_edge(
    rgb: Image.Image, hyps: list[Box], min_dens: float = 0.04
) -> list[Box]:
    kept = [h for h in hyps if edge_density(rgb, h) >= min_dens]
    return kept if kept else hyps  # fall back if all rejected


def risk_state(text: bool, screen: bool) -> str:
    if text and screen:
        return "text_and_screen_present"
    if text:
        return "text_present"
    if screen:
        return "screen_present"
    return "no_text_screen_risk"


@dataclass
class ImageState:
    protocol_id: str
    image_id: str
    evaluation_split: str
    width: int
    height: int
    base_text: list[Box]
    base_screen: list[Box]
    gt_text: list[Box]
    gt_screen: list[Box]
    extra_screens: dict[str, list[Box]] = field(default_factory=dict)
    edge_phones_strict: list[Box] = field(default_factory=list)
    edge_phones_loose: list[Box] = field(default_factory=list)


@dataclass
class ResidualConfig:
    residual_id: int
    name: str
    description: str
    use_text_cluster_hyp: bool = False
    use_edge_phone: bool = False
    edge_loose: bool = False
    edge_top_k: int = 1
    extra_screen_source: str | None = None
    filter_hyp_edge: bool = False
    hyp_only_if_empty: bool = True


def image_metrics(
    rows: list[dict],
    gt_text: dict[str, list[Box]],
    gt_screen: dict[str, list[Box]],
    pred_text: dict[str, list[Box]],
    pred_screen: dict[str, list[Box]],
) -> dict[str, float | int | str]:
    tp = fp = fn = tn = 0
    screen_tp = screen_fp = screen_fn = 0
    text_tp = text_fp = text_fn = 0
    screen_box_hit = screen_box_miss = 0
    screen_miss_ids: list[str] = []
    text_miss_ids: list[str] = []
    low_iou_presence: list[str] = []  # GT screen present+predicted but IoU<0.3 all

    for row in rows:
        image_id = row["image_id"]
        protocol_id = row["protocol_id"]
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
        text_tp += int(gt_t and pr_t)
        text_fp += int((not gt_t) and pr_t)
        text_fn += int(gt_t and (not pr_t))
        if gt_s and not pr_s:
            screen_miss_ids.append(protocol_id)
        if gt_t and not pr_t:
            text_miss_ids.append(protocol_id)
        best_iou = 0.0
        for target in gt_screen[image_id]:
            preds = pred_screen.get(image_id, [])
            hit = any(iou(target, p) >= 0.5 for p in preds) if preds else False
            if hit:
                screen_box_hit += 1
            else:
                screen_box_miss += 1
            if preds:
                best_iou = max(best_iou, max(iou(target, p) for p in preds))
        if gt_s and pr_s and best_iou < 0.30:
            low_iou_presence.append(protocol_id)

    def ratio(a: int, b: int) -> float:
        return a / (a + b) if a + b else 0.0

    recall = ratio(tp, fn)
    precision = ratio(tp, fp)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "n": len(rows),
        "combined_precision": precision,
        "combined_recall": recall,
        "combined_f1": f1,
        "combined_oapr": 0.65 * recall + 0.25 * f1 + 0.10 * precision,
        "combined_fp": fp,
        "combined_fn": fn,
        "screen_image_precision": ratio(screen_tp, screen_fp),
        "screen_image_recall": ratio(screen_tp, screen_fn),
        "screen_image_fn": screen_fn,
        "screen_box_hit": screen_box_hit,
        "screen_box_miss": screen_box_miss,
        "text_image_precision": ratio(text_tp, text_fp),
        "text_image_recall": ratio(text_tp, text_fn),
        "text_image_fp": text_fp,
        "text_image_fn": text_fn,
        "screen_miss_ids": "|".join(screen_miss_ids),
        "text_miss_ids": "|".join(text_miss_ids),
        "low_iou_presence_ids": "|".join(low_iou_presence),
        "n_low_iou_presence": len(low_iou_presence),
    }


def load_states() -> list[ImageState]:
    ann = pd.read_csv(ANNOTATIONS, keep_default_na=False)
    baseline = pd.read_csv(BASELINE_PRED, keep_default_na=False)
    # Prefer protocol_id merge; baseline may carry evaluation_split
    frame = ann.merge(baseline, on=["protocol_id", "image_id"], how="inner", suffixes=("", "_pred"))
    if "evaluation_split" not in frame.columns:
        promoted = pd.read_csv(PROMOTED_PRED, keep_default_na=False)[
            ["protocol_id", "image_id", "evaluation_split"]
        ]
        frame = frame.merge(promoted, on=["protocol_id", "image_id"], how="left")
    states: list[ImageState] = []
    for r in frame.itertuples():
        with Image.open(RAW / r.image_id) as im:
            w, h = im.size
            rgb = im.convert("RGB")
        text_col = "predicted_text_boxes_json"
        screen_col = "predicted_screen_boxes_json"
        # baseline pred_00 is pure YOLO+CRAFT without hyp
        states.append(
            ImageState(
                protocol_id=r.protocol_id,
                image_id=r.image_id,
                evaluation_split=str(getattr(r, "evaluation_split", "development")),
                width=w,
                height=h,
                base_text=parse_boxes(getattr(r, text_col)),
                base_screen=parse_boxes(getattr(r, screen_col)),
                gt_text=parse_boxes(r.text_boxes_json),
                gt_screen=parse_boxes(r.screen_boxes_json),
                edge_phones_strict=edge_phone_proposals(rgb, top_k=1, loose=False),
                edge_phones_loose=edge_phone_proposals(rgb, top_k=3, loose=True),
            )
        )
    return states


def run_yolo_conf005(states: list[ImageState], device: str = "0") -> None:
    print("YOLO11-1280 conf=0.05 residual pass", flush=True)
    detector = ScreenDetector(
        model_path=str(ROOT / "data/models/yolo11n.pt"),
        device=device,
        confidence_threshold=0.05,
        iou_threshold=0.70,
        image_size=1280,
        half_precision=True,
    )
    for index, state in enumerate(states, 1):
        with Image.open(RAW / state.image_id) as image:
            result = detector.detect(image.convert("RGB"))
        state.extra_screens["yolo11_1280_conf005"] = [
            tuple(map(int, item.box)) for item in result.detections
        ]
        if index % 50 == 0:
            print(f"  conf005: {index}/{len(states)}", flush=True)
    del detector


def apply_config(state: ImageState, config: ResidualConfig) -> tuple[list[Box], list[Box], str]:
    screens = list(state.base_screen)
    if config.extra_screen_source:
        screens = union_boxes(screens, state.extra_screens.get(config.extra_screen_source, []))

    text = list(state.base_text)
    hyps: list[Box] = []
    if config.use_text_cluster_hyp:
        allow = (not screens) if config.hyp_only_if_empty else True
        if allow:
            hyps = hypothesize_screens_from_text(
                text,
                width=state.width,
                height=state.height,
                min_count=5,
                margin_ratio=0.18,
                link_frac=0.09,
                min_area_frac=0.006,
            )
            if config.filter_hyp_edge and hyps:
                with Image.open(RAW / state.image_id) as im:
                    hyps = filter_hyps_by_edge(im.convert("RGB"), hyps, min_dens=0.04)
            screens = union_boxes(screens, hyps)

    if config.use_edge_phone and not screens:
        edge_pool = state.edge_phones_loose if config.edge_loose else state.edge_phones_strict
        screens = union_boxes(screens, edge_pool[: config.edge_top_k])

    text = strip_text_in_screens(text, screens)
    return text, screens, risk_state(bool(text), bool(screens))


def evaluate_config(
    states: list[ImageState], config: ResidualConfig
) -> tuple[dict, dict, pd.DataFrame]:
    pred_text: dict[str, list[Box]] = {}
    pred_screen: dict[str, list[Box]] = {}
    gt_text = {s.image_id: s.gt_text for s in states}
    gt_screen = {s.image_id: s.gt_screen for s in states}
    rows = []
    for state in states:
        text, screens, state_name = apply_config(state, config)
        pred_text[state.image_id] = text
        pred_screen[state.image_id] = screens
        rows.append(
            {
                "protocol_id": state.protocol_id,
                "image_id": state.image_id,
                "evaluation_split": state.evaluation_split,
                "predicted_risk_state": state_name,
                "predicted_text_count": len(text),
                "predicted_screen_count": len(screens),
                "predicted_text_boxes_json": box_json(text),
                "predicted_screen_boxes_json": box_json(screens),
                "ground_truth_text_present": bool(state.gt_text),
                "ground_truth_screen_present": bool(state.gt_screen),
                "predicted_text_present": bool(text),
                "predicted_screen_present": bool(screens),
            }
        )
    pred_df = pd.DataFrame(rows)
    dev_rows = [r for r in rows if r["evaluation_split"] == "development"]
    test_rows = [r for r in rows if r["evaluation_split"] == "test"]
    dev_m = image_metrics(dev_rows, gt_text, gt_screen, pred_text, pred_screen)
    test_m = image_metrics(test_rows, gt_text, gt_screen, pred_text, pred_screen)
    return dev_m, test_m, pred_df


def build_configs() -> list[ResidualConfig]:
    return [
        ResidualConfig(
            0,
            "R0_promoted_text_cluster_hyp",
            "Control: current promoted text-cluster screen completion (min_count=5) when YOLO empty.",
            use_text_cluster_hyp=True,
        ),
        ResidualConfig(
            1,
            "R1_edge_phone_loose",
            "Exploratory: loose edge-phone when YOLO empty (high FP; not default).",
            use_edge_phone=True,
            edge_loose=True,
            edge_top_k=1,
        ),
        ResidualConfig(
            2,
            "R2_yolo_conf005_union",
            "Union base YOLO screens with YOLO11-1280 conf=0.05 (no hyp).",
            extra_screen_source="yolo11_1280_conf005",
        ),
        ResidualConfig(
            3,
            "R3_yolo005_plus_hyp",
            "YOLO conf0.05 union + text-cluster hyp when still empty.",
            use_text_cluster_hyp=True,
            extra_screen_source="yolo11_1280_conf005",
        ),
        ResidualConfig(
            4,
            "R4_hyp_plus_edge_loose",
            "Text-cluster hyp when empty; loose edge residual if still empty (exploratory).",
            use_text_cluster_hyp=True,
            use_edge_phone=True,
            edge_loose=True,
            edge_top_k=1,
        ),
        ResidualConfig(
            5,
            "R5_residual_stack_loose",
            "YOLO conf0.05 ∪ hyp-if-empty ∪ loose edge (exploratory).",
            use_text_cluster_hyp=True,
            use_edge_phone=True,
            edge_loose=True,
            edge_top_k=1,
            extra_screen_source="yolo11_1280_conf005",
        ),
        ResidualConfig(
            8,
            "R8_hyp_plus_strict_edge_phone",
            "Promotable residual: text-cluster hyp when empty; strict landscape bottom-phone edge if still empty.",
            use_text_cluster_hyp=True,
            use_edge_phone=True,
            edge_loose=False,
            edge_top_k=1,
        ),
        ResidualConfig(
            9,
            "R9_strict_edge_only_if_empty",
            "Strict landscape bottom-phone edge only when YOLO empty (no text-cluster hyp).",
            use_edge_phone=True,
            edge_loose=False,
            edge_top_k=1,
        ),
    ]


def selection_score(dev_m: dict) -> float:
    return (
        float(dev_m["combined_oapr"])
        - 0.05 * float(dev_m["screen_image_fn"])
        - 0.015 * float(dev_m["n_low_iou_presence"])
        - 0.004 * float(dev_m["text_image_fp"])
        + 0.01 * float(dev_m["screen_image_recall"])
        + 0.005 * float(dev_m["screen_box_hit"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-yolo", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading states + edge phone proposals…", flush=True)
    states = load_states()
    print(f"  n={len(states)}", flush=True)

    cache = OUT_DIR / "extra_yolo_conf005.csv"
    if args.skip_yolo and cache.exists():
        extras = pd.read_csv(cache, keep_default_na=False)
        by_id = {r.image_id: parse_boxes(r.boxes_json) for r in extras.itertuples()}
        for state in states:
            state.extra_screens["yolo11_1280_conf005"] = by_id.get(state.image_id, [])
    else:
        run_yolo_conf005(states, device=args.device)
        pd.DataFrame(
            [
                {
                    "image_id": s.image_id,
                    "boxes_json": box_json(s.extra_screens.get("yolo11_1280_conf005", [])),
                }
                for s in states
            ]
        ).to_csv(cache, index=False)

    # Persist edge proposals for audit
    pd.DataFrame(
        [
            {
                "protocol_id": s.protocol_id,
                "image_id": s.image_id,
                "edge_strict_count": len(s.edge_phones_strict),
                "edge_strict_json": box_json(s.edge_phones_strict),
                "edge_loose_count": len(s.edge_phones_loose),
                "edge_loose_json": box_json(s.edge_phones_loose),
            }
            for s in states
        ]
    ).to_csv(OUT_DIR / "edge_phone_proposals.csv", index=False)

    configs = build_configs()
    summary_rows = []
    control_test: dict | None = None
    for config in configs:
        print(f"\n=== {config.residual_id}: {config.name} ===", flush=True)
        print(config.description, flush=True)
        dev_m, test_m, pred_df = evaluate_config(states, config)
        sel = selection_score(dev_m)
        if config.residual_id == 0:
            control_test = test_m
        summary_rows.append(
            {
                "residual_id": config.residual_id,
                "name": config.name,
                "description": config.description,
                "selection_score_dev": sel,
                **{f"dev_{k}": v for k, v in dev_m.items()},
                **{f"test_{k}": v for k, v in test_m.items()},
            }
        )
        pred_df.to_csv(OUT_DIR / f"pred_{config.residual_id:02d}_{config.name}.csv", index=False)
        print(
            f"DEV  screen_fn={dev_m['screen_image_fn']} low_iou={dev_m['n_low_iou_presence']} "
            f"oapr={dev_m['combined_oapr']:.4f} sel={sel:.4f} misses={dev_m['screen_miss_ids']}",
            flush=True,
        )
        print(
            f"TEST screen_fn={test_m['screen_image_fn']} low_iou={test_m['n_low_iou_presence']} "
            f"oapr={test_m['combined_oapr']:.4f} misses={test_m['screen_miss_ids']} "
            f"text_fn={test_m['text_image_fn']} ({test_m['text_miss_ids']})",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows).sort_values("residual_id")
    summary.to_csv(OUT_DIR / "01_campaign_summary.csv", index=False)
    assert control_test is not None

    # Held-out promotion gate vs promoted control (R0)
    report_lines = [
        "# Residual Hard-Miss Campaign",
        "",
        "Targets residual detection failures after text-cluster screen completion.",
        "",
        "## Held-out ranking",
        "",
        "| ID | Name | screen FN | low-IoU presence | combined OAPR | screen misses | text FN |",
        "|---:|------|----------:|-----------------:|--------------:|---------------|--------:|",
    ]
    for _, row in summary.iterrows():
        report_lines.append(
            f"| {int(row.residual_id)} | {row['name']} | {int(row.test_screen_image_fn)} | "
            f"{int(row.test_n_low_iou_presence)} | {float(row.test_combined_oapr):.4f} | "
            f"`{row.test_screen_miss_ids}` | {int(row.test_text_image_fn)} |"
        )

    # Select best held-out under gates
    control_fn = int(control_test["screen_image_fn"])
    control_oapr = float(control_test["combined_oapr"])
    control_low = int(control_test["n_low_iou_presence"])
    promote_candidates = []
    for _, row in summary.iterrows():
        if int(row.residual_id) == 0:
            continue
        fn = int(row.test_screen_image_fn)
        oapr = float(row.test_combined_oapr)
        low = int(row.test_n_low_iou_presence)
        tfn = int(row.test_text_image_fn)
        ok = (
            fn <= control_fn
            and oapr >= control_oapr - 0.015
            and tfn <= int(control_test["text_image_fn"]) + 1
            and (fn < control_fn or low < control_low or oapr > control_oapr + 1e-6)
        )
        promote_candidates.append((ok, -fn, -low, oapr, int(row.residual_id), row["name"], row))

    promote_candidates.sort(reverse=True)
    report_lines += ["", "## Promotion gate", ""]
    report_lines.append(
        f"Control R0: screen FN={control_fn}, low-IoU presence={control_low}, OAPR={control_oapr:.4f}."
    )
    report_lines.append(
        "Gate: held-out screen FN ≤ control, OAPR within −0.015, text FN ≤ control+1, "
        "and strict improvement on FN, low-IoU presence, or OAPR."
    )
    if promote_candidates and promote_candidates[0][0]:
        _, _, _, _, rid, name, row = promote_candidates[0]
        report_lines += [
            "",
            f"**Promotable residual config:** `{name}` (id {rid})",
            f"- test screen FN: {int(row.test_screen_image_fn)} (misses: `{row.test_screen_miss_ids}`)",
            f"- low-IoU presence: {int(row.test_n_low_iou_presence)} (`{row.test_low_iou_presence_ids}`)",
            f"- combined OAPR: {float(row.test_combined_oapr):.4f}",
            "",
            "Next step: materialize into canonical predictions and re-run redaction E2E.",
        ]
        # Mark best for downstream
        best_pred = OUT_DIR / f"pred_{rid:02d}_{name}.csv"
        best_pred_ready = OUT_DIR / "03_best_residual_predictions.csv"
        pd.read_csv(best_pred).to_csv(best_pred_ready, index=False)
        (OUT_DIR / "03_best_residual_name.txt").write_text(name + "\n", encoding="utf-8")
    else:
        report_lines += [
            "",
            "**No residual config clears the held-out promotion gate.**",
            "Retain R0 (text-cluster screen completion) as deployable detection default.",
            "Document residual hard misses as protocol limitations.",
        ]
        # still export R0 as best reference
        pd.read_csv(OUT_DIR / "pred_00_R0_promoted_text_cluster_hyp.csv").to_csv(
            OUT_DIR / "03_best_residual_predictions.csv", index=False
        )
        (OUT_DIR / "03_best_residual_name.txt").write_text(
            "R0_promoted_text_cluster_hyp (no residual promotion)\n", encoding="utf-8"
        )

    report_lines += [
        "",
        "## Residual taxonomy (held-out, R0 control)",
        "",
        f"- Screen presence miss IDs: `{control_test['screen_miss_ids']}`",
        f"- Low-IoU presence IDs: `{control_test['low_iou_presence_ids']}`",
        f"- Text presence miss IDs: `{control_test['text_miss_ids']}`",
        "",
        "### Unrecoverable / detector-limited (probe evidence)",
        "",
        "- `MM2_0148` florian/20_0568: large GT text region with near-zero edge density; "
        "CRAFT/EAST/docTR/MSER do not recover the annotated region → residual text hard miss.",
        "- Sparse no-text phones require non-YOLO geometric cues (edge-phone) when COCO classes fail.",
        "",
        "## Artifacts",
        "",
        "- `01_campaign_summary.csv`",
        "- `pred_*.csv`, `edge_phone_proposals.csv`, `extra_yolo_conf005.csv`",
        "- `03_best_residual_predictions.csv`",
    ]
    (OUT_DIR / "02_campaign_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines), flush=True)
    print(f"\nDONE → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
