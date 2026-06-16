"""Guard: decision-framework must not select methods on the held-out test split."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/oapr_routing/run_decision_framework_evaluation.py"


def _load_df_module():
    spec = importlib.util.spec_from_file_location("run_decision_framework_evaluation", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_select_variant_id_rejects_test_split_selection() -> None:
    mod = _load_df_module()
    table = pd.DataFrame(
        [
            {
                "modality": "screen",
                "variant": "a",
                "split": "test",
                "oapr_multimodal_score": 0.99,
                "recall": 0.99,
            },
            {
                "modality": "screen",
                "variant": "b",
                "split": "development",
                "oapr_multimodal_score": 0.50,
                "recall": 0.50,
            },
        ]
    )
    with pytest.raises(ValueError, match="Held-out selection is forbidden"):
        mod.select_variant_id(
            table,
            modality="screen",
            locked_variant="missing",
            selection_split="test",
            metric="oapr_multimodal_score",
        )


def test_select_variant_id_uses_locked_development_id_not_test_top() -> None:
    mod = _load_df_module()
    table = pd.DataFrame(
        [
            {
                "modality": "screen",
                "variant": "test_top_but_not_selected",
                "split": "test",
                "oapr_multimodal_score": 0.99,
                "recall": 0.99,
                "strict_iou50_recall": 0.99,
            },
            {
                "modality": "screen",
                "variant": "yolo11n_coco_640_1280_union",
                "split": "development",
                "oapr_multimodal_score": 0.60,
                "recall": 0.67,
                "strict_iou50_recall": 0.67,
            },
            {
                "modality": "screen",
                "variant": "yolo11n_coco_640_1280_union",
                "split": "test",
                "oapr_multimodal_score": 0.59,
                "recall": 0.64,
                "strict_iou50_recall": 0.64,
            },
        ]
    )
    chosen = mod.select_variant_id(
        table,
        modality="screen",
        locked_variant="yolo11n_coco_640_1280_union",
        selection_split="development",
    )
    assert chosen == "yolo11n_coco_640_1280_union"


def test_stage_multimodal_detection_locks_protocol_variants_and_reports_test_metrics() -> None:
    mod = _load_df_module()
    summary = mod.stage_multimodal_detection()
    assert not summary.empty
    row = summary.iloc[0]
    assert row["selection_split"] == "development"
    assert row["split"] == "test"
    assert row["screen_variant"] == mod.LOCKED_MM_SCREEN_VARIANT
    assert row["text_variant"] == mod.LOCKED_MM_TEXT_VARIANT
    assert row["combined_variant"] == mod.LOCKED_MM_COMBINED_VARIANT
    # Exclude the held-out-selected variant.
    assert row["screen_variant"] != "yolo11n_coco_1280_conf010"
    # Held-out metrics for the locked screen variant must match the comparison table.
    det = pd.read_csv(
        ROOT
        / "outputs/04_multimodal_privacy/01_multimodal_250_evidence/02_detection_method_comparison.csv"
    )
    expected = det[
        det.modality.eq("screen")
        & det.variant.eq(mod.LOCKED_MM_SCREEN_VARIANT)
        & det.split.eq("test")
    ].iloc[0]
    assert abs(float(row["screen_iou50_recall"]) - float(expected["strict_iou50_recall"])) < 1e-9


def test_source_has_no_test_sort_selection_anti_pattern() -> None:
    """Fail if the runner reintroduces selecting methods by sorting split==test rows."""
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    bad_snippets: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.name != "stage_multimodal_detection":
                self.generic_visit(node)
                return
            body_src = ast.get_source_segment(source, node) or ""
            # Forbidden: filter to test then sort for best variant selection.
            if 'split"].eq("test")' in body_src or "split'].eq('test')" in body_src:
                if "sort_values" in body_src and (
                    "screen_best" in body_src
                    or "text_best" in body_src
                    or "iloc[0]" in body_src
                    and "sort_values" in body_src
                ):
                    # Allow atomic comparison over test rows, but not best-of-test selection.
                    if "screen_best" in body_src or "text_best" in body_src:
                        bad_snippets.append(node.name)
            # Reject held-out selection variables.
            if "screen_best" in body_src or "text_best" in body_src:
                bad_snippets.append(f"{node.name}: uses screen_best/text_best")
            self.generic_visit(node)

    Visitor().visit(tree)
    assert not bad_snippets, f"Held-out selection anti-pattern in {bad_snippets}"
    # Positive requirement: selection helper must reject held-out splits.
    assert "Held-out selection is forbidden" in source
    assert "LOCKED_MM_SCREEN_VARIANT" in source
    assert "select_variant_id" in source
