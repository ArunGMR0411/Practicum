#!/usr/bin/env python3
"""Build gated OAPR report figure: sensitivity winners + presence vs localisation multimodal scores."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DF = ROOT / "outputs/05_oapr/decision_framework"
OUT_DIR = DF / "08_comparison_to_exploratory"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sens = pd.read_csv(DF / "05_sensitivity/02_face_eligible_winners_by_weight.csv")
    mm = pd.read_csv(DF / "03_stage_scores/03_multimodal_detection_deployment.csv").iloc[0]
    face = pd.read_csv(DF / "03_stage_scores/04_face_anonymisation_deployment.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    # --- Left: sensitivity winners ---
    ax = axes[0]
    weights = sens["privacy_weight"].astype(float).tolist()
    methods = sens["method"].tolist()
    scores = sens["score"].astype(float).tolist()
    colours = {
        "solid_mask_black": "#1f4e79",
        "layered_blur_downscale_noise": "#2e7d4f",
        "blur": "#b86b00",
    }
    bar_colors = [colours.get(m, "#666666") for m in methods]
    x = np.arange(len(weights))
    bars = ax.bar(x, scores, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"wp={w:.2f}" for w in weights])
    ax.set_ylabel("gated OAPR score (eligible only)")
    ax.set_xlabel("Privacy weight (utility = 0.90 − privacy; runtime/success = 0.05 each)")
    ax.set_title("A. Face anonymisation: eligible winner by weight")
    ax.set_ylim(0.70, max(scores) * 1.08 if scores else 1.0)
    for i, (m, s) in enumerate(zip(methods, scores)):
        short = {
            "solid_mask_black": "solid_mask",
            "layered_blur_downscale_noise": "layered",
            "blur": "blur",
        }.get(m, m[:12])
        ax.text(i, s + 0.004, short, ha="center", va="bottom", fontsize=8, rotation=0)
    # legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=l)
        for l, c in [
            ("solid_mask (privacy-leaning)", colours["solid_mask_black"]),
            ("layered (utility-leaning)", colours["layered_blur_downscale_noise"]),
            ("blur", colours["blur"]),
        ]
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # --- Right: presence vs localisation multimodal scores ---
    ax = axes[1]
    labels = [
        "Presence composite\npresence OAPR",
        "gated OAPR split\ndetection score",
        "Screen IoU50\nrecall",
        "Text region\nrecall",
        "Combined presence\nrecall",
    ]
    values = [
        float(mm["exploratory_combined_oapr"]),
        float(mm["oapr_deployment_multimodal_detection_score"]),
        float(mm["screen_iou50_recall"]),
        float(mm["text_region_recall"]),
        float(mm["combined_presence_recall"]),
    ]
    cols = ["#6b4c9a", "#c0392b", "#2980b9", "#16a085", "#7f8c8d"]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=cols, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Score / recall")
    ax.set_ylim(0, 1.15)
    ax.set_title("B. Multimodal: paired scores (stricter ≠ worse system)")
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.03,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(0.95, color="#6b4c9a", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.annotate(
        "Presence OAPR high\nwhile IoU/text expose limits",
        xy=(1, values[1]),
        xytext=(2.2, 0.35),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="#333"),
    )
    ax.grid(axis="y", alpha=0.3)

    # caption strip: research-only note
    research = face[~face["eligible_default_policy"].astype(bool)]["method"].tolist()
    fig.suptitle(
        "OAPR progressive evaluation - gated deployment selection (exploratory composites retained for comparison)\n"
        f"Research-only (gated out of default): {', '.join(research[:8])}"
        + ("…" if len(research) > 8 else ""),
        fontsize=10,
        y=1.02,
    )

    out_png = OUT_DIR / "03_sensitivity_and_localisation_scores_figure.png"
    out_pdf = OUT_DIR / "03_sensitivity_and_localisation_scores_figure.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_png)
    print("wrote", out_pdf)

    # small markdown companion
    md = OUT_DIR / "03_sensitivity_and_localisation_scores_figure.md"
    md.write_text(
        "\n".join(
            [
                "# OAPR progressive evaluation figure",
                "",
                "Canonical path: `outputs/05_oapr/decision_framework/08_comparison_to_exploratory/`.",
                "",
                f"![Sensitivity and presence vs localisation multimodal scores]({out_png.name})",
                "",
                "## Panel A - Sensitivity (eligible face methods only)",
                "",
                sens.to_markdown(index=False),
                "",
                "## Panel B - Presence vs localisation multimodal detection scores",
                "",
                f"- Presence composite (exploratory): `{float(mm['exploratory_combined_oapr']):.4f}`",
                f"- Localisation-oriented deployment score: `{float(mm['oapr_deployment_multimodal_detection_score']):.4f}`",
                f"- Screen IoU50 recall: `{float(mm['screen_iou50_recall']):.4f}`",
                f"- Text region recall: `{float(mm['text_region_recall']):.4f}`",
                f"- Combined presence recall: `{float(mm['combined_presence_recall']):.4f}`",
                "",
                "Interpretation: the drop from ~0.95 to ~0.62 is a **stricter metric**, not a system regression.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("wrote", md)


if __name__ == "__main__":
    main()
