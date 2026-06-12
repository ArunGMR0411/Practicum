#!/usr/bin/env python3
"""Build the single canonical method eligibility artefact.

Merges:
  - final method eligibility (visual quality hardening)
  - author-recipe five-case outcomes
  - structured author-inspection fail rates (when present)
  - App method id aliases

All routing and App method_catalog logic should consult this file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ELIG = ROOT / "outputs/03_anonymisation/16_visual_quality_hardening/03_final_method_eligibility.csv"
FIVE = ROOT / "outputs/03_anonymisation/17_author_recipe_visual_audit/02_five_case_visual_gate.csv"
DUAL = ROOT / "outputs/03_anonymisation/14_group2_comparison/14_expanded_review_provenance_rollup.csv"
OUT_DIR = ROOT / "outputs/03_anonymisation/20_canonical_method_eligibility"
OUT_CSV = OUT_DIR / "01_canonical_method_eligibility.csv"
OUT_JSON = OUT_DIR / "02_canonical_method_eligibility.json"
OUT_MD = OUT_DIR / "03_canonical_method_eligibility.md"

# Map scientific / evidence method names → App method_catalog ids
APP_ALIASES = {
    "solid_mask_black": ["solid_mask"],
    "layered_blur_downscale_noise": ["layered"],
    "blur": ["blur"],
    "pixelate": ["pixelate"],
    "no_action_copy": ["copy"],
    "nullface": ["nullface"],
    "diffusion_low_step": ["diffusion"],
    "reverse_personalization": ["reverse_personalization"],
    "styleid_stylegan": ["stylegan"],
    "fams": ["fams"],
    "riddle": ["riddle"],
    "falco": ["falco"],
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = list(csv.DictReader(ELIG.open(encoding="utf-8"))) if ELIG.is_file() else []

    five_fail: dict[str, int] = {}
    five_n: dict[str, int] = {}
    if FIVE.is_file():
        for row in csv.DictReader(FIVE.open(encoding="utf-8")):
            m = str(row.get("method", "")).lower()
            if m in {"styleid"}:
                m = "styleid_stylegan"
            if m in {"diffusion"}:
                m = "diffusion_low_step"
            five_n[m] = five_n.get(m, 0) + 1
            if str(row.get("default_route_pass", "no")).lower() in {"no", "0", "false"}:
                five_fail[m] = five_fail.get(m, 0) + 1

    review_fail: dict[str, float] = {}
    if DUAL.is_file():
        for row in csv.DictReader(DUAL.open(encoding="utf-8")):
            m = str(row.get("method", ""))
            try:
                review_fail[m] = float(row.get("structured_author_fail_rate") or 0)
            except ValueError:
                review_fail[m] = 0.0

    rows = []
    for row in base:
        method = str(row["method"])
        elig = str(row.get("eligibility", ""))
        role = str(row.get("final_role", ""))
        reason = str(row.get("reason", ""))
        default_ok = elig.upper() == "ELIGIBLE"
        research_only = "RESEARCH" in elig.upper() or elig.upper() == "EXCLUDED"
        app_ids = APP_ALIASES.get(method, [method])
        rows.append(
            {
                "method": method,
                "app_method_ids": "|".join(app_ids),
                "eligibility": elig,
                "final_role": role,
                "default_route_eligible": "yes" if default_ok else "no",
                "app_default_eligible": "yes" if default_ok else "no",
                "app_catalog_selectable": "yes" if elig.upper() != "EXCLUDED" else "no",
                "research_only": "yes" if research_only and not default_ok else "no",
                "author_recipe_five_case_fail_rate": (
                    f"{five_fail.get(method, 0) / five_n[method]:.3f}" if method in five_n else ""
                ),
                "structured_author_fail_rate": (
                    f"{review_fail[method]:.3f}" if method in review_fail else ""
                ),
                "reason": reason,
                "justifying_artefacts": (
                    "outputs/03_anonymisation/16_visual_quality_hardening/03_final_method_eligibility.csv; "
                    "outputs/03_anonymisation/17_author_recipe_visual_audit/; "
                    "outputs/03_anonymisation/14_group2_comparison/14_expanded_review_provenance_rollup.csv"
                ),
            }
        )

    fields = list(rows[0].keys()) if rows else []
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    payload = {
        "schema_version": "1.0.0",
        "canonical_path": str(OUT_CSV.relative_to(ROOT)),
        "methods": rows,
        "default_eligible_methods": [r["method"] for r in rows if r["default_route_eligible"] == "yes"],
        "app_default_eligible_app_ids": sorted(
            {
                aid
                for r in rows
                if r["app_default_eligible"] == "yes"
                for aid in r["app_method_ids"].split("|")
                if aid
            }
        ),
        "research_only_app_ids": sorted(
            {
                aid
                for r in rows
                if r["research_only"] == "yes"
                for aid in r["app_method_ids"].split("|")
                if aid
            }
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Canonical method eligibility",
        "",
        "Single artefact for scientific routing gates and App method catalog defaults.",
        "",
        f"- CSV: `{OUT_CSV.relative_to(ROOT)}`",
        f"- JSON: `{OUT_JSON.relative_to(ROOT)}`",
        "",
        "| Method | Eligibility | Default route | App selectable | Role |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        md.append(
            f"| {r['method']} | {r['eligibility']} | {r['default_route_eligible']} | "
            f"{r['app_catalog_selectable']} | {r['final_role']} |"
        )
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"n_methods": len(rows), "out": str(OUT_CSV)}, indent=2))


if __name__ == "__main__":
    main()
