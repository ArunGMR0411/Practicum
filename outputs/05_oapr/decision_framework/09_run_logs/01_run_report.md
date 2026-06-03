# deployment selection Run Report

Generated: 2026-07-20 02:47:17

## Completed stages

- Face detection re-score + recall floor gates
- Scene-condition score reweight + label eligibility export
- Multimodal detection split composite
- Face anonymisation gates + max-ReID privacy + non-sensitive utility sample
- Multimodal redaction miss→0 privacy + non-sensitive utility weighting
- Objective modes, sensitivity, Pareto, policy selection
- Exploratory-vs-deployment comparison (non-destructive)

## Snapshot

- Face detector top deployment score: `fusion_rfdetr_yolo11s_scrfd10g` score=0.9165 recall=0.9321 floor_pass=True
- Face anon eligible top: `solid_mask_black` score=0.8142 privacy=0.7073
- MM detection deployment score=0.6333 (presence composite=0.9458)
- MM redaction top test variant: `text_fill_screen_fill` score=0.8889

## Progressive evaluation (in-body)

Evidence lives under `outputs/05_oapr/decision_framework/` with stage mirrors next to
face detection, anonymisation, and multimodal packages. Exploratory composites remain
for comparison tables; gates govern deployable defaults. Ledger: `outputs/09_traceability/01_evidence_index.csv`.
