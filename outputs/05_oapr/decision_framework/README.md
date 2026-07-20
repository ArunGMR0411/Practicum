# OAPR decision framework (in-body evaluation)

Part of RQ4 / OAPR evidence under `outputs/05_oapr/`.

Progressive design:
1. Exploratory composite scores (method comparison tables elsewhere in `outputs/02–04`).
2. Safety gates and deployment selection scores (this package).
3. Sensitivity and Pareto analyses for objective-dependent policy claims.

Stage mirrors (navigation next to domain evidence):
- `outputs/02_face_detection/15_deployment_selection/`
- `outputs/03_anonymisation/18_deployment_selection/`
- `outputs/04_multimodal_privacy/01_multimodal_250_evidence/15_deployment_selection/`

```bash
.venv/bin/python scripts/oapr_routing/run_decision_framework_evaluation.py
.venv/bin/python scripts/oapr_routing/build_decision_framework_figure.py
```

The evaluation runner regenerates `09_run_logs/run_trace.txt`,
`09_run_logs/01_run_report.md`, and
`09_run_logs/03_evidence_source_priority.json`. These transient run snapshots
are not retained in the repository. The referenced validation record remains
under `09_run_logs/02_validation_pass.md`.
