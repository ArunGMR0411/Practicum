# Scripts

Run scripts from the repository root with the project virtual environment.

## Groups

- `data_protocol/`: build and verify locked manifests and subsets.
- `annotation_review/`: prepare and validate reviewed annotations.
- `detection/`: evaluate face detectors and scene-condition profiling.
- `anonymisation/`: run deterministic and advanced face protection methods.
- `multimodal/`: evaluate text and screen detection and redaction.
- `oapr_routing/`: build and evaluate routing policies.
- `metrics_statistics/`: compute shared privacy, utility, and statistical metrics.
- `reporting_traceability/`: generate tables, plots, and evidence maps.
- `app_validation/`: validate the demonstrator workflow.

## Core checks

Verify the locked protocols:

```bash
.venv/bin/python scripts/data_protocol/verify_locked_protocols.py
```

Validate face annotations:

```bash
.venv/bin/python scripts/annotation_review/validate_face_annotations.py
```

Individual scripts document their required inputs and output paths through command-line help or module docstrings.
