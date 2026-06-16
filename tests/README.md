# Tests

Run the full suite from the repository root:

```bash
.venv/bin/pytest
```

Run one group or file:

```bash
.venv/bin/pytest tests/detection
.venv/bin/pytest tests/pipeline_app/test_app.py
```

## Groups

- `core_structure/`: repository layout and compute policy.
- `data_protocol/`: locked splits, manifests, and materialisation.
- `annotation_review/`: annotation workflow.
- `detection/`: face, text, and screen detection.
- `anonymisation/`: anonymisers and redaction.
- `evaluation/`: perceptual, OCR, FID, and Re-ID metrics.
- `routing/`: rule-based and learned routing.
- `pipeline_app/`: demonstrator workflow and contracts.

Tests that require local CASTLE frames or model weights use the `e2e_assets` marker and are skipped when those assets are unavailable. Lightweight tests use synthetic fixtures in `tests/fixtures/public_micro/`.
