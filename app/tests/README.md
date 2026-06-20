# App Tests

App-level tests live in the repository-level `tests/` directory so they can reuse the shared CASTLE pipeline fixtures.

Relevant tests:

- `tests/pipeline_app/test_app.py`
- `tests/pipeline_app/test_pipeline_e2e.py`
- `tests/core_structure/test_compute_policy.py`

Run from the repository root:

```bash
.venv/bin/python -m pytest tests/pipeline_app/test_app.py tests/core_structure/test_compute_policy.py
```
