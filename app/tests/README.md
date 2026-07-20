# App Tests

App tests live in the repository-level `tests/pipeline_app/` directory so they
can reuse shared pipeline fixtures. Compute-policy structure tests are under
`tests/core_structure/`.

Run the App test group from the repository root:

```bash
.venv/bin/python -m pytest tests/pipeline_app -m "not e2e_assets"
```

Run the full local group, including tests that require installed model assets,
with `.venv/bin/python -m pytest tests/pipeline_app`.
