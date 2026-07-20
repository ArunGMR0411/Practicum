# Pipeline and App Tests

Integration tests for the demonstrator wizard, detection review, compute-aware
recommendations, method selection, policy registry, selected-versus-applied
logging, public fixtures, and end-to-end execution contracts.

Run lightweight cases from the repository root:

```bash
.venv/bin/python -m pytest tests/pipeline_app -m "not e2e_assets"
```

