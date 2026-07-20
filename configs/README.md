# Configuration

All four JSON files in this directory are required:

| File | Purpose | Lifecycle |
| --- | --- | --- |
| `policy_registry.json` | Scientific policy IDs, App profiles, compute tiers, methods, and fallbacks | Maintained source configuration |
| `scoring_definitions.json` | Canonical stage formulas, weights, and gates | Maintained source configuration |
| `evidence_source_priority.json` | Precedence rules that prevent stale secondary evidence from replacing canonical results | Maintained source configuration |
| `system_config.json` | Safe default runtime and batch-size profile | Required default template; may be regenerated for the current machine |

Regenerate only the machine profile with:

```bash
python3 src/utils/system_detector.py
```

The other files are authoritative inputs and must not be treated as generated
run output. Validate policy or score changes with the relevant tests before
regenerating evidence.
