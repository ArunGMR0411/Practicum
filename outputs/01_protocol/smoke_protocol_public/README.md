# Public smoke protocol (no CASTLE)

Licence-safe synthetic frames from `tests/fixtures/public_micro/` (24 images).

## Run end-to-end (fresh clone, no CASTLE)

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/app/src"
python scripts/smoke/run_public_smoke_protocol.py
```

Runs box-driven detection (from manifest), layered/solid_mask operators, and
objective_profile balanced routing (copy if no faces). Writes under `runs/`.
