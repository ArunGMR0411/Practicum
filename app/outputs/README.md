# App run outputs

Each wizard run creates a timestamped folder under `runs/`:

- `anonymised/` - privacy-processed images
- `side_by_side/` - before/after previews
- `detections/` - face/text/screen boxes
- `metadata/` - decisions and summaries
- `report/success_report.md` - run report
- `state.json` - wizard stage state

Do not commit private run outputs.
