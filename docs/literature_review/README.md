# Literature Review

This literature review was completed before implementation and reflects the project’s initial research direction. The final thesis paper contains an updated Related Work section aligned with the implemented methods, experiments, and final contribution.

## Contents

| Path | Description |
| --- | --- |
| `summary/` | Condensed literature review (LaTeX source + PDF) |
| `full/` | Extended literature review (LaTeX source + PDF) |
| `source_papers/` | PDF copies of the surveyed papers (`ref1`…`ref34`) |
| `images/` | Figures used in the full review |

Build (from `summary/` or `full/`):

```bash
pdflatex main && biber main && pdflatex main && pdflatex main
```
