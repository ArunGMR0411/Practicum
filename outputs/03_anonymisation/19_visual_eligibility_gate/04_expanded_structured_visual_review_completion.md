# Expanded structured visual review — provenance correction

**Status:** completed as **structured visual review** expansion; terminology corrected

**Canonical process description:** [`03_manual_author_partner_review_completion.md`](03_manual_author_partner_review_completion.md)
**Provenance:** [`05_manual_review_provenance.csv`](05_manual_review_provenance.csv)

## What the expansion artefacts contain

Tables under `outputs/03_anonymisation/14_group2_comparison/13_expanded_structured_visual_review.csv` and related provenance summaries aggregate:

| rating_source class | Provenance class |
| --- | --- |
| `group2_first_pass_100`, `diffusion_quality_pass1`, `author_recipe_five_case` | Structured author inspection |
| `dual20_partner_human`, `diffusion_quality_pass2`, recorded partner five-case rows | Genuine manual partner-review records; independence not established |
| `partner_align_author_fail_group2` | Inherited / reused author fail |
| `partner_residual_edge=*`, `author_residual_edge=*`, residual-edge deltas | Heuristic / automatically generated assessment |

## Forbidden claims from these tables

- Do **not** report agreement or reliability statistics over the mixed-source expanded set.
- Do **not** describe the expansion as dual-blind review.
- Do **not** treat residual-edge partner labels as independent human partner ratings.

## Allowed claim

Structured author and author–partner visual inspection (with some automated residual-edge assistance in the expansion pipeline) supports keeping generative methods **RESEARCH_ONLY_NOT_DEFAULT**.
