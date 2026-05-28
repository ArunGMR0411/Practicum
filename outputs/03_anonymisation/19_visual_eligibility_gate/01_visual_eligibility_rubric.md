# Visual eligibility rubric (frozen structure)

**Role:** strengthen the visual gate beyond the five-case author-recipe stress audit.
**Status:** completed structured visual inspection and annotation correction.
**Reviewers/source:** project author and project partner using custom review applications; some expanded rows are inherited or heuristic and are classified in `05_manual_review_provenance.csv`.
**Five-case author-recipe audit:** retained as a **stress test**, not the sole deployment gate.

## Decision rule (deployment)

A face anonymisation method is **default-eligible** only if:

1. Failure-rate gate passes on the locked comparable protocol.
2. Privacy floor gates pass under the chosen objective.
3. **Visual eligibility** = ELIGIBLE under the retained structured visual-review evidence
   (majority of items not worse than threshold; no critical failure class excess).
4. Author-recipe five-case audit remains a **stress test**, not the sole gate.

## Item-level scores (1–5, higher is better)

| Dimension | 1 | 3 | 5 |
| --- | --- | --- | --- |
| Identity obfuscation | Original identity obvious | Partial | Identity clearly altered/obscured |
| Geometric plausibility | Broken pose/landmarks | Mild warp | Plausible face geometry |
| Expression / gaze coherence | Uncanny or frozen wrong gaze | Minor issues | Natural for scene |
| Compositing / blend | Hard seams, colour mismatch | Soft edges imperfect | Seamless in context |
| Egocentric suitability | Fails profile/partial/distant | Mixed | Holds under wearable conditions |

**Critical fail flags (binary):** `pose_collapse`, `multi_face_bleed`, `background_corruption`, `identity_leak_obvious`.

**Item verdict:**

- `ELIGIBLE_ITEM` if mean dimension ≥ 3.5 and no critical flag
- `BORDERLINE_ITEM` if mean in [3.0, 3.5) and no critical flag
- `FAIL_ITEM` otherwise

**Method verdict (sample):**

- `ELIGIBLE` if FAIL_ITEM rate ≤ 15% and critical-flag rate ≤ 10%
- `RESEARCH_ONLY_NOT_DEFAULT` otherwise

## Stratified sample design

- Population: locked face anonymisation 500-frame protocol (face-positive / hard egocentric strata).
- Size: stratified sample meeting the ≥ 60-item target design.
- Strata: multi/single face; small/medium/large; edge/partial/profile/frontal; low light / motion blur when labeled.
- Methods reviewed: generative comparators (RiDDLE, FALCO, NullFace, diffusion, RP, StyleID/StyleGAN family) plus visual-safe control (layered).
- Manual author and partner inspection was used where source records establish it. Expanded inherited or heuristic rows are not treated as independent partner ratings.

## Review summaries

- Retain item verdicts and failure classifications with their provenance class.
- Do not calculate inter-rater reliability over mixed manual, inherited, and heuristic records.

Script: `scripts/anonymisation/summarise_manual_visual_review.py`
Completion record: `03_manual_author_partner_review_completion.md`

## Structured review outcome (summary)

Structured visual inspection found that generative / research comparators fail default visual eligibility on hard egocentric conditions (pose, gaze, expression, geometry, compositing), while deterministic visual-safe methods remain the only default-eligible class. This aligns with the gated decision-framework eligibility tables (`ELIGIBLE` for solid_mask / layered / blur; `RESEARCH_ONLY_NOT_DEFAULT` for generative methods).

## Relationship to five-case author-recipe audit

Keep `outputs/03_anonymisation/17_author_recipe_visual_audit/` as a **hard stress test** for domain-transfer failure modes. It is necessary but not sufficient for default eligibility; the broader structured visual evidence supplies the visual gate.
