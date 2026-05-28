# Manual author–partner visual and annotation review — completion record

**Status:** completed  
**Review type:** `manual_review` / `author_partner_review` / `structured_visual_review`  
**Not:** dual-blind study, blinded participant experiment, or independent inter-rater reliability trial  

**Rubric:** `01_visual_eligibility_rubric.md`  
**Sample template:** `02_structured_review_sample_template.csv`  
**Canonical provenance table:** `05_manual_review_provenance.csv`

## Reviewers

| Role | Identity |
| --- | --- |
| Reviewer 1 | Project author (primary investigator) |
| Reviewer 2 | Project partner |

## Actual process

- The project author and project partner used **custom review applications** and worksheets.
- Images, annotations, detection boxes, and relevant anonymisation outputs were **manually inspected**.
- Missing, false, or inaccurate annotations were **corrected**.
- Visual output inspection supported **method-quality and eligibility** decisions (especially generative methods).
- This was a **structured project review and correction process**, not a blinded human-participant study.

## What is retained

- Corrected protocol annotations and face-box handoff surfaces used by later stages.
- Structured visual inspection records that justify keeping generative methods as **RESEARCH_ONLY_NOT_DEFAULT**.
- Author first-pass operational reviews (e.g. Group2 100-crop) and author–partner confirmation samples where present.

## What is not claimed

- No dual-blind protocol.
- No independent blinded ratings.
- No agreement or reliability statistic is reported. The expanded pipeline includes **inherited**, **heuristic residual-edge**, and **alignment** partner labels that must not be treated as independent ratings.

## Method-level outcome (unchanged)

| Method class | Manual visual inspection outcome | Default deployment role |
| --- | --- | --- |
| solid_mask / layered / blur (visual-safe deterministic) | Pass structured visual eligibility under rubric | Default-eligible (subject to other gates) |
| Generative / research comparators (RiDDLE, FALCO, NullFace, diffusion, RP, StyleID, etc.) | Fail default visual eligibility on hard egocentric strata | **RESEARCH_ONLY_NOT_DEFAULT** |

Consistent with `outputs/03_anonymisation/16_visual_quality_hardening/03_final_method_eligibility.csv` and decision-framework eligibility tables.

## Canonical claim wording

> The author and project partner manually inspected and corrected annotations and relevant outputs using dedicated review applications. This was a structured project review process, not a blinded human-participant study.

## Supersedes

This file replaces the former completion artifact whose filename and claims incorrectly described the work as a blinded study.
