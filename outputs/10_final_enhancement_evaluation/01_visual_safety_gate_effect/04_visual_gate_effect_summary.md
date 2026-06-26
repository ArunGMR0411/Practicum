# Visual-safety gate effect (structured summary)

Purpose: support the deployable-policy decision that advanced/generative
methods remain research-only despite strong numerical privacy metrics.

This package **restructures existing audits** (uniform visual review +
author-recipe five-case gate + decision-framework eligibility). It is not a
new large visual study.

## Key result

- Methods scored for deployment: `9`.
- Default-eligible under gates: `3`.
- Strong-numeric but blocked by visual gate: `4`.

Blocked strong-numeric methods (illustrative):

| method | exploratory | deployment | mean max Re-ID | gate effect |
|--------|------------:|-----------:|---------------:|-------------|
| riddle | 0.9031 | 0.8130 | 0.2967 | BLOCKED_BY_VISUAL_GATE |
| diffusion_low_step | 0.9127 | 0.7829 | 0.3320 | BLOCKED_BY_VISUAL_GATE |
| falco | 0.8107 | 0.7661 | 0.2956 | BLOCKED_BY_VISUAL_GATE |
| nullface | 0.8179 | 0.7218 | 0.3741 | BLOCKED_BY_VISUAL_GATE |

## Interpretation

> Numerical privacy/utility composites can rank generative methods highly while
> visual eligibility fails on egocentric pose/gaze/expression/compositing.
> Hard visual gates therefore block default promotion; methods remain full
> comparable research evidence.

Sources: `outputs/03_anonymisation/16_visual_quality_hardening/`,
`outputs/03_anonymisation/17_author_recipe_visual_audit/`,
`outputs/05_oapr/decision_framework/`.
