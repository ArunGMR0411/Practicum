# Group 2 Visual Quality Review

A deterministic random sample of 100 operational face boxes was inspected at crop level against the original frame. The screen records obvious synthesis/compositing artifacts and face-region plausibility; it does not establish demographic preservation, identity semantics, or universal visual quality.

| method   |   sample_size |   visually_plausible_count |   obvious_artifact_count |   visually_plausible_rate |
|:---------|--------------:|---------------------------:|-------------------------:|--------------------------:|
| riddle   |           100 |                         86 |                       14 |                      0.86 |
| falco    |           100 |                         78 |                       22 |                      0.78 |

Two reviewed no-face cases received operational detector boxes and therefore exposed false-positive face insertion in both generative methods. This supports retaining detector confidence, artifact checks, and deterministic privacy fallbacks in the policy.
