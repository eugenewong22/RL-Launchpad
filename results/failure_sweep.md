# Failure-mode sweep on held-out initial states

Env seeds 2000..2199 — disjoint from both the training seeds and the R4 eval seeds (10000+).

| Run | Success | Failures | Failing seed (required push m / progress / closest approach m) |
|---|---|---|---|
| push_td3_her_seed0 | 0.995 | 1/200 | 2081 (0.298 / 17% / 0.043) |

Success vs required push distance — `push_td3_her_seed0`:

| Required push (m) | n | Success |
|---|---|---|
| 0.00–0.10 | 39 | 1.000 |
| 0.10–0.15 | 41 | 1.000 |
| 0.15–0.20 | 42 | 1.000 |
| 0.20–0.25 | 36 | 1.000 |
| 0.25–0.60 | 42 | 0.976 |

