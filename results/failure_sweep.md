# Failure-mode sweep on held-out initial states

Env seeds 2000..2199 — disjoint from both the training seeds and the R4 eval seeds (10000+). Every seed is run against every policy, so a state that one policy fails and another solves is distinguishable from a state that is intrinsically hard.

**599/600** across all runs (1 failure).

| Run | Success | Failures | Failing seed (required push m / progress / closest approach m) |
|---|---|---|---|
| push_td3_her_seed0 | 0.995 | 1/200 | 2081 (0.298 / 17% / 0.043) |
| push_td3_her_seed1 | 1.000 | 0/200 | none |
| push_td3_her_seed2 | 1.000 | 0/200 | none |

Success vs required push distance — `push_td3_her_seed0`:

| Required push (m) | n | Success |
|---|---|---|
| 0.00–0.10 | 39 | 1.000 |
| 0.10–0.15 | 41 | 1.000 |
| 0.15–0.20 | 42 | 1.000 |
| 0.20–0.25 | 36 | 1.000 |
| 0.25–0.60 | 42 | 0.976 |

Success vs required push distance — `push_td3_her_seed1`:

| Required push (m) | n | Success |
|---|---|---|
| 0.00–0.10 | 39 | 1.000 |
| 0.10–0.15 | 41 | 1.000 |
| 0.15–0.20 | 42 | 1.000 |
| 0.20–0.25 | 36 | 1.000 |
| 0.25–0.60 | 42 | 1.000 |

Success vs required push distance — `push_td3_her_seed2`:

| Required push (m) | n | Success |
|---|---|---|
| 0.00–0.10 | 39 | 1.000 |
| 0.10–0.15 | 41 | 1.000 |
| 0.15–0.20 | 42 | 1.000 |
| 0.20–0.25 | 36 | 1.000 |
| 0.25–0.60 | 42 | 1.000 |

