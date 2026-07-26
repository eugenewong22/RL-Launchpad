# Single-factor ablation of the stabilizer package

The reported config deviates from SB3's published Fetch values in five places (`results/config_diff.md`). The original campaign changed all five at once, so `docs/writeup.md` §5 could report only that the package works. Each run below reverts exactly ONE setting to its `TrainConfig` default — the value the failed campaign used — holding everything else at the reported config. 1M env steps each. Seed 0 is the screen; arms whose seed-0 result looked like an effect were re-run on seeds 1 and 2 (the `n` column).

**Read the third column, not the second.** Final success saturates: every single-factor arm reaches ~1.0 by 1M steps, so final success alone says nothing matters. Steps to a *durable* 0.9 (reached and never dropping below 0.8 again) is where the differences live.

The reported config's own three seeds reach durable 0.9 at 410,000–600,000 steps. That band is the noise floor: with one seed per ablation arm, a result inside it is not evidence of an effect.

| Reverted setting | Change | n | Success | Steps to 0.9 | Contact | Effect |
|---|---|---|---|---|---|---|
| *none (reported run)* | — | 3 | **1.000** | **450,000** | 0.97 | — |
| `expl_noise` | 0.2 → 0.1 | 1 | 1.000 | 600,000 | 0.97 | within seed noise |
| `normalize_obs` | True → False | 3 | 0.967 | 580,000 (910k, 470k, 360k) | 0.97 | within seed noise |
| `tau` | 0.05 → 0.005 | 1 | 1.000 | 390,000 | 0.97 | within seed noise |
| `action_l2` | 1.0 → 0.0 | 1 | 1.000 | 350,000 | 0.98 | within seed noise |
| `random_eps` | 0.3 → 0.0 | 3 | 0.050 | **never** (all 3 seeds) | 0.04 | **collapses** |
| *all five (archived)* | — | 1 | 0.050 | never | — | **collapses** |

`contact` is `contact_frac`, the fraction of episodes that moved the block — the mechanism signal, since the original campaign died with a saturated actor that never reached the block.

## What this shows

**Exactly one of the five is necessary.** Reverting `random_eps` (0.3 → 0.0) puts the run back on the floor for all 1M steps at contact 0.04 — the original campaign's failure reproduced by a single-line change. The other 4 are individually removable: the agent still solves the task without any one of them.

**Not individually load-bearing:** `action_l2` (n=1), `normalize_obs` (n=3), `tau` (n=1), `expl_noise` (n=1). Each lands inside the reported config's own seed spread (410,000–600,000 steps), so there is no evidence any of them changes the outcome on its own.

**Necessary is not sufficient.** During the original diagnosis, adding sustained ε-random exploration to the *broken* config did not rescue it (the block moved in 0/50 episodes — a collapsed policy acts as a restoring force against per-step random actions). Removing it from the *working* config breaks it. Both are true: it is required, and it needs at least one of the others alongside it.

Caveat: the arms still at n=1 can identify a collapse but not resolve a small regression. The one arm where a single seed *did* look like an effect is instructive: reverting `normalize_obs` cost 910k steps on seed 0, which read as a 2x sample-efficiency penalty — seeds 1 and 2 then came back at 470k and 360k, inside the reported band and one of them faster than any reported seed. We withdrew that claim rather than keep the number that suited us.

