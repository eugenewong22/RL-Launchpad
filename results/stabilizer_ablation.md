# Single-factor ablation of the stabilizer package

The reported config deviates from SB3's published Fetch values in five places (`results/config_diff.md`). The original campaign changed all five at once, so `docs/writeup.md` §5 could report only that the package works. Each run below reverts exactly ONE setting to its `TrainConfig` default — the value the failed campaign used — holding everything else at the reported config. Seed 0, 1M env steps.

**Read the third column, not the second.** Final success saturates: every single-factor arm reaches ~1.0 by 1M steps, so final success alone says nothing matters. Steps to a *durable* 0.9 (reached and never dropping below 0.8 again) is where the differences live.

The reported config's own three seeds reach durable 0.9 at 410,000–600,000 steps. That band is the noise floor: with one seed per ablation arm, a result inside it is not evidence of an effect.

| Reverted setting | Change | Success | Steps to 0.9 | Contact | Effect |
|---|---|---|---|---|---|
| *none (reported run)* | — | **1.000** | **450,000** | 0.97 | — |
| `normalize_obs` | True → False | 0.920 | 910,000 | 0.98 | **slower** (2.0x) |
| `expl_noise` | 0.2 → 0.1 | 1.000 | 600,000 | 0.97 | within seed noise |
| `tau` | 0.05 → 0.005 | 1.000 | 390,000 | 0.97 | within seed noise |
| `action_l2` | 1.0 → 0.0 | 1.000 | 350,000 | 0.98 | within seed noise |
| *all five (archived)* | — | 0.050 | never | — | **collapses** |

`contact` is `contact_frac`, the fraction of episodes that moved the block. It stays high in every single-factor arm — so actor saturation, the mechanism the original campaign died of, does not recur when only one setting is reverted.

**What this does and does not license.** No single reversion reproduces the collapse, so no one setting is *necessary*. That is not the same as the changes not mattering: the all-five-reverted control flat-lined at the floor on three independent arms, so the failure was real and reproducible. Two explanations remain indistinguishable at n=1 per arm — genuine redundancy (several mechanisms each suffice), or a collapse fragile enough that almost any perturbation escapes it. Separating them needs 3 seeds per arm.

> **Incomplete — 1 of 5 arms not yet run:** `random_eps`. Run `bash scripts/run_ablation.sh --resume` and regenerate. Conclusions above are provisional until every arm is present.

