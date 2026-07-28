# Does our exploration fix port to SB3's TD3?

Our stabilizer ablation (`results/stabilizer_ablation.md`) shows sustained ε-random mixing is the one setting of five whose removal collapses our agent, on all three seeds. SB3's TD3+HER also fails on FetchPush and has no equivalent knob. So: does adding that one mechanism explain SB3's failure?

`src/baseline/td3_eps_random.py` subclasses SB3's TD3 and mixes in uniform actions at the same rate our agent uses (0.3), matching `src/agent/train.behavior_action` exactly. Every other value stays at SB3's published settings, so a difference is attributable to this and nothing else. 3 seeds × 1M env steps.

| Arm | Final success (mean ± std) | Per-seed final | Per-seed best-ever |
|---|---|---|---|
| **SB3 TD3+HER + our ε-random** *(modified)* | 0.050 ± 0.000 | 0.05, 0.05, 0.05 | 0.05, 0.05, 0.05 |
| SB3 TD3+HER, published settings | 0.050 ± 0.000 | 0.05, 0.05, 0.05 | 0.05, 0.05, 0.05 |
| TD3+HER, from scratch (ours) | 1.000 ± 0.000 | 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00 |

## Answer: no

Adding sustained ε-random exploration does **not** rescue SB3's TD3+HER. All three seeds finish at the 0.05 eval floor, and none exceeds it at any point in 1M steps — this is not a policy that learned and then collapsed, it never left the floor.

This refutes the obvious inference from the ablation. Removing ε-random collapses *our* agent (3/3 seeds), so it is necessary for us — but adding it to SB3's TD3 changes nothing, so it is not *sufficient* there. The gap between the two implementations is therefore not solely the exploration schedule, and we do not know what else it is. The two remaining candidates are the mechanisms SB3 has no knob for — the action-L2 penalty and observation normalization — but our own ablation found neither individually necessary, so we decline to guess.

**Why this result is trustworthy rather than a null.** A silently inactive override produces exactly this flat curve, and the two readings are opposite: *the mechanism does not help* versus *we never tested the mechanism*. Measured inside `model.learn()`, the override fires at 0.302 against 0.300 configured; `tests/test_td3_eps_random.py::test_override_is_live_during_learn` pins it.

