# Final evaluation — FetchPush-v4

50 episodes per seed, deterministic policy, eval seeds 10000..10049 (disjoint from training).

| Arm | Seeds | Success rate (mean ± std) | Per-seed |
|---|---|---|---|
| push_sb3_her | 3 | 0.040 ± 0.000 | s0=0.04, s1=0.04, s2=0.04 |
| push_sb3_sac | 3 | 0.993 ± 0.009 | s0=1.00, s1=0.98, s2=1.00 |
| push_td3_her | 3 | 0.993 ± 0.009 | s0=1.00, s1=1.00, s2=0.98 |
| push_td3_noher | 3 | 0.040 ± 0.000 | s0=0.04, s1=0.04, s2=0.04 |
