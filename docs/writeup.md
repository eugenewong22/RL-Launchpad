# From-Scratch TD3 + HER on Sparse-Reward Fetch Manipulation

*LaunchPad 2026 — Griffin Labs RL-From-Scratch track. ≤4 pages.*

## 1. Problem

Contact-rich tabletop manipulation with **sparse rewards**: a 7-DoF Fetch
arm must push a block to a commanded 3-D goal (`FetchPush-v4`, stock task,
no modifications). The reward is −1 every step until the block is within
5 cm of the goal, then 0. This is the regime where scripted automation
breaks: a pre-programmed push sequence assumes fixed block and goal poses,
and any perturbation of either invalidates the contact schedule. A policy
must *perceive* the current block/goal configuration and re-plan the
contact through it.

Why existing approaches are insufficient here:
- **Scripted / classical control** (PID to pre-computed waypoints) cannot
  generalize across the task's randomized block and goal poses — the
  waypoints themselves are functions of state it doesn't react to.
- **Dense-reward RL** requires hand-shaping (distance terms, contact
  bonuses) that is task-specific engineering and a known source of reward
  hacking; the sparse task statement is the honest one.
- **Vanilla off-policy RL on sparse reward** almost never sees a success
  and cannot bootstrap value — our no-HER ablation quantifies exactly this.

**Success criteria, fixed before building:** (a) from-scratch agent
reaches a mean success rate within the SB3 TD3+HER baseline's mean±std
band at 1M env steps, over 3 seeds × 50 fixed eval episodes; (b) the
no-HER ablation clearly underperforms, demonstrating the mechanism we
claim matters actually does; (c) a judge reproduces our eval from a clean
clone in under 15 minutes.

## 2. Approach

**Algorithm: TD3 + Hindsight Experience Replay, written from scratch in
PyTorch** (`src/agent/`, ~500 lines). Training loop, replay buffer, HER
relabeling, networks, and update rule are all ours; only autograd, Adam,
and the simulator are library code (per R1).

```
                    ┌─────────────────────────────┐
 obs(10) ┐          │ Actor: 13 → 256 → 256 → 4   │ → tanh·max_action → a
 goal(3) ┴ s(13) ──▶├─────────────────────────────┤
                    │ Critic₁: 13+4 → 256 → 256 →1│ ┐
                    │ Critic₂: 13+4 → 256 → 256 →1│ ┴→ min(Q₁,Q₂) targets
                    └─────────────────────────────┘
   (+ Polyak-averaged target copies of all three, τ=0.005)
```

Each major decision, with the alternative we ruled out:

| Decision | Why | Rejected alternative & its shortcoming |
|---|---|---|
| Off-policy + HER | Sparse goal-conditioned reward: relabeling failed episodes with achieved goals is the only signal source | PPO (on-policy): cannot reuse relabeled experience; dense shaping: reward engineering we'd have to defend per-task |
| TD3 over SAC | Fewer moving parts under a deadline; deterministic eval; each trick is an explainable overestimation fix | SAC: entropy temperature is one more tunable, stochastic eval adds variance to R4 numbers. **In hindsight this was the costly choice**: a deterministic actor is what saturates under sparse reward (§5), and SAC's entropy bonus makes it structurally immune. We reached the same final performance, but SAC would have got there without the three stabilizers. We report this rather than presenting TD3 as obviously correct. |
| HER `future`, k=4 | Relabels 80% of sampled transitions from later same-episode states; the paper's recommended strategy | `final` strategy: fewer distinct goals per episode, weaker coverage near trajectory ends |
| Sample-time relabeling | Fresh counterfactual goals every epoch from the same episodes | Store-time relabeling: freezes k copies, inflates memory k× |
| MLP 256-256 | Fetch state is 13-D; capacity is not the bottleneck, stability is | Deeper/wider nets: slower, no gain at this input size (deliberate simplicity) |
| 1 gradient step per env step | Matches SB3's tuned throughput → fair same-x-axis comparison | Higher update ratios: better sample efficiency but confounds the R2 comparison |

What we deliberately did **not** build: distributional critics, prioritized
replay, parallel envs, observation normalization. Each was considered and
cut because the baseline comparison, not peak performance, is the claim.

Hyperparameters follow SB3's published tuned Fetch values (γ=0.95,
τ=0.005, lr=1e-3, batch 256, buffer 1e6) — deviations would need defending
and none were needed. <!-- TODO: update if any change before submission -->

## 3. Evidence

*All numbers regenerate from committed CSVs via `scripts/make_plots.py`;
eval protocol: deterministic policy, 50 episodes, eval seeds 10000–10049,
disjoint from all training seeds (R4).*

**Correctness gate (FetchReach):** 100% success from 7.5k env steps,
sustained through 50k, 1.9 min wall-clock on laptop CPU
(`results/reach_smoke_seed0/`).

**Classical baseline (scripted two-phase push controller,
`scripts/diagnose_push.py`):** 54% success on the identical 50-episode
R4 protocol. This is the "simple controller" bar the learned policies
must beat — and it is not trivial: it fails exactly where open-loop
scripting should (goals requiring re-approach after overshoot).

**FetchPush, 3 seeds × 1M steps** (in-training eval, 20 episodes on the
fixed eval seeds; 50-episode R4 numbers in `results/final_eval.md`):

| Arm | Final success (per seed) | Verdict |
|---|---|---|
| **TD3+HER (from scratch)** | **1.00, 1.00, 1.00** | Solved on every seed |
| SAC+HER (SB3 baseline) | 1.00, 0.95, 1.00 | Matches us — the strongest baseline |
| TD3+HER (SB3 baseline) | 0.05, 0.05, 0.05 | Never learned (see below) |
| TD3 no-HER (our ablation) | 0.05, 0.05, 0.05 | Never learned; contact ≤7% |
| Scripted classical controller | 0.54 (deterministic) | The simple-baseline bar |

Four results, each load-bearing:

1. **We match the strongest library baseline and beat the classical one.**
   Our from-scratch agent solves all three seeds (mean return ≈ −10.6,
   i.e. ~11-step solves); SB3's SAC+HER reaches the same plateau. Matching
   a mature, tuned implementation with code written from scratch is the
   claim we set out to prove, and we make it on equal terms: same task,
   same observation/action space, same eval seeds, same protocol.
2. **HER is load-bearing, and we measured how.** The no-HER ablation is
   identical except `her_k=0`, and it never leaves the floor — with
   object contact in ≤7% of episodes versus ~95% for the HER arms. The
   mechanism is visible in the logged `contact_frac`, not inferred.
3. **The stabilizers are load-bearing too — and this is where the
   from-scratch requirement paid off.** SB3's TD3+HER is flat on all three
   seeds under the same budget and hyperparameters, because it has no
   action-L2 penalty, no observation normalization, and no sustained
   ε-random mixing. Our first implementation failed the same way (§5), and
   because we owned every line, we could diagnose it (actor saturation)
   and fix it. This arm is therefore both a baseline and an ablation of
   the three additions.
4. **Deterministic-actor methods need those stabilizers here; SAC does
   not.** SAC's entropy-regularized stochastic policy cannot saturate the
   way a deterministic actor does, which is why it is the one
   out-of-the-box baseline that works. That is the non-obvious domain
   insight this project produced.

**Robustness beyond the protocol:** 199/200 held-out (non-eval) initial
states solved (~0.5% failure); the single failure is in the demo video,
labeled.

**Stretch task — FetchPickAndPlace (same config, zero re-tuning):**
seed 0 = **0.740** on the 50-episode R4 protocol at 1M steps (plateau
70–75%; grasping + in-air goals, so no pushing shortcut exists). The
demo video includes both successes and failures from the protocol seeds
(3/6 in the first six). One config transferring across two contact
tasks without touching a hyperparameter is evidence the recipe, not
per-task tuning, is doing the work.

![learning curves](../results/learning_curves.png)

<!-- TODO: 1-2 sentences interpreting the curves: match/beat/lose vs SB3,
and the honest analysis of why. -->

## 4. Constraints

- **Sample efficiency:** all curves share the env-steps x-axis; the
  no-HER ablation shows what the relabeling buys per step.
- **Compute honesty (R6):** every arm ran 1M env steps — identical
  budgets, so the comparison is on equal terms. Wall-clock (CPU-only
  throughout; per-run hardware in `results/compute_table.md`):

  | Arm | Wall-clock, 1M steps |
  |---|---|
  | TD3+HER (ours) | 42 min laptop (M-series) / ~4.8 h per cluster seed |
  | TD3 no-HER (ours) | ~3.9–4.8 h per cluster seed |
  | TD3+HER (SB3) | ~5.5–6.9 h |
  | SAC+HER (SB3) | ~6.9–8.8 h |

  Two honest notes. The laptop core is ~3× faster per-core than the
  cluster's older Xeons, so cross-hardware times are not comparable —
  env steps are the axis all curves share. And our agent is the
  *cheapest* arm per env step (one deterministic actor, two critics; SAC
  additionally samples and back-props through a stochastic policy and
  tunes an entropy temperature) — worth noting, but it is a consequence
  of the algorithm choice, not something we optimized for.
- **Why CPU:** we measured GPU dispatch to be slower for 256-wide MLPs
  at batch 256; MuJoCo stepping is CPU-bound regardless, so a GPU can
  accelerate only the update half of the loop. We ran on 8 CPU cores per
  job instead, and report that choice rather than assuming more
  resources would have been better.
- **Control-rate realism:** the policy is a 4-layer-equivalent MLP,
  ~0.1 ms/action on CPU — far inside a 25 Hz control budget; no
  inference-side compute concerns at deployment scale.

## 5. Honesty & Trajectory

**Known failure modes:** on 200 held-out initial states (env seeds
2000–2199, disjoint from both the training seeds and the R4 eval seeds)
the seed-0 policy fails once — 0.5%, seed 2081, captured in
`results/push_failure_seed2081.mp4`. Regenerate with
`uv run python scripts/failure_sweep.py`.

The mechanism is **contact without sustained pushing**, not an overshoot.
The gripper reaches the block normally — closest approach 0.043 m, i.e.
touching — and then delivers only 17% of the required displacement: the
block advances 0.298 m → 0.248 m in three small increments separated by
long stationary stretches, against a 0.05 m success threshold. It is
never pushed past the goal, so the policy is not overshooting and failing
to re-approach; it simply stops generating lateral force while in
contact.

We looked for a cause and did not find one we can defend. The failing
state needs a 0.298 m push, placing it in the longest quintile — but 41
of the 42 other states requiring >0.25 m succeed, including states
requiring 0.375 m, further than the one that fails. Required push
distance therefore does not explain it. With a single failure in 200 we
report the mechanism and decline to attribute a cause.
<!-- TODO: re-run the sweep for seeds 1-2 once their checkpoints are off
the cluster; scripts/failure_sweep.py already skips them cleanly. -->

**Negative results (found the hard way, diagnosed systematically):**
Our first full FetchPush campaign — from-scratch TD3+HER, an SB3 TD3+HER
baseline, and a no-HER ablation, 1M env steps each — all flat-lined at
the eval floor (one eval seed starts in a success state; success = 0.05
throughout). Diagnosis with pre-declared probes: (1) a scripted
controller solves the task (env fine); (2) the behavior policy moved the
block in 2% of episodes; (3) the trained actor had **saturated** — mean
|action| = 1.0, gripper pinned ~1.6 m from the block, tanh gradients
dead. Adding sustained ε-random exploration alone did NOT fix it (block
still moved in 0/50 episodes — a collapsed policy acts as a restoring
force against per-step random actions). The failure and its mechanism
shaped the final design: action-L2 penalty, observation normalization,
and reference target-update rate. <!-- TODO: finalize once the fixed
config's runs land; include the flat curves as a figure. -->

**What we'd do with two more weeks:**
1. FetchPickAndPlace (grasping adds a contact mode Push lacks) with the
   same harness — the code path is identical, only the config changes.
2. Vision-from-pixels variant: replace the 13-D state with a CNN encoder
   over rendered frames, keeping the same TD3+HER core, to close the gap
   toward Griffin's VLA-style perception stack.
3. Domain randomization (friction, block mass) + actuation delay to
   quantify the sim-to-real gap rather than assert it.

---
*Repo: pinned deps (`uv.lock`), seeded configs, judge path in README.
From-scratch code (`src/agent/`) is fully separated from baseline code
(`src/baseline/`). Any team member can walk through the loss function and
any architecture decision live (R1).*
