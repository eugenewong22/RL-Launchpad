# From-Scratch TD3 + HER on Sparse-Reward Fetch Manipulation

*LaunchPad 2026 — Griffin Labs RL-From-Scratch track. ≤4 pages.*

## 1. Problem

Contact-rich tabletop manipulation with **sparse rewards**: a 7-DoF Fetch
arm must push a block to a commanded 3-D goal (`FetchPush-v4`, stock task,
no modifications), reward −1 every step until the block is within 5 cm of
the goal, then 0. This is where scripted automation breaks — a
pre-programmed push assumes fixed block and goal poses, and any
perturbation invalidates the contact schedule. A policy must perceive the
current configuration and re-plan the contact through it.

Why the obvious approaches fall short: **scripted control** computes
waypoints that don't react to randomized block and goal poses;
**dense-reward RL** needs per-task hand-shaping, a known reward-hacking
source; **vanilla off-policy RL** almost never sees a success and cannot
bootstrap value — our no-HER ablation quantifies that.

**Success criteria, fixed before building:** (a) match the SB3 baseline's
mean±std band at 1M env steps over 3 seeds × 50 fixed eval episodes;
(b) the no-HER ablation clearly underperforms; (c) a judge reproduces our
eval from a clean clone in under 15 minutes.

## 2. Approach

**Algorithm: TD3 + Hindsight Experience Replay, written from scratch in
PyTorch** (`src/agent/`, ~680 lines). Training loop, replay buffer, HER
relabeling, networks, normalizer, and update rule are all ours; the only
imports are `torch`, `numpy`, `gymnasium`/`gymnasium-robotics`, `yaml`
and the standard library, so autograd, Adam and the simulator are the
entire extent of the library code (per R1).

```
                    ┌─────────────────────────────┐
 obs(10) ┐          │ Actor: 13 → 256 → 256 → 4   │ → tanh·max_action → a
 goal(3) ┴ s(13) ──▶├─────────────────────────────┤
                    │ Critic₁: 13+4 → 256 → 256 →1│ ┐
                    │ Critic₂: 13+4 → 256 → 256 →1│ ┴→ min(Q₁,Q₂) targets
                    └─────────────────────────────┘
   (+ Polyak target copies of all three, τ=0.05 — not the SB3 default
    0.005, see below; running obs-normalization sits in front of s)
```

Each major decision, with the alternative we ruled out:

| Decision | Why | Rejected alternative & its shortcoming |
|---|---|---|
| Off-policy + HER | Sparse goal-conditioned reward: relabeling failed episodes with achieved goals is the only signal source | PPO (on-policy): cannot reuse relabeled experience; dense shaping: reward engineering we'd have to defend per-task |
| TD3 over SAC | Fewer moving parts under a deadline; deterministic eval | SAC: one more tunable (entropy temperature), stochastic eval adds R4 variance. **In hindsight, the costly choice** — a deterministic actor is exactly what saturates under sparse reward (§5), and SAC's entropy bonus is structurally immune. Same final performance, but SAC would have got there without the stabilizers below. |
| HER `future`, k=4 | Relabels 80% of sampled transitions from later same-episode states; the paper's strategy | `final`: fewer distinct goals per episode, weak coverage near trajectory ends |
| MLP 256-256, relabel at sample time | State is 13-D — capacity isn't the bottleneck, stability is; sampling fresh counterfactual goals each epoch beats freezing k copies at store time (k× memory) | Deeper/wider nets: slower, no gain at this input size |
| 1 grad step per env step | Matches SB3's throughput → fair same-x-axis comparison | Higher ratios: better sample efficiency, but confounds R2 |

Not built, deliberately: distributional critics, prioritized replay,
parallel envs — the baseline comparison, not peak performance, is the claim.

**Hyperparameters, and where we deviate.** We started from SB3's published
tuned Fetch values; that configuration **did not learn** (§5). The reported
runs deviate in five places (diff generated from the two committed configs
into `results/config_diff.md`):

| Key | Failed | Reported | Why |
|---|---|---|---|
| `tau` | 0.005 | **0.05** | 10× faster target tracking; slow targets let a saturated actor's value estimates stay self-consistent |
| `action_l2` | — | **1.0** | penalizes squared action magnitude; intended as anti-saturation, but §5 finds it not individually necessary |
| `normalize_obs` | — | **true** | Fetch positions and velocities differ by ~2 orders of magnitude |
| `expl_noise` | 0.1 | **0.2** | wider Gaussian exploration |
| `random_eps` | — | **0.3** | 30% fully-random actions, sustained — **the one change §5's ablation shows is necessary** |

γ=0.95, lr=1e-3, batch 256, buffer 1e6, network width, policy delay and
target-noise clipping are unchanged from SB3. Every reported from-scratch
run — three FetchPush seeds, the no-HER ablation, PickAndPlace — uses this
identical config; nothing was tuned per task or per seed. These five are
*ours*, so the SB3 TD3+HER baseline does not have them; §3 covers what that
does and does not license us to conclude.

## 3. Evidence

*All numbers regenerate from committed CSVs via `scripts/make_plots.py`;
eval protocol: deterministic policy, 50 episodes, eval seeds 10000–10049,
disjoint from all training seeds (R4).*

**Correctness gate (FetchReach):** first 10/10 in-training eval at 7.5k
env steps, holding 10/10 for 17 of the 19 later evals (exceptions: 8/10 at
10k, 9/10 at 47.5k — 10-episode samples, so noise, not regression), 1.9 min
on laptop CPU. On the 50-episode R4 protocol the same checkpoint scores
**0.98** — the gap from the 10-episode 1.00 is a live demonstration of why
R4 mandates ≥50 episodes.

**Classical baseline (scripted two-phase push controller,
`scripts/diagnose_push.py`):** 54% on the identical 50-episode protocol —
the "simple controller" bar, and not a trivial one: it fails exactly where
open-loop scripting should, on goals needing re-approach after overshoot.

**FetchPush, 3 seeds × 1M steps, on the R4 protocol**
(`results/final_eval_push.md`):

| Arm | Success (mean ± std) | Per-seed |
|---|---|---|
| **TD3+HER (from scratch)** | **0.993 ± 0.009** | 1.00, 1.00, 0.98 |
| SAC+HER (SB3 baseline) | 0.993 ± 0.009 | 1.00, 0.98, 1.00 |
| TD3+HER (SB3 baseline) | 0.040 ± 0.000 | 0.04, 0.04, 0.04 |
| TD3 no-HER (our ablation) | 0.040 ± 0.000 | 0.04, 0.04, 0.04 |
| Scripted classical controller | 0.54 | deterministic |

Both floors are one fact at two sample sizes: 2 of the 50 R4 seeds start
already inside the 5 cm threshold (0.040), as does 1 of the 20 in-training
seeds (0.05, the floor in the figures) — a policy that never moves scores
that, not zero.

Four results, each load-bearing:

1. **We match the strongest library baseline and beat the classical one**,
   on equal terms: same task, observation/action space, eval seeds and
   protocol. The match is exact: both score 0.993 ± 0.009, differing only
   in which seed drops one episode (ours seed 2, SAC's seed 1). Equality,
   not superiority — one episode in 150 separates them.
2. **HER is load-bearing, and we measured how.** The ablation is identical
   except `her_k=0` and never leaves the floor — object contact in ≤7% of
   episodes (0.07/0.04/0.04) versus 0.95 on every HER seed. Visible in the
   logged `contact_frac`, not inferred.
3. **Two independent TD3+HER implementations fail at the same
   hyperparameters — which points at the config, not either codebase.**
   SB3's TD3+HER is flat on all three seeds; so was *ours*, on the same task
   and budget, until the five changes in §2 (flat curves in §5; the runs
   are committed under `results/archive_broken_config/`). We diagnosed the
   mechanism in our own code, where we could instrument it: the actor had
   **saturated**, mean |action| = 1.0, tanh gradients dead.
4. **Deterministic-actor methods need those stabilizers here; SAC does
   not** — its entropy-regularized stochastic policy cannot saturate the way
   a deterministic actor does, which is why it is the one out-of-the-box
   baseline that works. That is the non-obvious insight this project
   produced, and it is why §2 records TD3-over-SAC as the costly choice.

On the third point we claim only that the SB3 defaults do not train TD3+HER
on FetchPush in 1M steps, and that two implementations agree on it. We do
**not** claim our five changes would fix SB3's TD3+HER — we never ran that
experiment. That arm is a *failed baseline configuration*, not an ablation
of our additions, and the ~20× gap against it is **not** evidence that our
implementation beats TD3+HER as published. Our fair comparison is SAC+HER,
which we match. Reporting the gap as a win would be the easiest way to
mislead a reader here, so we say plainly it isn't.

**Beyond the protocol:** 599/600 held-out non-eval initial states solved
across the three seeds — one failure total, dissected in §5. **Stretch
task — FetchPickAndPlace**, same config, zero re-tuning: seed 0 = **0.740**
at 1M steps (plateau 70–75%; grasping and in-air goals, so no pushing
shortcut exists). One config transferring across two contact tasks is
evidence the recipe, not per-task tuning, is doing the work.

![learning curves](../results/learning_curves.png)

**Reading the curves.** Both working arms are flat at the floor while the
buffer fills with relabeled failures, then rise sharply once HER has enough
near-goal experience to bootstrap from. On the mean curve we hit 0.5 at
390k env steps to SAC's 440k, and 0.9 at 550k to 670k — but we do **not**
claim a sample-efficiency win, because per-seed steps to a durable 0.9
(reached, never dropping below 0.8 again) don't support it:

| Arm | seed 0 | seed 1 | seed 2 | spread |
|---|---|---|---|---|
| TD3+HER (ours) | 450k | 410k | 600k | 190k |
| SAC+HER (SB3) | 380k | 760k | 650k | 380k |

SAC's *fastest* seed beats our fastest; our better mean rides on SAC's one
slow seed, and the 120k gap in means is smaller than either arm's own
spread, with the ±1σ bands overlapping across the whole rise. The
defensible claim is that we **match** SAC+HER on final success and sample
efficiency — n=3 cannot resolve a difference this size. One qualitative
difference is real: SAC leaves the floor earlier (~150k vs ~250k) but climbs
more gradually, crossing us around 400k — entropy-driven exploration versus
a deterministic actor that needs its first successes before it commits.

## 4. Constraints

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
  cluster's older Xeons, so cross-hardware times are not comparable — env
  steps are the shared axis. And our agent is the *cheapest* arm per env
  step (one deterministic actor, two critics; SAC also back-props a
  stochastic policy and tunes a temperature) — a consequence of the
  algorithm choice, not something we optimized for.
- **Why CPU:** GPU dispatch measured slower for 256-wide MLPs at batch 256,
  and MuJoCo stepping is CPU-bound regardless, so a GPU accelerates only
  half the loop. 8 CPU cores per job; we report the choice rather than
  assume more hardware would have helped.
- **Control rate:** ~0.1 ms/action on CPU, far inside a 25 Hz budget.

## 5. Honesty & Trajectory

**Known failure modes.** Across **600** held-out initial states (env seeds
2000–2199 against all three policies, disjoint from training and R4 eval
seeds) there is exactly **one** failure: seed 0 on state 2081
(`results/push_failure_seed2081.mp4`; sweep in `results/failure_sweep.md`).

The mechanism is **contact without sustained pushing**, not overshoot: the
gripper reaches the block (closest approach 0.043 m) then delivers 17% of
the required displacement, 0.298 m → 0.248 m against a 0.05 m threshold,
never pushing it past the goal. Running every state against every seed tells
us what one seed could not — **seeds 1 and 2 both solve state 2081**, and
every other state needing >0.25 m succeeds on every seed. So it is neither
an intrinsically hard state nor a push-distance limit; it is a property of
seed 0's policy. We localize it that far and, from one instance, decline to
invent a cause.

**Negative results (found the hard way, diagnosed systematically):**
Our first full FetchPush campaign — from-scratch TD3+HER, an SB3 TD3+HER
baseline, and a no-HER ablation, 1M env steps each — all flat-lined at
the eval floor (one eval seed starts in a success state; success = 0.05
throughout). Pre-declared probes: (1) a scripted controller
solves the task, so the env is fine; (2) the behavior policy moved the block
in 2% of episodes; (3) the trained actor had **saturated** — mean
|action| = 1.0, gripper pinned ~1.6 m from the block, tanh gradients dead.
Sustained ε-random exploration alone did not fix it (block moved in 0/50
episodes — a collapsed policy acts as a restoring force against per-step
random actions).

![the campaign that failed](../results/negative_result.png)

That diagnosis shaped the final design (§2) — and we then ablated it, one
setting at a time (`results/stabilizer_ablation.md`). **Exactly one of the
five is necessary.** Reverting `random_eps` alone puts the run back on the
floor for all 1M steps at 3% object contact: the original failure
reproduced by a one-line change. The other four are individually removable.
Dropping observation normalization only doubles the cost — 910k steps to a
durable 0.9 against the reported 450k — and `action_l2`, `tau` and
`expl_noise` all land inside the reported config's own three-seed spread
(410–600k), so at n=1 there is no evidence they matter alone.

Necessary is not sufficient: adding ε-random mixing to the *broken* config
did not rescue it either (block moved in 0/50). It is required, and needs
at least one of the others with it. The honest reading is that **we
over-corrected** — five changes where the evidence supports one, plus one
that buys sample efficiency. We report that rather than present the package
as if it had been tuned. One seed per arm identifies a collapse, not small
effects. The failed runs stay committed, and every figure here regenerates
from their CSVs.

**With two more weeks:** (1) the single-factor stabilizer sweep above;
(2) PickAndPlace to 3 seeds — the code path is identical, only the config
changes; (3) a vision-from-pixels variant, swapping the 13-D state for a
CNN encoder over rendered frames behind the same TD3+HER core, toward
Griffin's VLA-style perception stack; (4) domain randomization (friction,
block mass) plus actuation delay, to *quantify* the sim-to-real gap rather
than assert it.

---
*Pinned deps (`uv.lock`), seeded configs, judge path in README; `src/agent/`
(ours) is fully separated from `src/baseline/` (SB3). Any of us can walk
through the loss function or any architecture decision live (R1).*
