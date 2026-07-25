# Live-Walkthrough Prep (R1)

*Judges will ask any team member to explain (a) the loss function and
(b) one design decision in the network architecture. If nobody can, the
entry loses its algorithm credit. Everything below maps to specific
lines in `src/agent/td3.py` and `src/agent/networks.py`.*

## The critic loss, line by line

```python
with torch.no_grad():                      # the target is a fixed regression
    noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
    next_action = (actor_target(next_state) + noise).clamp(-max_action, max_action)
    q1_t, q2_t = critic_target(next_state, next_action)
    target_q = reward + gamma * torch.min(q1_t, q2_t)

q1, q2 = critic(state, action)
critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
```

Spoken version: *"The critics learn to satisfy the Bellman equation: the
value of (state, action) should equal the reward plus the discounted
value of what the target policy does next. Three TD3-specific choices
guard that target. First, the next action comes from a slowly-moving
**target actor**, plus clipped Gaussian noise — 'target policy
smoothing' — so the critic can't exploit a sharp spike in its own value
landscape at exactly the policy's action. Second, we take the
**minimum of two independently-initialized critics**, because a single
maximizing critic systematically overestimates (Jensen's inequality:
max of noisy estimates is biased upward); the min turns that bias
pessimistic, which is stable. Third, the whole target is inside
`no_grad` — it's a fixed regression target, not something to backprop
through."*

Follow-ups to expect:
- **Why MSE and not Huber?** Rewards are bounded in [−1, 0] and γ=0.95
  bounds Q in [−20, 0], so there are no outlier targets to clip; MSE is
  the maximum-likelihood choice under Gaussian TD noise.
- **Why no terminal mask?** Fetch episodes end only by time limit —
  truncation, not environment death. Bootstrapping through the cutoff is
  the correct value of the continuing task (same reasoning as SB3's
  timeout handling).
- **Where does HER enter the loss?** It doesn't — and that's the elegant
  part. HER only edits the *batch* (goal component of state, and the
  recomputed reward). The TD3 update is completely unaware relabeling
  happened.

## The actor loss

```python
pi = actor(state)
actor_loss = -critic.q1(state, pi).mean() + action_l2 * (pi / max_action).pow(2).mean()
```

*"Deterministic policy gradient: push actions uphill on the critic's
value surface. We use only Q1 — using min(Q1,Q2) here buys little
(the pessimism matters for **targets**, not for the ascent direction)
and costs a second forward pass. The actor updates every second critic
update ('delayed'), so it always climbs a partially-converged, less
exploitable landscape. The L2 term penalizes action magnitude, the
`action_l2` stabilizer from the reference HER implementation. We added it
after measuring the actor saturate — mean |action| = 1.0, gripper parked
1.6 m from the block — on the theory that it was the counter-pressure to
that. Be careful here if asked: our own ablation
(`results/stabilizer_ablation.md`) shows removing it alone costs nothing
measurable, so it is not what rescues the run. The necessary one turned
out to be sustained ε-random exploration."*

## Architecture decisions (pick any one)

1. **Goal-concatenated input (13 = 10 obs + 3 goal), not a separate
   goal encoder.** The goal is 3 numbers in the same Cartesian frame as
   the state; a learned embedding would add parameters to learn the
   identity function. UVFA-style concatenation is the whole trick that
   makes one network represent a *family* of tasks.
2. **256-256 MLPs.** Input is 13-D and smooth; capacity is not the
   bottleneck — stability under bootstrapped targets is. This matches
   the SB3 baseline's width, which keeps the R2 comparison about the
   *algorithm*, not the architecture.
3. **tanh output scaling on the actor.** Action space is [−1,1]⁴;
   tanh makes bounds differentiable and saturating, versus clipping
   which zeroes gradients exactly where exploration pushes hardest.
4. **Twin critics share nothing.** Two truly independent MLPs (separate
   init, separate gradients) — weight sharing would correlate their
   errors and defeat the min's de-biasing purpose (this is what
   `test_twin_critic_returns_two_independent_q_estimates` pins down).

## HER in one whiteboard sketch

*"An episode that pushed the block to the wrong place is a failure for
the commanded goal but a **perfect demonstration** for the goal 'where
the block actually went'. At sample time we swap the goal for an
achieved goal from a later step of the same episode (probability
k/(k+1)=0.8, k=4) and recompute the reward — instantly turning a
reward-free replay buffer into one dense with successes. The relabeled
goal must be identical in state and next_state, and the reward is
recomputed from the *next* achieved goal — both are unit-tested."*

## The hard questions (rehearse these, not the easy ones)

These are the questions our own results invite. Each has a short honest
answer; none of them has a spin that survives a follow-up.

**"Your SB3 TD3+HER baseline scored 0.05 and you scored ~1.0. Isn't that
too good to be true?"**
*"Yes, and we don't claim it. That arm is a failed baseline configuration,
not a scalp. Two things make it non-suspicious: SB3's SAC+HER solves the
task at ~1.0 under the exact same harness, so the env, the eval protocol
and the observation/action spaces are all fine — and our own TD3+HER was
equally flat at those same hyperparameters until we changed five settings.
Two independent implementations failing identically points at the config,
not at either codebase. Our fair comparison is SAC+HER, and against it we
match rather than beat."*

**"Then would your five changes fix SB3's TD3?"**
*"We don't know — we never ran it. The changes live in our training loop
and porting them into SB3 was out of scope. Saying anything stronger would
be claiming an experiment we didn't do."*

**"You changed five things at once. Which one mattered?"**
*"We ran that ablation — `results/stabilizer_ablation.md`, one arm per
setting. Exactly one is necessary: sustained ε-random mixing. Revert
`random_eps` to 0 and the run sits on the floor for all 1M steps at 3%
contact, which is the original failure exactly. The other four you can
remove individually and it still solves the task. Dropping observation
normalization doubles the steps to a durable 0.9, 910k against 450k, but
still gets there. So we over-corrected — and we say that rather than
pretend the package was tuned."*

**"Then was your original diagnosis wrong?"**
*"Partly, and we corrected the write-up. We'd described action-L2 as the
direct counter-pressure to saturation; the ablation says removing it alone
costs nothing measurable. What survives is narrower and better supported:
the collapse is an exploration failure, and sustained random actions are
what prevent it. Note it's necessary but not sufficient — adding ε-random
to the broken config didn't rescue it either, so it needs at least one of
the others alongside."*

**"τ=0.05 is 10× the usual value. Justify it."**
*"It's the value the reference HER implementations use for Fetch, not
something we tuned into — and honestly, our ablation says it isn't doing
much: reverting it to 0.005 still solves the task at 390k steps, inside our
own seed spread. We kept it because it matches the reference, not because
we can show it earns its place."*

**"How do you know HER is what's doing the work, and not the stabilizers?"**
*"Because the no-HER ablation has all five stabilizers and still never
leaves the floor — `her_k=0` is the only difference. We log `contact_frac`:
the HER arms touch the block in 95% of episodes, the no-HER arm in 4–7%.
That's in the committed CSVs, not inferred."*

**"Anything you'd do differently?"**
*"Pick SAC. A deterministic actor is exactly what saturates under sparse
reward; SAC's entropy term makes it structurally immune, which is why it's
the one out-of-the-box baseline that worked. We chose TD3 for fewer moving
parts and paid for it in three stabilizers and a diagnosis campaign."*

## Honest numbers to have ready

- Eval protocol: 50 episodes, seeds 10000–10049, disjoint from training;
  deterministic policy; success = env's `is_success` at episode end.
- FetchReach: 98% (50-ep protocol) — and the 10-ep in-training eval said
  100%, a live example of why R4 mandates ≥50 episodes.
- FetchPush, R4 protocol (`results/final_eval_push.md` — quote the file,
  not memory): **ours 0.993 ± 0.009** (1.00, 1.00, 0.98) and **SAC+HER
  0.993 ± 0.009** (1.00, 0.98, 1.00) — an exact tie, differing only in
  which seed drops one episode. SB3 TD3+HER and the no-HER ablation both
  sit at the floor, 0.040 ± 0.000.
- The floor is 0.040 on the R4 protocol and 0.05 in the figures because
  it's 2/50 versus 1/20 — the same ~4% of initial states that start with
  the block already inside the 5 cm threshold. If a judge spots the two
  numbers, that's the answer.
- FetchPickAndPlace: 0.740, seed 0, same config, no re-tuning.
- Robustness: 199/200 held-out non-eval initial states
  (`results/failure_sweep.md`). If asked about the one failure: contact
  without sustained pushing, 17% of the required displacement, and we
  decline to attribute a cause from a single sample.
