# Live-Walkthrough Prep (R1)

*Judges will ask any team member to explain (a) the loss function and
(b) one design decision in the network architecture. If nobody can, the
entry loses its algorithm credit. Everything below maps to specific
lines in `src/agent/td3.py` and `src/agent/networks.py`.*

## The critic loss, line by line

```python
noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
next_action = (actor_target(next_state) + noise).clamp(-max_action, max_action)
q1_t, q2_t = critic_target(next_state, next_action)
target_q = reward + gamma * torch.min(q1_t, q2_t)
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
actor_loss = -critic.q1(state, actor(state)).mean()
```

*"Deterministic policy gradient: push actions uphill on the critic's
value surface. We use only Q1 — using min(Q1,Q2) here buys little
(the pessimism matters for **targets**, not for the ascent direction)
and costs a second forward pass. The actor updates every second critic
update ('delayed'), so it always climbs a partially-converged, less
exploitable landscape."*

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

## Honest numbers to have ready

- Eval protocol: 50 episodes, seeds 10000–10049, disjoint from training;
  deterministic policy; success = env's `is_success` at episode end.
- FetchReach: 98% (50-ep protocol) — and the 10-ep in-training eval said
  100%, a live example of why R4 mandates ≥50 episodes.
- FetchPush numbers: see `results/final_eval.md` once the matrix lands.
