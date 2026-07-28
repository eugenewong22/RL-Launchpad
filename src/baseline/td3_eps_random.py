"""SB3's TD3 with our sustained epsilon-random exploration added.

This is the experiment `docs/writeup.md` §3 explicitly declined to claim.
Our ablation (`results/stabilizer_ablation.md`) shows sustained ε-random
mixing is the one setting of five whose removal collapses our own agent.
SB3's TD3+HER also fails on FetchPush at its published settings — and it
has no such knob. This tests whether that single missing mechanism is
enough to explain SB3's failure.

It is deliberately the ONLY change. Everything else stays at SB3's
published values, so a difference is attributable to this and nothing else.
Reported as a *modified baseline*, never as "the SB3 baseline".

The semantics match `src/agent/train.behavior_action` exactly: with
probability `random_eps` take a uniform action from the action space,
otherwise the policy's action with SB3's usual action noise applied. Any
other reading would be testing a lookalike rather than our change.
"""

import numpy as np
from gymnasium import spaces
from stable_baselines3 import TD3


class TD3EpsRandom(TD3):
    """TD3 with a sustained probability of a fully random action.

    SB3 takes uniform random actions only during `learning_starts`, then
    switches to policy-plus-noise for the rest of training. Ours keeps
    mixing them in at a fixed rate for the whole run — the difference this
    class exists to isolate.
    """

    def __init__(self, *args, random_eps: float = 0.3, **kwargs):
        # Set before super().__init__, which may sample actions during setup.
        self.random_eps = random_eps
        super().__init__(*args, **kwargs)
        # Own RNG, seeded off the model seed, so exploration draws are
        # reproducible and independent of SB3's internal streams.
        self._eps_rng = np.random.default_rng(self.seed if self.seed is not None else 0)
        self.n_random_actions = 0

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)

        # Warmup is already fully random; overriding there would double-count
        # and change SB3's warmup semantics rather than add to them.
        if self.num_timesteps < learning_starts:
            return action, buffer_action
        if self._eps_rng.random() >= self.random_eps:
            return action, buffer_action

        self.n_random_actions += 1
        unscaled = np.array([self.action_space.sample() for _ in range(n_envs)])
        if isinstance(self.action_space, spaces.Box):
            # Mirror SB3's own scaling path: the buffer stores the scaled
            # action, the env receives the unscaled one. Getting this
            # backwards silently corrupts the replay buffer.
            buffer_action = self.policy.scale_action(unscaled)
            action = self.policy.unscale_action(buffer_action)
        else:
            action = buffer_action = unscaled
        return action, buffer_action
