"""The epsilon-random override must actually fire, at the right rate, and
produce actions SB3's buffer can consume.

Written before the 45-minute training run: a silently-inactive override
would produce a null result indistinguishable from "the change doesn't
help", which is exactly the wrong conclusion to draw.
"""

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers Fetch env IDs)
import numpy as np
import pytest
from stable_baselines3.her import HerReplayBuffer

from src.baseline.td3_eps_random import TD3EpsRandom

ENV_ID = "FetchPush-v4"


def make_model(random_eps, seed=0):
    env = gym.make(ENV_ID)
    return TD3EpsRandom(
        "MultiInputPolicy",
        env,
        random_eps=random_eps,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future"},
        learning_starts=10,
        seed=seed,
        verbose=0,
    )


@pytest.mark.parametrize("eps", [0.0, 0.3, 1.0])
def test_random_action_rate_matches_epsilon(eps):
    """Fires at approximately the configured rate, past warmup."""
    model = make_model(eps)
    model._last_obs = model.env.reset()
    model.num_timesteps = 10_000  # well past learning_starts

    n = 2000
    for _ in range(n):
        model._sample_action(learning_starts=10, action_noise=None, n_envs=1)

    rate = model.n_random_actions / n
    assert abs(rate - eps) < 0.05, f"eps={eps} but observed rate {rate:.3f}"


def test_no_override_during_warmup():
    """SB3's warmup is already uniform-random; we must not double-count it."""
    model = make_model(1.0)
    model._last_obs = model.env.reset()
    model.num_timesteps = 0  # inside learning_starts

    for _ in range(50):
        model._sample_action(learning_starts=1000, action_noise=None, n_envs=1)
    assert model.n_random_actions == 0


def test_actions_are_valid_and_correctly_scaled():
    """The env action and the buffer action must each be in their own space.

    Swapping scale/unscale here would corrupt the replay buffer silently —
    training would run to completion and simply learn nothing.
    """
    model = make_model(1.0)  # force the override on every call
    model._last_obs = model.env.reset()
    model.num_timesteps = 10_000
    space = model.action_space

    for _ in range(200):
        action, buffer_action = model._sample_action(10, None, 1)
        assert space.contains(np.asarray(action[0], dtype=space.dtype)), action
        # SB3 stores actions scaled to [-1, 1].
        assert np.all(buffer_action >= -1.0) and np.all(buffer_action <= 1.0), buffer_action


def test_matches_our_agents_semantics():
    """Same rule as src/agent/train.behavior_action: uniform from the action
    space, not noise-scaled policy output."""
    model = make_model(1.0)
    model._last_obs = model.env.reset()
    model.num_timesteps = 10_000

    samples = np.array([model._sample_action(10, None, 1)[0][0] for _ in range(400)])
    # A uniform draw over [-1, 1]^4 has mean ~0 and spread near the bounds;
    # a saturated-policy draw would clump at the edges instead.
    assert abs(samples.mean()) < 0.1, samples.mean()
    assert samples.max() > 0.85 and samples.min() < -0.85


def test_override_is_live_during_learn():
    """The override must fire inside model.learn(), not just when called.

    This is what makes a NEGATIVE result trustworthy. The experiment found
    that adding epsilon-random to SB3's TD3 does not rescue it — a flat
    curve. But a silently-inactive override produces an identical flat
    curve, and the two conclusions are opposite: "the mechanism does not
    help" vs "we never tested the mechanism". The unit tests above call
    _sample_action directly; only this one proves SB3's training loop
    actually routes through it.
    """
    model = make_model(0.3)
    warmup = 200
    model.learning_starts = warmup
    total = 1200
    model.learn(total_timesteps=total)

    assert model.n_random_actions > 0, (
        "the override never fired during learn() — any conclusion drawn from "
        "the training curves would be about an experiment that did not run"
    )
    rate = model.n_random_actions / (total - warmup)
    assert abs(rate - 0.3) < 0.08, f"fired at {rate:.3f}, configured 0.300"
