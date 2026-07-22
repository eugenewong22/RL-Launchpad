import gymnasium as gym
import numpy as np

from src.agent.train import behavior_action


class StubAgent:
    """Always proposes the same fixed action; records the noise it got."""

    def __init__(self):
        self.fixed = np.full(4, 0.123, dtype=np.float32)
        self.last_noise = None

    def select_action(self, state, noise_std=0.0):
        self.last_noise = noise_std
        return self.fixed


def make_space(seed=0):
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    space.seed(seed)
    return space


def test_eps_zero_returns_policy_action_with_configured_noise():
    agent = StubAgent()
    action = behavior_action(
        agent, make_space(), state=np.zeros(5), expl_noise=0.2, random_eps=0.0,
        rng=np.random.default_rng(0),
    )
    assert np.array_equal(action, agent.fixed)
    assert agent.last_noise == 0.2


def test_eps_one_returns_uniform_random_actions_in_bounds():
    agent = StubAgent()
    actions = [
        behavior_action(
            agent, make_space(i), state=np.zeros(5), expl_noise=0.2, random_eps=1.0,
            rng=np.random.default_rng(i),
        )
        for i in range(20)
    ]
    assert agent.last_noise is None  # the policy is never consulted
    assert all(np.all(np.abs(a) <= 1.0) for a in actions)
    # 20 uniform draws are essentially never all equal to the stub's action
    assert not any(np.array_equal(a, agent.fixed) for a in actions)


def test_eps_fraction_mixes_both_sources():
    agent = StubAgent()
    rng = np.random.default_rng(42)
    space = make_space()
    actions = [
        behavior_action(agent, space, np.zeros(5), 0.2, random_eps=0.3, rng=rng)
        for _ in range(500)
    ]
    n_policy = sum(np.array_equal(a, agent.fixed) for a in actions)
    assert 0.6 < n_policy / 500 < 0.8  # ~70% policy actions at eps=0.3
