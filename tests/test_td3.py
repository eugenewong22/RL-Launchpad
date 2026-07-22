import copy

import numpy as np
import torch

from src.agent.td3 import TD3

STATE_DIM, ACT_DIM = 13, 4


def make_agent(**kwargs):
    defaults = dict(state_dim=STATE_DIM, act_dim=ACT_DIM, max_action=1.0, seed=0)
    defaults.update(kwargs)
    return TD3(**defaults)


def make_batch(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(n, STATE_DIM, generator=g)
    action = torch.rand(n, ACT_DIM, generator=g) * 2 - 1
    reward = -(torch.rand(n, 1, generator=g) > 0.2).float()
    next_state = state + 0.01 * torch.randn(n, STATE_DIM, generator=g)
    return state, action, reward, next_state


def _params_equal(a, b):
    return all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))


def test_targets_start_as_copies_of_online_networks():
    agent = make_agent()
    assert _params_equal(agent.actor, agent.actor_target)
    assert _params_equal(agent.critic, agent.critic_target)


def test_select_action_is_bounded_and_deterministic_without_noise():
    agent = make_agent()
    state = np.random.default_rng(0).normal(size=STATE_DIM).astype(np.float32)
    a1 = agent.select_action(state, noise_std=0.0)
    a2 = agent.select_action(state, noise_std=0.0)
    assert a1.shape == (ACT_DIM,)
    assert np.array_equal(a1, a2)
    assert np.all(np.abs(a1) <= 1.0)
    a3 = agent.select_action(state, noise_std=0.5)
    assert np.all(np.abs(a3) <= 1.0)  # noise must not break bounds


def test_critic_loss_decreases_on_fixed_batch():
    agent = make_agent()
    batch = make_batch()
    first = agent.train_step(batch)["critic_loss"]
    for _ in range(200):
        last = agent.train_step(batch)["critic_loss"]
    assert last < first / 2, (first, last)


def test_actor_update_is_delayed():
    agent = make_agent(policy_delay=2)
    batch = make_batch()
    actor_before = copy.deepcopy(agent.actor)

    agent.train_step(batch)  # step 1: critic only
    assert _params_equal(agent.actor, actor_before)

    agent.train_step(batch)  # step 2: actor + targets update
    assert not _params_equal(agent.actor, actor_before)


def test_target_networks_polyak_track_online_networks():
    tau = 0.005
    agent = make_agent(policy_delay=1, tau=tau)  # targets update every step
    critic_target_before = copy.deepcopy(agent.critic_target)

    agent.train_step(make_batch())

    for p_new, p_old, p_online in zip(
        agent.critic_target.parameters(),
        critic_target_before.parameters(),
        agent.critic.parameters(),
    ):
        expected = tau * p_online + (1 - tau) * p_old
        assert torch.allclose(p_new, expected, atol=1e-6)


def test_save_load_roundtrip(tmp_path):
    agent = make_agent()
    agent.train_step(make_batch())
    path = tmp_path / "ckpt.pt"
    agent.save(path)

    fresh = make_agent(seed=99)
    assert not _params_equal(fresh.actor, agent.actor)
    fresh.load(path)
    assert _params_equal(fresh.actor, agent.actor)
    assert _params_equal(fresh.critic, agent.critic)
    assert _params_equal(fresh.critic_target, agent.critic_target)
