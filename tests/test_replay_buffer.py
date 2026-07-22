import numpy as np
import torch

from src.agent.replay_buffer import HerReplayBuffer

OBS_DIM, GOAL_DIM, ACT_DIM, T = 10, 3, 4, 50


def sparse_reward(achieved, goal):
    """Fetch-style: 0 if within 5cm of goal, else -1 (vectorized)."""
    d = np.linalg.norm(achieved - goal, axis=-1)
    return -(d > 0.05).astype(np.float32)


def make_buffer(her_k=4, seed=0):
    return HerReplayBuffer(
        capacity=10_000,
        obs_dim=OBS_DIM,
        goal_dim=GOAL_DIM,
        act_dim=ACT_DIM,
        episode_len=T,
        reward_fn=sparse_reward,
        her_k=her_k,
        seed=seed,
    )


def make_episode(achieved_value=0.0, desired_value=10.0, rng=None):
    """Episode whose achieved goal is constant and (by default) never
    reaches the desired goal."""
    rng = rng or np.random.default_rng(0)
    observations = rng.normal(size=(T + 1, OBS_DIM)).astype(np.float32)
    achieved = np.full((T + 1, GOAL_DIM), achieved_value, dtype=np.float32)
    desired = np.full((T, GOAL_DIM), desired_value, dtype=np.float32)
    actions = rng.uniform(-1, 1, size=(T, ACT_DIM)).astype(np.float32)
    return observations, achieved, desired, actions


def test_sample_shapes_and_state_is_obs_concat_goal():
    buf = make_buffer(her_k=0)
    obs, ach, des, act = make_episode()
    buf.add_episode(obs, ach, des, act)
    assert len(buf) == T

    state, action, reward, next_state = buf.sample(8)
    assert state.shape == (8, OBS_DIM + GOAL_DIM)
    assert action.shape == (8, ACT_DIM)
    assert reward.shape == (8, 1)
    assert next_state.shape == (8, OBS_DIM + GOAL_DIM)
    assert all(isinstance(t, torch.Tensor) for t in (state, action, reward, next_state))
    # With k=0 no relabeling happens: the goal part of every state must be
    # the stored desired goal.
    assert torch.allclose(state[:, OBS_DIM:], torch.full((8, GOAL_DIM), 10.0))


def test_no_her_on_failed_episode_gives_only_failure_rewards():
    buf = make_buffer(her_k=0)
    buf.add_episode(*make_episode(achieved_value=0.0, desired_value=10.0))
    _, _, reward, _ = buf.sample(256)
    assert torch.all(reward == -1.0)


def test_her_relabeling_turns_failures_into_successes():
    # Achieved goal is constant, so any future-achieved goal used for
    # relabeling equals the next achieved goal -> reward 0. With k=4 the
    # relabel probability is k/(k+1) = 0.8; seeded RNG makes the sampled
    # fraction deterministic.
    buf = make_buffer(her_k=4, seed=0)
    buf.add_episode(*make_episode(achieved_value=0.0, desired_value=10.0))
    _, _, reward, _ = buf.sample(512)
    frac_success = (reward == 0.0).float().mean().item()
    assert 0.7 < frac_success < 0.9, frac_success


def test_relabeled_goal_identical_in_state_and_next_state():
    buf = make_buffer(her_k=4, seed=1)
    buf.add_episode(*make_episode())
    state, _, _, next_state = buf.sample(256)
    assert torch.allclose(state[:, OBS_DIM:], next_state[:, OBS_DIM:])


def test_ring_overwrite_keeps_capacity_bounded():
    buf = make_buffer(her_k=0)
    n_episodes_capacity = 10_000 // T
    for _ in range(n_episodes_capacity + 5):
        buf.add_episode(*make_episode())
    assert len(buf) == n_episodes_capacity * T
