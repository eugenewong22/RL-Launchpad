import torch

from src.agent.networks import Actor, TwinCritic

OBS_DIM, GOAL_DIM, ACT_DIM, MAX_ACTION = 10, 3, 4, 1.0
BATCH = 32


def _inputs():
    obs = torch.randn(BATCH, OBS_DIM + GOAL_DIM)
    act = torch.rand(BATCH, ACT_DIM) * 2 - 1
    return obs, act


def test_actor_outputs_bounded_actions_of_correct_shape():
    actor = Actor(OBS_DIM + GOAL_DIM, ACT_DIM, max_action=MAX_ACTION)
    obs, _ = _inputs()
    actions = actor(obs)
    assert actions.shape == (BATCH, ACT_DIM)
    assert torch.all(actions <= MAX_ACTION) and torch.all(actions >= -MAX_ACTION)


def test_twin_critic_returns_two_independent_q_estimates():
    critic = TwinCritic(OBS_DIM + GOAL_DIM, ACT_DIM)
    obs, act = _inputs()
    q1, q2 = critic(obs, act)
    assert q1.shape == (BATCH, 1) and q2.shape == (BATCH, 1)
    # Independently initialized heads must disagree on random inputs,
    # otherwise min(q1, q2) degenerates to a single critic.
    assert not torch.allclose(q1, q2)


def test_critic_q1_matches_first_head():
    critic = TwinCritic(OBS_DIM + GOAL_DIM, ACT_DIM)
    obs, act = _inputs()
    q1, _ = critic(obs, act)
    assert torch.allclose(critic.q1(obs, act), q1)
