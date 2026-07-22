"""Episode-structured replay buffer with Hindsight Experience Replay.

HER ('future' strategy, Andrychowicz et al. 2017): when sampling a
transition at step t, with probability her_k/(her_k+1) replace its
desired goal with the achieved goal of a random step t' in (t, T],
then recompute the reward against the substituted goal. her_k=0
disables relabeling entirely (the ablation arm).

Episodes have fixed length (Fetch: 50 steps), so storage is a ring of
episode slots: observations/achieved goals are (T+1, .) per episode so
every transition has its next state available in-slot.
"""

from typing import Callable

import numpy as np
import torch


class HerReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        goal_dim: int,
        act_dim: int,
        episode_len: int,
        reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        her_k: int = 4,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.T = episode_len
        self.max_episodes = capacity // episode_len
        self.reward_fn = reward_fn
        self.her_k = her_k
        self.rng = np.random.default_rng(seed)
        self.device = device

        E, T = self.max_episodes, episode_len
        self.observations = np.zeros((E, T + 1, obs_dim), dtype=np.float32)
        self.achieved = np.zeros((E, T + 1, goal_dim), dtype=np.float32)
        self.desired = np.zeros((E, T, goal_dim), dtype=np.float32)
        self.actions = np.zeros((E, T, act_dim), dtype=np.float32)
        self.n_stored = 0  # episodes currently valid
        self.write_idx = 0  # next episode slot to overwrite

    def __len__(self) -> int:
        return self.n_stored * self.T

    def add_episode(
        self,
        observations: np.ndarray,  # (T+1, obs_dim)
        achieved: np.ndarray,  # (T+1, goal_dim)
        desired: np.ndarray,  # (T, goal_dim)
        actions: np.ndarray,  # (T, act_dim)
    ) -> None:
        i = self.write_idx
        self.observations[i] = observations
        self.achieved[i] = achieved
        self.desired[i] = desired
        self.actions[i] = actions
        self.write_idx = (i + 1) % self.max_episodes
        self.n_stored = min(self.n_stored + 1, self.max_episodes)

    def sample(self, batch_size: int):
        ep = self.rng.integers(0, self.n_stored, size=batch_size)
        t = self.rng.integers(0, self.T, size=batch_size)

        goals = self.desired[ep, t].copy()
        if self.her_k > 0:
            relabel = self.rng.random(batch_size) < self.her_k / (self.her_k + 1)
            # Future achieved goal: uniform t' in (t, T] per transition.
            future_t = self.rng.integers(t + 1, self.T + 1)
            idx = np.where(relabel)[0]
            goals[idx] = self.achieved[ep[idx], future_t[idx]]

        rewards = self.reward_fn(self.achieved[ep, t + 1], goals).astype(np.float32)
        states = np.concatenate([self.observations[ep, t], goals], axis=1)
        next_states = np.concatenate([self.observations[ep, t + 1], goals], axis=1)

        to = lambda x: torch.as_tensor(x, device=self.device)
        return (
            to(states),
            to(self.actions[ep, t]),
            to(rewards).reshape(-1, 1),
            to(next_states),
        )
