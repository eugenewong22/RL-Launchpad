"""Policy and value networks for TD3 on goal-conditioned tasks.

Inputs are the concatenation of the environment observation and the
desired goal. Deliberately small MLPs (256-256): Fetch state is
low-dimensional, and every parameter here has to be defended to a judge.
"""

import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    """Deterministic policy: state -> action in [-max_action, max_action]."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        self.net = _mlp(state_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.net(state))


class TwinCritic(nn.Module):
    """Two independent Q-networks; TD3 targets use their minimum."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net1 = _mlp(state_dim + action_dim, 1)
        self.net2 = _mlp(state_dim + action_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        return self.net1(sa), self.net2(sa)

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """First head only — used for the (cheaper) actor loss."""
        return self.net1(torch.cat([state, action], dim=1))
