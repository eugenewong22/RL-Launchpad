"""TD3 (Fujimoto et al. 2018) written from scratch.

The three departures from DDPG, each attacking value overestimation or
its consequences:
  1. Clipped double-Q: targets use min(Q1', Q2') of twin critics.
  2. Target policy smoothing: clipped Gaussian noise on the target
     action regularizes the value estimate around the policy.
  3. Delayed policy updates: the actor (and both target nets) update
     once per `policy_delay` critic updates, so the actor always climbs
     a relatively settled value landscape.

No terminal flag: Fetch episodes end by time limit only, so the target
always bootstraps (equivalent to SB3's timeout handling for this task).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.agent.networks import Actor, TwinCritic


class TD3:
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        lr: float = 1e-3,
        action_l2: float = 0.0,
        device: str = "cpu",
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_delay = policy_delay
        self.action_l2 = action_l2
        self.max_action = max_action
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.actor = Actor(state_dim, act_dim, max_action).to(device)
        self.critic = TwinCritic(state_dim, act_dim).to(device)
        self.actor_target = Actor(state_dim, act_dim, max_action).to(device)
        self.critic_target = TwinCritic(state_dim, act_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.train_steps = 0

    @torch.no_grad()
    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(s).squeeze(0).cpu().numpy()
        if noise_std > 0:
            action = action + self.rng.normal(0, noise_std * self.max_action, action.shape)
        return np.clip(action, -self.max_action, self.max_action).astype(np.float32)

    def train_step(self, batch) -> dict:
        state, action, reward, next_state = (t.to(self.device) for t in batch)
        self.train_steps += 1

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            q1_t, q2_t = self.critic_target(next_state, next_action)
            target_q = reward + self.gamma * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        metrics = {"critic_loss": critic_loss.item(), "q1_mean": q1.mean().item()}

        if self.train_steps % self.policy_delay == 0:
            pi = self.actor(state)
            # L2 penalty on actions (baselines-HER `action_l2`): the
            # anti-saturation stabilizer — without it, sparse rewards let
            # the actor climb a garbage critic surface to the tanh bounds
            # and pin there with vanishing gradients.
            actor_loss = (
                -self.critic.q1(state, pi).mean()
                + self.action_l2 * (pi / self.max_action).pow(2).mean()
            )
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            metrics["actor_loss"] = actor_loss.item()

        return metrics

    @torch.no_grad()
    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module) -> None:
        for p, p_t in zip(online.parameters(), target.parameters()):
            p_t.mul_(1 - self.tau).add_(self.tau * p)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.train_steps = ckpt["train_steps"]
