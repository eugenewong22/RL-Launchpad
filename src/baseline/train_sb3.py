"""SB3 TD3 + HER baseline (R2). This is the *published* baseline the
from-scratch agent is measured against — none of this code is part of
the submission's algorithm.

Hyperparameters mirror the from-scratch config (which itself follows
SB3's tuned Fetch values), and evaluation goes through the SAME
`src.agent.evaluate.evaluate` protocol, so curves are comparable
point-for-point: same eval seeds, same success metric, same x-axis.
"""

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import yaml
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her import HerReplayBuffer

from src.agent.evaluate import evaluate


@dataclass
class BaselineConfig:
    env_id: str = "FetchPush-v4"
    total_env_steps: int = 1_000_000
    her_k: int = 4
    gamma: float = 0.95
    tau: float = 0.005
    lr: float = 1e-3
    batch_size: int = 256
    buffer_capacity: int = 1_000_000
    expl_noise: float = 0.1
    learning_starts: int = 1_000
    eval_every: int = 10_000
    n_eval_episodes: int = 20
    eval_seed_base: int = 10_000
    seed: int = 0
    run_dir: str = "results/push_sb3_her"


class CsvEvalCallback(BaseCallback):
    """Periodic eval through our shared protocol, logged in our CSV schema."""

    def __init__(self, cfg: BaselineConfig, log_path: Path):
        super().__init__()
        self.cfg = cfg
        self.log_file = open(log_path, "w", newline="")
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(
            ["env_steps", "wall_clock_s", "success_rate", "mean_return", "critic_loss"]
        )
        self.start = time.monotonic()
        self.next_eval = cfg.eval_every

    def _eval_and_log(self) -> None:
        policy_fn = lambda obs: self.model.predict(obs, deterministic=True)[0]
        result = evaluate(policy_fn, self.cfg.env_id, self.cfg.n_eval_episodes, self.cfg.eval_seed_base)
        wall = time.monotonic() - self.start
        critic_loss = self.model.logger.name_to_value.get("train/critic_loss", float("nan"))
        self.writer.writerow(
            [self.num_timesteps, f"{wall:.1f}", result["success_rate"], result["mean_return"], critic_loss]
        )
        self.log_file.flush()
        print(
            f"[SB3 {self.cfg.env_id} seed={self.cfg.seed}] steps={self.num_timesteps} "
            f"success={result['success_rate']:.2f} wall={wall:.0f}s"
        )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self._eval_and_log()
            while self.next_eval <= self.num_timesteps:
                self.next_eval += self.cfg.eval_every
        return True

    def _on_training_end(self) -> None:
        self._eval_and_log()  # final point at total_env_steps
        self.log_file.close()


def run_baseline(cfg: BaselineConfig) -> None:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f)

    env = gym.make(cfg.env_id)
    n_actions = env.action_space.shape[0]
    model = TD3(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=cfg.her_k, goal_selection_strategy="future"),
        buffer_size=cfg.buffer_capacity,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        tau=cfg.tau,
        learning_rate=cfg.lr,
        learning_starts=cfg.learning_starts,
        action_noise=NormalActionNoise(np.zeros(n_actions), cfg.expl_noise * np.ones(n_actions)),
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=cfg.seed,
        verbose=0,
    )
    callback = CsvEvalCallback(cfg, run_dir / "progress.csv")
    model.learn(total_timesteps=cfg.total_env_steps, callback=callback)
    model.save(run_dir / "checkpoint_latest")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train the SB3 TD3+HER baseline")
    parser.add_argument("--config", required=True, help="YAML file of BaselineConfig overrides")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        overrides = yaml.safe_load(f) or {}
    cfg = BaselineConfig(**overrides)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_dir is not None:
        cfg.run_dir = args.run_dir
    elif args.seed is not None:
        cfg.run_dir = f"{cfg.run_dir}_seed{cfg.seed}"
    run_baseline(cfg)


if __name__ == "__main__":
    main()
