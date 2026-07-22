"""Training loop: collect one episode -> HER-relabel at sample time ->
one gradient step per env step -> periodic deterministic eval + checkpoint.

Everything that touches randomness (env resets, exploration noise,
buffer sampling, network init) derives from the single --seed flag.
Eval seeds live in a disjoint range (see evaluate.py).
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

from src.agent.evaluate import concat_obs, evaluate
from src.agent.normalizer import RunningNormalizer
from src.agent.replay_buffer import HerReplayBuffer
from src.agent.td3 import TD3


@dataclass
class TrainConfig:
    env_id: str = "FetchPush-v4"
    total_env_steps: int = 1_000_000
    warmup_steps: int = 1_000  # random actions before the policy acts
    expl_noise: float = 0.1  # exploration noise std (fraction of max action)
    random_eps: float = 0.0  # prob. of a fully random action, sustained all run (HER-paper recipe)
    buffer_capacity: int = 1_000_000
    batch_size: int = 256
    her_k: int = 4  # 0 disables HER (ablation arm)
    gamma: float = 0.95
    tau: float = 0.005
    lr: float = 1e-3
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    action_l2: float = 0.0  # actor-loss L2 penalty on actions (anti-saturation)
    normalize_obs: bool = False  # running obs/goal normalization (baselines-HER style)
    eval_every: int = 5_000  # env steps between evals
    n_eval_episodes: int = 20  # in-training eval; final eval uses 50 (R4)
    eval_seed_base: int = 10_000
    seed: int = 0
    device: str = "auto"
    run_dir: str = "results/run"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def behavior_action(agent, action_space, state, expl_noise, random_eps, rng):
    """Exploration policy: with probability random_eps take a fully random
    action, else the policy's action plus Gaussian noise. Sustained random
    actions (not just warmup) are what keep generating object-contact
    episodes on contact tasks — without them the behavior policy can
    collapse to never touching the block and HER has nothing to relabel."""
    if rng.random() < random_eps:
        return action_space.sample()
    return agent.select_action(state, noise_std=expl_noise)


def run_training(cfg: TrainConfig) -> dict:
    device = cfg.resolved_device()
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f)

    env = gym.make(cfg.env_id)
    T = env.spec.max_episode_steps
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = obs["observation"].shape[0]
    goal_dim = obs["desired_goal"].shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.action_space.seed(cfg.seed)

    agent = TD3(
        state_dim=obs_dim + goal_dim,
        act_dim=act_dim,
        max_action=max_action,
        gamma=cfg.gamma,
        tau=cfg.tau,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        policy_delay=cfg.policy_delay,
        lr=cfg.lr,
        action_l2=cfg.action_l2,
        device=device,
        seed=cfg.seed,
    )
    normalizer = RunningNormalizer(dim=obs_dim + goal_dim) if cfg.normalize_obs else None
    norm = (lambda x: normalizer.normalize(x)) if normalizer else (lambda x: x)
    reward_fn = lambda ag, g: env.unwrapped.compute_reward(ag, g, {})
    buffer = HerReplayBuffer(
        capacity=cfg.buffer_capacity,
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        act_dim=act_dim,
        episode_len=T,
        reward_fn=reward_fn,
        her_k=cfg.her_k,
        seed=cfg.seed,
        device=device,
    )

    log_path = run_dir / "progress.csv"
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["env_steps", "wall_clock_s", "success_rate", "mean_return", "critic_loss"])

    explore_rng = np.random.default_rng(cfg.seed + 777)  # eps-random decisions
    env_steps = 0
    next_eval = cfg.eval_every
    best_success = -1.0
    start = time.monotonic()
    recent_losses: list[float] = []
    last_result: dict = {}

    while env_steps < cfg.total_env_steps:
        # ---- collect one episode ----
        obs, _ = env.reset(seed=cfg.seed + 1 + env_steps)  # distinct, reproducible resets
        ep_obs = np.zeros((T + 1, obs_dim), dtype=np.float32)
        ep_ach = np.zeros((T + 1, goal_dim), dtype=np.float32)
        ep_des = np.zeros((T, goal_dim), dtype=np.float32)
        ep_act = np.zeros((T, act_dim), dtype=np.float32)
        ep_obs[0], ep_ach[0] = obs["observation"], obs["achieved_goal"]

        for t in range(T):
            if env_steps < cfg.warmup_steps:
                action = env.action_space.sample()
            else:
                action = behavior_action(
                    agent, env.action_space, norm(concat_obs(obs)),
                    cfg.expl_noise, cfg.random_eps, explore_rng,
                )
            ep_des[t], ep_act[t] = obs["desired_goal"], action
            obs, _, _, _, _ = env.step(action)
            ep_obs[t + 1], ep_ach[t + 1] = obs["observation"], obs["achieved_goal"]
            env_steps += 1
        buffer.add_episode(ep_obs, ep_ach, ep_des, ep_act)
        if normalizer is not None:
            normalizer.update(np.concatenate([ep_obs[:T], ep_des], axis=1))

        # ---- one gradient step per env step, after warmup ----
        if env_steps >= cfg.warmup_steps:
            for _ in range(T):
                state, action_b, reward, next_state = buffer.sample(cfg.batch_size)
                metrics = agent.train_step((norm(state), action_b, reward, norm(next_state)))
                recent_losses.append(metrics["critic_loss"])

        # ---- periodic eval ----
        if env_steps >= next_eval or env_steps >= cfg.total_env_steps:
            policy_fn = lambda o: agent.select_action(norm(concat_obs(o)), noise_std=0.0)
            result = evaluate(policy_fn, cfg.env_id, cfg.n_eval_episodes, cfg.eval_seed_base)
            last_result = result
            wall = time.monotonic() - start
            mean_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            recent_losses.clear()
            logger.writerow(
                [env_steps, f"{wall:.1f}", result["success_rate"], result["mean_return"], mean_loss]
            )
            log_file.flush()
            print(
                f"[{cfg.env_id} seed={cfg.seed}] steps={env_steps} "
                f"success={result['success_rate']:.2f} loss={mean_loss:.4f} wall={wall:.0f}s"
            )
            agent.save(run_dir / "checkpoint_latest.pt")
            if normalizer is not None:
                normalizer.save(run_dir / "normalizer.npz")
            if result["success_rate"] >= best_success:
                best_success = result["success_rate"]
                agent.save(run_dir / "checkpoint_best.pt")
            while next_eval <= env_steps:
                next_eval += cfg.eval_every

    env.close()
    log_file.close()
    return last_result


def main():
    parser = argparse.ArgumentParser(description="Train from-scratch TD3+HER")
    parser.add_argument("--config", required=True, help="YAML file of TrainConfig overrides")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--run-dir", default=None, help="override config run_dir")
    args = parser.parse_args()

    with open(args.config) as f:
        overrides = yaml.safe_load(f) or {}
    cfg = TrainConfig(**overrides)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_dir is not None:
        cfg.run_dir = args.run_dir
    elif args.seed is not None:
        cfg.run_dir = f"{cfg.run_dir}_seed{cfg.seed}"
    run_training(cfg)


if __name__ == "__main__":
    main()
