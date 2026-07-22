"""R4 evaluation protocol: fixed, enumerable eval seeds disjoint from
training seeds; deterministic policy; success = env's own is_success
flag at episode end.

Eval seed for episode i is eval_seed_base + i. Training uses seeds in
[0, ~1000), eval bases start at 10_000, so the sets never overlap.
"""

import argparse

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np


def concat_obs(obs: dict) -> np.ndarray:
    return np.concatenate([obs["observation"], obs["desired_goal"]], dtype=np.float32)


def load_policy(checkpoint_path, env_id: str):
    """Build a deterministic policy_fn from a from-scratch checkpoint,
    loading the run's normalizer.npz (if present, from the same
    directory) — normalization stats are part of the learned artifact."""
    from pathlib import Path

    from src.agent.td3 import TD3

    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    state_dim = obs["observation"].shape[0] + obs["desired_goal"].shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.close()

    agent = TD3(state_dim, act_dim, max_action=max_action)
    agent.load(checkpoint_path)

    norm_path = Path(checkpoint_path).parent / "normalizer.npz"
    if norm_path.exists():
        from src.agent.normalizer import RunningNormalizer

        normalizer = RunningNormalizer.load(norm_path)
        return lambda obs: agent.select_action(
            normalizer.normalize(concat_obs(obs)), noise_std=0.0
        )
    return lambda obs: agent.select_action(concat_obs(obs), noise_std=0.0)


def evaluate(policy_fn, env_id: str, n_episodes: int, eval_seed_base: int) -> dict:
    """policy_fn maps the raw goal-conditioned obs dict to an action.

    Taking a bare function (not an agent class) lets the from-scratch
    agent and the SB3 baseline run through this exact code path.
    """
    env = gym.make(env_id)
    successes, returns, seeds = [], [], []
    for i in range(n_episodes):
        seed = eval_seed_base + i
        seeds.append(seed)
        obs, _ = env.reset(seed=seed)
        ep_return, success = 0.0, 0.0
        while True:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            if terminated or truncated:
                success = float(info.get("is_success", 0.0))
                break
        successes.append(success)
        returns.append(ep_return)
    env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "n_episodes": n_episodes,
        "episode_seeds": tuple(seeds),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint (R4 protocol)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="FetchPush-v4")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-seed-base", type=int, default=10_000)
    args = parser.parse_args()

    policy_fn = load_policy(args.checkpoint, args.env_id)
    result = evaluate(policy_fn, args.env_id, args.episodes, args.eval_seed_base)
    print(
        f"{args.env_id}: success_rate={result['success_rate']:.3f} "
        f"mean_return={result['mean_return']:.1f} over {result['n_episodes']} episodes "
        f"(eval seeds {result['episode_seeds'][0]}..{result['episode_seeds'][-1]})"
    )


if __name__ == "__main__":
    main()
