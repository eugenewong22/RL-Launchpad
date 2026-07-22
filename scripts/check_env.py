"""Sanity-check the simulator stack: random agent on a Fetch env.

Run: uv run python scripts/check_env.py [env_id]
Verifies the goal-conditioned obs dict and the compute_reward API that
HER relabeling depends on. Exits non-zero on any mismatch.
"""

import sys

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers Fetch envs)


def main(env_id: str = "FetchReach-v4") -> None:
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)

    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}, obs.keys()
    print(f"{env_id}")
    print(f"  observation:   {obs['observation'].shape}")
    print(f"  achieved_goal: {obs['achieved_goal'].shape}")
    print(f"  desired_goal:  {obs['desired_goal'].shape}")
    print(f"  action space:  {env.action_space.shape}, bounds ±{env.action_space.high[0]}")

    total_reward = 0.0
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    # HER needs to recompute rewards for relabeled goals.
    r_same = env.unwrapped.compute_reward(obs["achieved_goal"], obs["achieved_goal"], {})
    r_far = env.unwrapped.compute_reward(obs["achieved_goal"], obs["achieved_goal"] + 1.0, {})
    assert r_same == 0.0, f"success reward should be 0.0, got {r_same}"
    assert r_far == -1.0, f"failure reward should be -1.0, got {r_far}"
    print(f"  sparse reward OK: success={r_same}, failure={r_far}")
    print(f"  200 random steps OK (sum of rewards: {total_reward:.0f})")
    env.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
