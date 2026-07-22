"""Render evaluation episodes from a checkpoint to MP4 (R4: the video
comes from the same checkpoint the reported numbers come from, on the
same eval seeds).

Usage:
  uv run python scripts/record_video.py --checkpoint results/push_td3_her_seed0/checkpoint_best.pt \
      --env-id FetchPush-v4 --episodes 5 --out results/demo.mp4
"""

import argparse

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import imageio
import numpy as np

from src.agent.evaluate import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="FetchPush-v4")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-seed-base", type=int, default=10_000)
    parser.add_argument("--out", default="results/demo.mp4")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    env = gym.make(args.env_id, render_mode="rgb_array")
    policy_fn = load_policy(args.checkpoint, args.env_id)

    frames, successes = [], []
    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.eval_seed_base + i)
        while True:
            frames.append(env.render())
            action = policy_fn(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                successes.append(float(info.get("is_success", 0.0)))
                break

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(
        f"wrote {args.out}: {args.episodes} episodes, "
        f"success {np.mean(successes):.2f}, seeds {args.eval_seed_base}..{args.eval_seed_base + args.episodes - 1}"
    )
    env.close()


if __name__ == "__main__":
    main()
