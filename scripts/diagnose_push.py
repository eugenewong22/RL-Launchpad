"""Diagnostic probes for the FetchPush non-learning failure (all three
arms flat at ~0.05 success over 1M steps).

Probe 1 physics   — scripted push controller: is the task solvable at all?
                    PASS: block displaces >=1cm when contacted, and the
                    scripted policy succeeds on >=30% of episodes.
Probe 2 explore   — does the block ever move under (a) zero actions,
                    (b) uniform random actions, (c) checkpoint policy
                    + Gaussian 0.1 noise (the training-time behavior
                    policy after warmup)?
                    Key number: fraction of episodes with >1cm block
                    displacement — that is the raw material HER relabels.
Probe 3 evalfloor — how many of the 20 in-training eval seeds start with
                    the block already within 5cm of the goal (predicting
                    the constant 0.05 floor).

Run: uv run python scripts/diagnose_push.py [--checkpoint path.pt]
"""

import argparse

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np

# FetchPush observation layout: grip_pos is observation[:3];
# the block position is exactly obs["achieved_goal"].


def episode_stats(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    block_start = obs["achieved_goal"].copy()
    success = 0.0
    while True:
        action = policy(obs)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            success = float(info.get("is_success", 0.0))
            break
    displacement = float(np.linalg.norm(obs["achieved_goal"] - block_start))
    return displacement, success


def scripted_push_policy(obs):
    """Classic two-phase scripted Push: get behind the block (relative to
    the goal), then push through it toward the goal."""
    grip = obs["observation"][:3]
    block = obs["achieved_goal"]
    goal = obs["desired_goal"]
    to_goal = goal - block
    dist_bg = np.linalg.norm(to_goal)
    behind = block - 0.06 * (to_goal / (dist_bg + 1e-8))
    behind[2] = block[2]

    if np.linalg.norm(grip - behind) > 0.02 and np.linalg.norm(grip - block) > 0.06:
        target = behind.copy()
        if np.linalg.norm((grip - behind)[:2]) > 0.05:
            target[2] = block[2] + 0.08  # approach above to avoid bumping the block
        delta = target - grip
    else:
        delta = goal - block  # push through the block toward the goal
    action = np.zeros(4, dtype=np.float32)
    action[:3] = np.clip(10.0 * delta, -1.0, 1.0)
    return action


def run_probe(env, name, policy, seeds, disp_threshold=0.01):
    stats = [episode_stats(env, policy, s) for s in seeds]
    disps = np.array([d for d, _ in stats])
    succ = np.array([s for _, s in stats])
    moved = float((disps > disp_threshold).mean())
    print(
        f"  {name:28s} moved>1cm: {moved:5.0%}   mean disp: {disps.mean()*100:5.1f}cm   "
        f"max disp: {disps.max()*100:5.1f}cm   success: {succ.mean():5.0%}"
    )
    return moved, float(succ.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="FetchPush-v4")
    parser.add_argument("--checkpoint", default="results/push_td3_her_seed0/checkpoint_best.pt")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    env = gym.make(args.env_id)
    seeds = list(range(500, 500 + args.episodes))

    print("== Probe 1: physics / task solvability (scripted controller) ==")
    moved, succ = run_probe(env, "scripted push", scripted_push_policy, seeds)
    print(f"  verdict: {'PASS — env physics fine, task solvable' if moved > 0.5 and succ >= 0.3 else 'FAIL — investigate env itself'}")

    print("== Probe 2: does exploration ever move the block? ==")
    rng = np.random.default_rng(0)
    run_probe(env, "do-nothing", lambda obs: np.zeros(4, dtype=np.float32), seeds)
    run_probe(
        env,
        "uniform random (warmup)",
        lambda obs: rng.uniform(-1, 1, 4).astype(np.float32),
        seeds,
    )
    try:
        from src.agent.evaluate import concat_obs
        from src.agent.td3 import TD3

        obs0, _ = env.reset(seed=0)
        state_dim = obs0["observation"].shape[0] + obs0["desired_goal"].shape[0]
        agent = TD3(state_dim, env.action_space.shape[0], max_action=float(env.action_space.high[0]))
        agent.load(args.checkpoint)
        run_probe(
            env,
            "trained policy + 0.1 noise",
            lambda obs: agent.select_action(concat_obs(obs), noise_std=0.1),
            seeds,
        )
    except FileNotFoundError:
        print(f"  (no checkpoint at {args.checkpoint}; skipping policy probe)")

    print("== Probe 3: eval-seed floor (block starts within 5cm of goal?) ==")
    free = []
    for s in range(10_000, 10_020):
        obs, _ = env.reset(seed=s)
        d = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        if d < 0.05:
            free.append(s)
    print(f"  eval seeds with success-at-reset: {free} -> floor = {len(free)}/20 = {len(free)/20:.2f}")
    env.close()


if __name__ == "__main__":
    main()
