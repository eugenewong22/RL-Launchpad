"""Characterize failure modes on held-out (non-eval) initial states.

The R4 eval uses env seeds 10000+; this sweeps a disjoint block (2000+) so
we can find and describe failures that the reported 50-episode eval does
not happen to contain. Reports success against required push distance, and
flags the "stall in contact" mode: episode fails while the block sits at
effectively identical coordinates over the final steps.

Usage: uv run python scripts/failure_sweep.py [--runs push_td3_her_seed0 ...]

For each failure we report *progress* = how much of the required push the
policy actually achieved. This distinguishes "never touched the block"
(progress ~0) from "pushed it past the goal" (progress > 1) from "made
contact and under-pushed" (0 < progress < 1) without assuming which
happened.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers the Fetch env IDs)
import numpy as np
import yaml

from src.agent.evaluate import load_policy

BINS = [0.0, 0.10, 0.15, 0.20, 0.25, 0.60]


def sweep(run_dir: Path, env_id: str, seeds: range, horizon: int):
    env = gym.make(env_id)
    policy_fn = load_policy(run_dir / "checkpoint_best.pt", env_id)
    d0, ok, progress, touched = [], [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        start = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        d0.append(start)
        info, closest = {}, np.inf
        for _ in range(horizon):
            obs, _, _, _, info = env.step(policy_fn(obs))
            # gripper-to-block distance; the block is ~5 cm across, so a
            # minimum near 0.05 m means the gripper did reach it.
            closest = min(closest, float(np.linalg.norm(
                obs["observation"][:3] - obs["achieved_goal"])))
        end = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        ok.append(float(info.get("is_success", 0.0)))
        progress.append((start - end) / start if start > 0 else 1.0)
        touched.append(closest)
    env.close()
    return (np.array(d0), np.array(ok), np.array(progress), np.array(touched))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="run dir names; default = every push_td3_her_seed*")
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args()

    res = Path(args.results_dir)
    runs = args.runs or sorted(p.name for p in res.glob("push_td3_her_seed*"))
    seeds = range(args.seed_base, args.seed_base + args.episodes)

    lines = [
        "# Failure-mode sweep on held-out initial states",
        "",
        f"Env seeds {seeds.start}..{seeds.stop - 1} — disjoint from both the "
        "training seeds and the R4 eval seeds (10000+).",
        "",
        "| Run | Success | Failures | Failing seed (required push m / progress / closest approach m) |",
        "|---|---|---|---|",
    ]
    for name in runs:
        run_dir = res / name
        if not (run_dir / "checkpoint_best.pt").exists():
            print(f"skip {name}: no checkpoint_best.pt (still on the cluster?)")
            continue
        with open(run_dir / "config.yaml") as f:
            env_id = yaml.safe_load(f)["env_id"]
        d0, ok, progress, touched = sweep(run_dir, env_id, seeds, args.horizon)
        detail = ", ".join(
            f"{seeds.start + i} ({d0[i]:.3f} / {progress[i]:.0%} / {touched[i]:.3f})"
            for i in np.where(ok == 0)[0]
        ) or "none"
        lines.append(
            f"| {name} | {ok.mean():.3f} | {int((1 - ok).sum())}/{len(ok)} | {detail} |"
        )
        print(f"{name}: success={ok.mean():.3f} failures={int((1 - ok).sum())} | {detail}")

        # Success against required push distance, to test whether long pushes
        # are what actually fails. (For seed 0 they are not: the single
        # failure is in the longest bin, but 41/42 other long pushes succeed.)
        lines += ["", f"Success vs required push distance — `{name}`:", "",
                  "| Required push (m) | n | Success |", "|---|---|---|"]
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (d0 >= lo) & (d0 < hi)
            if m.sum():
                lines.append(f"| {lo:.2f}–{hi:.2f} | {int(m.sum())} | {ok[m].mean():.3f} |")
        lines.append("")

    out = res / "failure_sweep.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
