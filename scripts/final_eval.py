"""Final R4 evaluation: 50 episodes on the fixed eval seeds for every
checkpoint_best in results/, aggregated as mean ± std across seeds per arm.

Usage: uv run python scripts/final_eval.py [--env-id FetchPush-v4] [--episodes 50]
Writes results/final_eval.md and prints it.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import yaml

from src.agent.evaluate import evaluate, load_policy


def load_policy_fn(run_dir: Path, env_id: str):
    """Return (policy_fn, kind) for a run dir holding either our .pt
    checkpoint or an SB3 .zip — or None if neither exists."""
    pt = run_dir / "checkpoint_best.pt"
    zp = run_dir / "checkpoint_latest.zip"
    if pt.exists():
        return load_policy(pt, env_id), "from-scratch"
    if zp.exists():
        from stable_baselines3 import TD3 as SB3TD3

        model = SB3TD3.load(zp)
        return lambda obs: model.predict(obs, deterministic=True)[0], "sb3"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="FetchPush-v4")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-seed-base", type=int, default=10_000)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    arms = defaultdict(list)
    for run_dir in sorted(Path(args.results_dir).glob("*_seed*")):
        m = re.match(r"(.+)_seed(\d+)$", run_dir.name)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        # Each run records its own env; --env-id filters which arms to score.
        with open(run_dir / "config.yaml") as f:
            run_env = yaml.safe_load(f)["env_id"]
        if run_env != args.env_id:
            continue
        loaded = load_policy_fn(run_dir, args.env_id)
        if loaded is None:
            continue
        policy_fn, kind = loaded
        result = evaluate(policy_fn, args.env_id, args.episodes, args.eval_seed_base)
        print(f"{run_dir.name} ({kind}): success={result['success_rate']:.3f}")
        arms[arm].append((seed, result["success_rate"]))

    lines = [
        f"# Final evaluation — {args.env_id}",
        "",
        f"{args.episodes} episodes per seed, deterministic policy, eval seeds "
        f"{args.eval_seed_base}..{args.eval_seed_base + args.episodes - 1} (disjoint from training).",
        "",
        "| Arm | Seeds | Success rate (mean ± std) | Per-seed |",
        "|---|---|---|---|",
    ]
    for arm, rows in sorted(arms.items()):
        rates = np.array([r for _, r in sorted(rows)])
        per_seed = ", ".join(f"s{seed}={rate:.2f}" for seed, rate in sorted(rows))
        lines.append(
            f"| {arm} | {len(rates)} | {rates.mean():.3f} ± {rates.std():.3f} | {per_seed} |"
        )
    out = Path(args.results_dir) / "final_eval.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
