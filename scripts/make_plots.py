"""Regenerate every reported figure and the compute-honesty table from
committed progress.csv files (R2/R4/R6). No numbers exist outside CSVs.

Usage: uv run python scripts/make_plots.py [--results-dir results]
Groups runs named <arm>_seed<k>/ and plots mean ± std per arm against
env steps on a shared x-axis.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARM_LABELS = {
    "push_td3_her": "TD3+HER (from scratch)",
    "push_sb3_her": "TD3+HER (SB3 baseline)",
    "push_sb3_sac": "SAC+HER (SB3 baseline)",
    "push_td3_noher": "TD3 without HER (ablation)",
    "pickplace_td3_her": "PickAndPlace TD3+HER (from scratch)",
    "reach_smoke": "FetchReach smoke",
}


def load_runs(results_dir: Path) -> dict:
    """arm -> list of {env_steps: [...], success_rate: [...], wall_clock_s: [...]}"""
    arms = defaultdict(list)
    for csv_path in sorted(results_dir.glob("*/progress.csv")):
        run_name = csv_path.parent.name
        arm = re.sub(r"_seed\d+$", "", run_name)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        arms[arm].append(
            {
                "run": run_name,
                "env_steps": np.array([int(r["env_steps"]) for r in rows]),
                "success_rate": np.array([float(r["success_rate"]) for r in rows]),
                "wall_clock_s": np.array([float(r["wall_clock_s"]) for r in rows]),
            }
        )
    return dict(arms)


def plot_arm(ax, runs: list, label: str):
    # All runs of an arm share eval_every; align on the shortest run.
    n = min(len(r["env_steps"]) for r in runs)
    steps = runs[0]["env_steps"][:n]
    curves = np.stack([r["success_rate"][:n] for r in runs])
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    (line,) = ax.plot(steps, mean, label=f"{label} (n={len(runs)} seeds)")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=line.get_color())


def compute_table(arms: dict) -> str:
    lines = [
        "| Run | Env steps | Wall-clock (min) |",
        "|---|---|---|",
    ]
    for _, runs in sorted(arms.items()):
        for r in runs:
            lines.append(
                f"| {r['run']} | {r['env_steps'][-1]:,} | {r['wall_clock_s'][-1] / 60:.1f} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    arms = load_runs(results_dir)
    if not arms:
        raise SystemExit(f"no */progress.csv found under {results_dir}/")

    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, runs in sorted(arms.items()):
        plot_arm(ax, runs, ARM_LABELS.get(arm, arm))
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.set_title("Eval success rate vs environment steps (mean ± std across seeds)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = results_dir / "learning_curves.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    table = compute_table(arms)
    table_path = results_dir / "compute_table.md"
    table_path.write_text(table + "\n")
    print(f"wrote {table_path}\n\n{table}")


if __name__ == "__main__":
    main()
