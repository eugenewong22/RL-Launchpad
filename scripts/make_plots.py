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
import yaml

ARM_LABELS = {
    "push_td3_her": "TD3+HER (from scratch)",
    "push_sb3_her": "TD3+HER (SB3 baseline)",
    "push_sb3_sac": "SAC+HER (SB3 baseline)",
    "push_td3_noher": "TD3 without HER (ablation)",
    "pickplace_td3_her": "PickAndPlace TD3+HER (from scratch)",
    "reach_smoke": "FetchReach smoke",
}

# The two failing arms both sit at the 0.05 eval floor and would render as a
# single line; distinct dash patterns keep both visible.
ARM_STYLES = {
    "push_sb3_her": "--",
    "push_td3_noher": ":",
}


def load_runs(results_dir: Path) -> dict:
    """(env_id, arm) -> list of {env_steps, success_rate, wall_clock_s}.

    Keyed by env as well as arm: FetchReach/Push/PickAndPlace are different
    problems, so their curves must never share an axis.
    """
    arms = defaultdict(list)
    for csv_path in sorted(results_dir.glob("*/progress.csv")):
        run_name = csv_path.parent.name
        arm = re.sub(r"_seed\d+$", "", run_name)
        if arm.startswith("probe"):
            continue  # short diagnostic probes, not reported arms
        cfg_path = csv_path.parent / "config.yaml"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            env_id = yaml.safe_load(f)["env_id"]
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        arms[(env_id, arm)].append(
            {
                "run": run_name,
                "env_steps": np.array([int(r["env_steps"]) for r in rows]),
                "success_rate": np.array([float(r["success_rate"]) for r in rows]),
                "wall_clock_s": np.array([float(r["wall_clock_s"]) for r in rows]),
            }
        )
    return dict(arms)


def plot_arm(ax, runs: list, label: str, style: str = "-"):
    # All runs of an arm share eval_every; align on the shortest run.
    n = min(len(r["env_steps"]) for r in runs)
    steps = runs[0]["env_steps"][:n]
    curves = np.stack([r["success_rate"][:n] for r in runs])
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    (line,) = ax.plot(steps, mean, style, label=f"{label} (n={len(runs)} seeds)")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=line.get_color())


def compute_table(arms: dict) -> str:
    lines = [
        "| Run | Task | Env steps | Wall-clock (min) |",
        "|---|---|---|---|",
    ]
    for (env_id, _), runs in sorted(arms.items()):
        for r in runs:
            lines.append(
                f"| {r['run']} | {env_id} | {r['env_steps'][-1]:,} "
                f"| {r['wall_clock_s'][-1] / 60:.1f} |"
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

    # One figure per task. FetchPush is the headline result and keeps the
    # filename the write-up references; other tasks get their own file.
    envs = sorted({env_id for env_id, _ in arms})
    for env_id in envs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (e, arm), runs in sorted(arms.items()):
            if e != env_id:
                continue
            plot_arm(ax, runs, ARM_LABELS.get(arm, arm), ARM_STYLES.get(arm, "-"))
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Eval success rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
        ax.set_title(f"{env_id}: eval success rate (mean ± std across seeds)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        stem = "learning_curves" if env_id.startswith("FetchPush") else f"learning_curves_{env_id.split('-')[0]}"
        out = results_dir / f"{stem}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

    table = compute_table(arms)
    table_path = results_dir / "compute_table.md"
    table_path.write_text(table + "\n")
    print(f"wrote {table_path}\n\n{table}")


if __name__ == "__main__":
    main()
