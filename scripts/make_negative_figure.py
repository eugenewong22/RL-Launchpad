"""Render the negative result: the first FetchPush campaign, which flat-lined
on every arm, against the fixed config on the same axes and the same budget.

Also emits the broken->fixed config diff as markdown. The write-up quotes
that diff, and quoting a *generated* table is the only way the prose cannot
drift from the configs it describes — which it already had once, naming
three stabilizers when the diff carries five.

Usage: uv run python scripts/make_negative_figure.py
Writes results/negative_result.png and results/config_diff.md.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

# The archived campaign and its post-fix counterpart. Same run names on both
# sides on purpose: same seed, same task, same 1M-step budget, so the only
# thing that differs is the config.
BROKEN_DIR = "archive_broken_config"
# All three failed arms sit at exactly 0.05 for most of the run, so they
# would render as a single line. Distinct dash patterns interleave and keep
# all three readable — the same trick make_plots.py uses, and the honest
# alternative to nudging the curves apart vertically.
ARMS = {
    "push_td3_her_seed0": ("TD3+HER (ours)", "#4da3ff", (0, (5, 2))),
    "push_sb3_her_seed0": ("TD3+HER (SB3)", "#6f7d8a", (0, (1, 3))),
    "push_td3_noher_seed0": ("TD3 no-HER (ablation)", "#c08b5c", (0, (3, 1, 1, 1))),
}
# Only our own arm was re-run under the fixed config; the SB3 arms were
# rerun too but their failure is the reported baseline result, not a bug we
# fixed, so plotting them post-fix would confuse the two stories.
FIXED_ARM = "push_td3_her_seed0"

# Keys that differ between run dirs for reasons unrelated to the fix.
IGNORED_KEYS = {"run_dir"}


def load_curve(csv_path: Path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["env_steps"]) for r in rows]
    succ = [float(r["success_rate"]) for r in rows]
    return steps, succ


def config_diff(broken: Path, fixed: Path) -> list[tuple[str, str, str]]:
    """(key, broken_value, fixed_value) for every key that changed.

    Absent keys render as "-" rather than being skipped: three of the five
    changes are *additions*, and an added knob is the most interesting kind
    of change to report.
    """
    with open(broken) as f:
        b = yaml.safe_load(f)
    with open(fixed) as f:
        g = yaml.safe_load(f)
    rows = []
    for key in sorted(set(b) | set(g)):
        if key in IGNORED_KEYS:
            continue
        bv, gv = b.get(key, "—"), g.get(key, "—")
        if bv != gv:
            rows.append((key, str(bv), str(gv)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    res = Path(args.results_dir)
    broken_root = res / BROKEN_DIR

    if not broken_root.exists():
        raise SystemExit(f"{broken_root} not found — nothing to plot")

    # Wide/short to match make_plots.py — see the note there.
    fig, ax = plt.subplots(figsize=(9, 2.6))

    for run, (label, color, dashes) in ARMS.items():
        csv_path = broken_root / run / "progress.csv"
        if not csv_path.exists():
            print(f"skip {run}: no progress.csv under {BROKEN_DIR}/")
            continue
        steps, succ = load_curve(csv_path)
        ax.plot(steps, succ, linestyle=dashes, color=color, linewidth=1.6,
                label=f"{label} — original config")

    fixed_csv = res / FIXED_ARM / "progress.csv"
    if fixed_csv.exists():
        steps, succ = load_curve(fixed_csv)
        ax.plot(steps, succ, "-", color=ARMS[FIXED_ARM][1], linewidth=2.2,
                label=f"{ARMS[FIXED_ARM][0]} — after the fix")

    # The floor is not zero: one of the 20 in-training eval seeds starts with
    # the block already inside the goal threshold, so a policy that does
    # nothing at all scores 0.05. Marking it stops the flat lines reading as
    # "some small amount of learning".
    ax.axhline(0.05, color="#ff8f6b", linewidth=0.9, alpha=0.7)
    # Sits to the right of the rise and above the flat lines, the one region
    # of the axes no curve occupies.
    ax.annotate("0.05 = eval floor: 1 of the 20 eval seeds starts already solved,\n"
                "so a policy that never moves scores 0.05, not 0",
                xy=(0.46, 0.155), xycoords=("axes fraction", "data"),
                fontsize=8, color="#ff8f6b")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("FetchPush-v4: the first campaign flat-lined on every arm")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = res / "negative_result.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")

    rows = config_diff(broken_root / FIXED_ARM / "config.yaml",
                       res / FIXED_ARM / "config.yaml")
    lines = [
        "# Config diff: the campaign that failed vs the one that worked",
        "",
        f"`results/{BROKEN_DIR}/{FIXED_ARM}/config.yaml` → "
        f"`results/{FIXED_ARM}/config.yaml`. Same task, same seed, same 1M-step "
        "budget. Generated by `scripts/make_negative_figure.py` so the write-up's "
        "account of the fix cannot drift from the configs.",
        "",
        "| Key | Failed campaign | Fixed campaign |",
        "|---|---|---|",
        *(f"| `{k}` | {b} | {g} |" for k, b, g in rows),
        "",
        f"{len(rows)} changes. `—` means the key was absent.",
    ]
    out_md = res / "config_diff.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}\n")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
