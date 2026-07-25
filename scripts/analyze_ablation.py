"""Report which of the five stabilizers are load-bearing.

Reads the ablation runs' progress.csv files and writes
results/stabilizer_ablation.md. Two controls bracket the five single-factor
runs and both already exist, so they cost nothing:

  none reverted  = results/push_td3_her_seed0            (the reported run)
  all 5 reverted = results/archive_broken_config/...     (the failed campaign)

A single-factor run is called load-bearing if reverting it drops final
success materially below the reported run. `contact_frac` is reported
alongside because it is the mechanism signal: a saturated actor does not
touch the block at all, which distinguishes "learned worse" from "collapsed
the way the original campaign did".

Usage: uv run python scripts/analyze_ablation.py
"""

import argparse
import csv
from pathlib import Path

import yaml

REPORTED = "push_td3_her_seed0"
BROKEN = Path("archive_broken_config") / "push_td3_her_seed0"
STABILIZERS = ["action_l2", "normalize_obs", "tau", "expl_noise", "random_eps"]

# train.py appends _seed<N> to run_dir when --seed is given without
# --run-dir (see src/agent/train.py), so the configs' `results/ablate_<key>`
# actually lands at `results/ablate_<key>_seed0`. Matching the repo's
# existing <arm>_seed<k> convention.
RUN_SUFFIX = "_seed0"

# Final success within this margin of the reported run counts as "no effect".
# 0.10 is deliberately generous: with n=1 seed per arm we can identify a
# collapse, not a small regression, and the table says so.
TOLERANCE = 0.10


def read_final(run_dir: Path) -> dict | None:
    csv_path = run_dir / "progress.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    last = rows[-1]
    # Average the last 5 evals: single evals are 20-episode samples and
    # bounce by a couple of points even on a solved policy.
    tail = rows[-5:]
    return {
        "success": sum(float(r["success_rate"]) for r in tail) / len(tail),
        "final": float(last["success_rate"]),
        "contact": (
            sum(float(r["contact_frac"]) for r in tail) / len(tail)
            if "contact_frac" in last else None
        ),
        "steps": int(last["env_steps"]),
        "wall_min": float(last["wall_clock_s"]) / 60,
    }


def fmt(v, spec="{:.3f}", dash="—"):
    return dash if v is None else spec.format(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    res = Path(args.results_dir)

    reported = read_final(res / REPORTED)
    if reported is None:
        raise SystemExit(f"{res / REPORTED}/progress.csv missing — nothing to compare against")
    broken = read_final(res / BROKEN)

    rows, missing = [], []
    for key in STABILIZERS:
        run = res / f"ablate_{key}{RUN_SUFFIX}"
        got = read_final(run)
        if got is None:
            missing.append(key)
            continue
        base_cfg = yaml.safe_load((res / REPORTED / "config.yaml").read_text())
        run_cfg = yaml.safe_load((run / "config.yaml").read_text())
        drop = reported["success"] - got["success"]
        rows.append({
            "key": key,
            "from": base_cfg.get(key),
            "to": run_cfg.get(key),
            **got,
            "drop": drop,
            "load_bearing": drop > TOLERANCE,
        })

    lines = [
        "# Single-factor ablation of the stabilizer package",
        "",
        "The reported config deviates from SB3's published Fetch values in five "
        "places (`results/config_diff.md`). The original campaign changed all "
        "five at once, so `docs/writeup.md` §5 could report only that the "
        "package works. Each run below reverts exactly ONE setting to its "
        "`TrainConfig` default — the value the failed campaign used — holding "
        "everything else at the reported config. Seed 0, 1M env steps, "
        "identical eval protocol.",
        "",
        "Success is the mean of the last 5 in-training evals (20 episodes each); "
        "`contact` is `contact_frac`, the fraction of episodes that moved the "
        "block, which is how a saturated actor is distinguished from one that "
        "merely learned worse.",
        "",
        "| Reverted setting | Change | Success | Δ vs reported | Contact | Load-bearing? |",
        "|---|---|---|---|---|---|",
        f"| *none (reported run)* | — | **{reported['success']:.3f}** | — | "
        f"{fmt(reported['contact'], '{:.2f}')} | — |",
    ]
    for r in sorted(rows, key=lambda r: -r["drop"]):
        verdict = "**yes**" if r["load_bearing"] else "no"
        lines.append(
            f"| `{r['key']}` | {r['from']!r} → {r['to']!r} | {r['success']:.3f} | "
            f"−{r['drop']:.3f} | {fmt(r['contact'], '{:.2f}')} | {verdict} |"
        )
    if broken:
        lines.append(
            f"| *all five (archived)* | — | {broken['success']:.3f} | "
            f"−{reported['success'] - broken['success']:.3f} | "
            f"{fmt(broken['contact'], '{:.2f}')} | — |"
        )

    lines += [
        "",
        f"Load-bearing = reverting it costs more than {TOLERANCE:.2f} success. "
        "With one seed per arm this identifies collapses, not small "
        "regressions; a settled question would need 3 seeds per arm.",
        "",
    ]
    if missing:
        lines += [
            f"**Incomplete:** no progress.csv yet for {', '.join(missing)}. "
            "Run `bash scripts/run_ablation.sh` and regenerate.",
            "",
        ]

    out = res / "stabilizer_ablation.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")
    if missing:
        print(f"\nWARNING: {len(missing)} of 5 arms have no data yet: {missing}")


if __name__ == "__main__":
    main()
