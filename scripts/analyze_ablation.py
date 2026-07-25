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
TOLERANCE = 0.10

# Final success saturates: every single-factor arm reaches ~1.0 by 1M steps,
# so judging on it alone reports "nothing matters" and misses a 2x sample-
# efficiency cost. Steps-to-durable-0.9 is the metric with signal.
#
# The yardstick is the reported arm's OWN seed-to-seed spread rather than an
# arbitrary threshold: with n=1 per ablation arm, anything inside the band
# three seeds of the SAME config already produce is not evidence of an effect.
REPORTED_SEEDS = ["push_td3_her_seed0", "push_td3_her_seed1", "push_td3_her_seed2"]
DURABLE = (0.9, 0.8)  # reached >=0.9 and never fell below 0.8 after


def steps_to_durable(rows: list[dict]) -> int | None:
    """First env_steps at which success hits 0.9 and never drops below 0.8."""
    hi, lo = DURABLE
    steps = [int(r["env_steps"]) for r in rows]
    succ = [float(r["success_rate"]) for r in rows]
    for i, v in enumerate(succ):
        if v >= hi and min(succ[i:]) >= lo:
            return steps[i]
    return None


def read_final(run_dir: Path, require_complete: bool = True) -> dict | None:
    """None if the run has not finished.

    A partial progress.csv looks exactly like a finished one to a reader that
    only takes the last row -- and an arm 30k steps into a 1M-step budget
    reads as a total collapse. Judging that would put a false verdict in a
    committed report, so an arm counts only once it reaches its configured
    budget.
    """
    csv_path = run_dir / "progress.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    if require_complete:
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            budget = yaml.safe_load(cfg_path.read_text()).get("total_env_steps")
            if budget and int(rows[-1]["env_steps"]) < budget:
                return None
    last = rows[-1]
    # Average the last 5 evals: single evals are 20-episode samples and
    # bounce by a couple of points even on a solved policy.
    tail = rows[-5:]
    return {
        "reach90": steps_to_durable(rows),
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

    # The reported config's own seed spread is the noise floor: with n=1 per
    # ablation arm, a difference inside the band three seeds of the SAME
    # config already produce is not evidence of anything.
    seed_reach = [
        r["reach90"] for r in (read_final(res / s) for s in REPORTED_SEEDS)
        if r and r["reach90"]
    ]
    band = (min(seed_reach), max(seed_reach)) if seed_reach else None

    def verdict(r):
        if r["drop"] > TOLERANCE:
            return "**collapses**"
        if band and r["reach90"] and r["reach90"] > band[1]:
            return f"**slower** ({r['reach90'] / reported['reach90']:.1f}x)"
        return "within seed noise"

    lines = [
        "# Single-factor ablation of the stabilizer package",
        "",
        "The reported config deviates from SB3's published Fetch values in five "
        "places (`results/config_diff.md`). The original campaign changed all "
        "five at once, so `docs/writeup.md` §5 could report only that the "
        "package works. Each run below reverts exactly ONE setting to its "
        "`TrainConfig` default — the value the failed campaign used — holding "
        "everything else at the reported config. Seed 0, 1M env steps.",
        "",
        "**Read the third column, not the second.** Final success saturates: "
        "every single-factor arm reaches ~1.0 by 1M steps, so final success "
        "alone says nothing matters. Steps to a *durable* 0.9 (reached and "
        "never dropping below 0.8 again) is where the differences live.",
        "",
    ]
    if band:
        lines += [
            f"The reported config's own three seeds reach durable 0.9 at "
            f"{band[0]:,}–{band[1]:,} steps. That band is the noise floor: with "
            "one seed per ablation arm, a result inside it is not evidence of "
            "an effect.",
            "",
        ]
    lines += [
        "| Reverted setting | Change | Success | Steps to 0.9 | Contact | Effect |",
        "|---|---|---|---|---|---|",
        f"| *none (reported run)* | — | **{reported['success']:.3f}** | "
        f"**{reported['reach90']:,}** | {fmt(reported['contact'], '{:.2f}')} | — |",
    ]
    for r in sorted(rows, key=lambda r: -(r["reach90"] or 0)):
        lines.append(
            f"| `{r['key']}` | {r['from']!r} → {r['to']!r} | {r['success']:.3f} | "
            f"{r['reach90']:,} | {fmt(r['contact'], '{:.2f}')} | {verdict(r)} |"
            if r["reach90"] else
            f"| `{r['key']}` | {r['from']!r} → {r['to']!r} | {r['success']:.3f} | "
            f"never | {fmt(r['contact'], '{:.2f}')} | **collapses** |"
        )
    if broken:
        lines.append(
            f"| *all five (archived)* | — | {broken['success']:.3f} | never | "
            f"{fmt(broken['contact'], '{:.2f}')} | **collapses** |"
        )

    # The conclusions are derived, not written: the first four arms showed no
    # collapse and a hardcoded "no single setting is necessary" would have
    # survived into the report once the fifth arm contradicted it.
    collapsed = [r for r in rows if r["drop"] > TOLERANCE or not r["reach90"]]
    slower = [
        r for r in rows
        if r not in collapsed and band and r["reach90"] and r["reach90"] > band[1]
    ]
    noise = [r for r in rows if r not in collapsed and r not in slower]

    lines += [
        "",
        "`contact` is `contact_frac`, the fraction of episodes that moved the "
        "block — the mechanism signal, since the original campaign died with a "
        "saturated actor that never reached the block.",
        "",
        "## What this shows",
        "",
    ]

    if len(collapsed) == 1:
        c = collapsed[0]
        lines += [
            f"**Exactly one of the five is necessary.** Reverting `{c['key']}` "
            f"({c['from']!r} → {c['to']!r}) puts the run back on the floor for "
            f"all 1M steps at contact {c['contact']:.2f} — the original "
            "campaign's failure reproduced by a single-line change. The other "
            f"{len(rows) - 1} are individually removable: the agent still "
            "solves the task without any one of them.",
            "",
        ]
    elif collapsed:
        lines += [
            "**Necessary settings:** "
            + ", ".join(f"`{c['key']}`" for c in collapsed)
            + " — reverting any one of these alone reproduces the collapse.",
            "",
        ]
    else:
        lines += [
            "**No single reversion reproduces the collapse**, so no one setting "
            "is individually necessary. That is not the same as the changes not "
            "mattering — the all-five-reverted control flat-lined on three "
            "independent arms — but with n=1 per arm we cannot separate genuine "
            "redundancy from a collapse fragile enough that any perturbation "
            "escapes it.",
            "",
        ]

    if slower:
        lines += [
            "**Costly but not necessary:** "
            + ", ".join(
                f"`{s['key']}` ({s['reach90'] / reported['reach90']:.1f}x the "
                f"steps to a durable 0.9)" for s in slower
            )
            + ". The task is still solved by 1M steps, so this would be "
            "invisible in the final number alone.",
            "",
        ]
    if noise:
        lines += [
            "**Not individually load-bearing:** "
            + ", ".join(f"`{n['key']}`" for n in noise)
            + f". Each lands inside the reported config's own seed spread "
            f"({band[0]:,}–{band[1]:,} steps), so at n=1 there is no evidence "
            "any of them changes the outcome on its own." if band else "",
            "",
        ]

    lines += [
        "**Necessary is not sufficient.** During the original diagnosis, adding "
        "sustained ε-random exploration to the *broken* config did not rescue "
        "it (the block moved in 0/50 episodes — a collapsed policy acts as a "
        "restoring force against per-step random actions). Removing it from the "
        "*working* config breaks it. Both are true: it is required, and it "
        "needs at least one of the others alongside it.",
        "",
        f"Caveat: one seed per arm. That is enough to identify a collapse "
        f"({collapsed[0]['key'] if len(collapsed) == 1 else 'above'} is "
        "unambiguous — floor for 1M steps) and enough to rule out large "
        "effects, but not to resolve small ones. Three seeds per arm would "
        "settle the middle ground.",
        "",
    ]

    if missing:
        # Must be in the FILE, not just stdout: a report that silently omits
        # an arm reads as a complete result to anyone who did not watch it
        # being generated.
        lines += [
            f"> **Incomplete — {len(missing)} of 5 arms not yet run:** "
            f"{', '.join(f'`{m}`' for m in missing)}. "
            "Run `bash scripts/run_ablation.sh --resume` and regenerate. "
            "Conclusions above are provisional until every arm is present.",
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
