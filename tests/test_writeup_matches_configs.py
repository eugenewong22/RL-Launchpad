"""Guard the write-up's factual claims against the artifacts that back them.

This exists because the write-up drifted from the repo twice: it stated
tau=0.005 in two places while every reported run used 0.05, and listed
observation normalization under "deliberately did not build" while
normalize_obs was true in all five configs. Both configs are committed and
the README invites judges to open them, so the prose was checkably wrong.

The repo's stated principle is that no number exists outside the artifact
that produced it (scripts/make_plots.py). Numbers that obeyed it stayed
correct; the ones transcribed by hand did not. These tests extend that
principle to the write-up: a claim about a config is checked against the
config, so drift fails CI instead of reaching a judge.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "docs" / "writeup.md"
RESULTS = ROOT / "results"

REPORTED_RUN = "push_td3_her_seed0"
FAILED_RUN = Path("archive_broken_config") / "push_td3_her_seed0"

# Every from-scratch run the write-up claims shares one config.
IDENTICAL_CONFIG_RUNS = [
    "push_td3_her_seed0",
    "push_td3_her_seed1",
    "push_td3_her_seed2",
    "push_td3_noher_seed0",
    "pickplace_td3_her_seed0",
]
# ...except the one key the ablation is defined by.
ABLATION_KEYS = {"her_k", "env_id", "run_dir", "seed", "total_env_steps"}


def load_cfg(rel: str | Path) -> dict:
    path = RESULTS / rel / "config.yaml"
    if not path.exists():
        pytest.skip(f"{path} not present (checkpoints may still be on the cluster)")
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def writeup() -> str:
    return WRITEUP.read_text()


def parse_deviation_table(text: str) -> dict[str, tuple[str, str]]:
    """key -> (failed_value, reported_value) from the section-2 table.

    Matches rows shaped `| \\`tau\\` | 0.005 | **0.05** | why |`. The values
    carry markdown bold and an em-dash for "absent", both stripped here.
    """
    rows = {}
    for line in text.splitlines():
        m = re.match(
            r"\|\s*`(\w+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*[^|]*\|\s*$", line
        )
        if m:
            key, failed, reported = (s.strip() for s in m.groups())
            clean = lambda v: v.replace("**", "").strip()  # noqa: E731
            rows[key] = (clean(failed), clean(reported))
    return rows


def as_config_value(raw) -> str:
    """Render a config value the way the write-up's table writes it.

    YAML booleans arrive as Python bools, which str() renders "True"; the
    table writes them lowercase, matching the YAML source.
    """
    if isinstance(raw, bool):
        return "true" if raw else "false"
    return str(raw)


def test_deviation_table_matches_both_configs(writeup):
    """Every row of the section-2 table must match the two committed configs."""
    table = parse_deviation_table(writeup)
    assert table, "no deviation table found in docs/writeup.md"

    failed_cfg, reported_cfg = load_cfg(FAILED_RUN), load_cfg(REPORTED_RUN)
    for key, (failed_claim, reported_claim) in table.items():
        # "—" is how the table spells "key absent from that config".
        if failed_claim == "—":
            assert key not in failed_cfg, (
                f"writeup says `{key}` was absent from the failed config, "
                f"but it is present with value {failed_cfg[key]!r}"
            )
        else:
            assert as_config_value(failed_cfg[key]).lower() == failed_claim.lower(), (
                f"`{key}`: writeup says the failed campaign used {failed_claim!r}, "
                f"config says {failed_cfg[key]!r}"
            )
        assert as_config_value(reported_cfg[key]).lower() == reported_claim.lower(), (
            f"`{key}`: writeup says the reported runs use {reported_claim!r}, "
            f"config says {reported_cfg[key]!r}"
        )


def test_deviation_table_is_complete(writeup):
    """The table must list *every* difference, not a convenient subset.

    The original prose named three stabilizers when the diff carried five.
    """
    table = parse_deviation_table(writeup)
    failed_cfg, reported_cfg = load_cfg(FAILED_RUN), load_cfg(REPORTED_RUN)
    ignored = {"run_dir"}
    actual = {
        k
        for k in set(failed_cfg) | set(reported_cfg)
        if k not in ignored and failed_cfg.get(k) != reported_cfg.get(k)
    }
    assert actual == set(table), (
        f"writeup's deviation table lists {sorted(table)}, "
        f"but the configs actually differ in {sorted(actual)}"
    )


def test_architecture_diagram_tau_matches_config(writeup):
    """The tau in the ASCII diagram is the one a judge reads first."""
    cfg = load_cfg(REPORTED_RUN)
    m = re.search(r"Polyak target copies of all three, τ=([\d.]+)", writeup)
    assert m, "could not find the tau annotation in the architecture diagram"
    assert float(m.group(1)) == cfg["tau"], (
        f"diagram says τ={m.group(1)}, config says τ={cfg['tau']}"
    )


def test_reported_runs_share_one_config():
    """The write-up claims nothing was tuned per task or per seed."""
    base = load_cfg(REPORTED_RUN)
    for run in IDENTICAL_CONFIG_RUNS[1:]:
        cfg = load_cfg(run)
        for key in set(base) | set(cfg):
            if key in ABLATION_KEYS:
                continue
            assert base.get(key) == cfg.get(key), (
                f"{run} differs from {REPORTED_RUN} on `{key}`: "
                f"{cfg.get(key)!r} vs {base.get(key)!r} — the write-up claims "
                "every from-scratch run uses an identical config"
            )


def test_agent_code_imports_no_rl_library():
    """R1: the from-scratch code must not import an RL library."""
    banned = re.compile(
        r"^\s*(?:import|from)\s+(stable_baselines3|sb3|cleanrl|rsl_rl|tianshou|ray)",
        re.M,
    )
    offenders = [
        p.relative_to(ROOT)
        for p in (ROOT / "src" / "agent").rglob("*.py")
        if banned.search(p.read_text())
    ]
    assert not offenders, f"RL-library import inside src/agent/ (R1 violation): {offenders}"


RESULTS_TABLE = RESULTS / "final_eval_push.md"

# Display label in prose -> arm key in the generated table. Spelled out
# rather than fuzzy-matched, because a *swapped* mapping is precisely the
# error worth catching: two arms tie at 0.993 ± 0.009, so comparing the
# numbers as an unordered set would not notice them being attributed to the
# wrong arms. Both dash styles appear (README uses em-dashes, the write-up
# parentheses).
LABEL_TO_ARM = {
    "TD3+HER (from scratch)": "push_td3_her",
    "TD3+HER — from scratch": "push_td3_her",
    "SAC+HER (SB3 baseline)": "push_sb3_sac",
    "SAC+HER — SB3 baseline": "push_sb3_sac",
    "TD3+HER (SB3 baseline)": "push_sb3_her",
    "TD3+HER — SB3 baseline": "push_sb3_her",
    "TD3 no-HER (our ablation)": "push_td3_noher",
    "TD3 no-HER — our ablation": "push_td3_noher",
}

def doc_result_rows(text: str):
    """Yield (label, 'mean ± std', [per-seed]) from markdown table rows.

    Split on pipes per line rather than matched with one big regex: markdown
    bold around a cell's contents defeats a value-anchored pattern, and a
    lazy `.+?` label happily runs past a row boundary. Cells are what the
    format actually has, so cells are what we parse.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.replace("*", "").strip() for c in line.strip("|").split("|")]
        if len(cells) != 3:
            continue
        label, value, per_seed = cells
        if re.fullmatch(r"\d\.\d+ ± \d\.\d+", value):
            yield label, value, re.findall(r"\d\.\d+", per_seed)


def generated_results() -> dict[str, tuple[str, list[str]]]:
    """arm -> ('mean ± std', [per-seed]) straight from final_eval_push.md."""
    if not RESULTS_TABLE.exists():
        pytest.skip(f"{RESULTS_TABLE} not present")
    out = {}
    for line in RESULTS_TABLE.read_text().splitlines():
        m = re.match(
            r"\|\s*(\w+)\s*\|\s*\d+\s*\|\s*([\d.]+ ± [\d.]+)\s*\|\s*(.+?)\s*\|$", line
        )
        if m:
            out[m.group(1)] = (m.group(2), re.findall(r"s\d+=([\d.]+)", m.group(3)))
    return out


@pytest.mark.parametrize("doc", ["README.md", "docs/writeup.md"])
def test_prose_results_tables_match_generated(doc):
    """The hand-written results tables must match the generated one.

    README.md and docs/writeup.md each restate results/final_eval_push.md in
    a friendlier form. That restating is transcription, and transcription is
    what drifted before — so it is checked rather than trusted.
    """
    gen = generated_results()
    text = (ROOT / doc).read_text()

    checked = set()
    for label, value, got_seeds in doc_result_rows(text):
        if label not in LABEL_TO_ARM:
            continue
        arm = LABEL_TO_ARM[label]
        assert arm in gen, f"{doc} names an arm absent from {RESULTS_TABLE}: {arm}"
        want_value, want_seeds = gen[arm]
        assert value == want_value, (
            f"{doc}: '{label}' says {value!r}, {RESULTS_TABLE.name} says "
            f"{want_value!r}"
        )
        assert got_seeds == want_seeds, (
            f"{doc}: '{label}' per-seed {got_seeds} != generated {want_seeds}"
        )
        checked.add(arm)

    assert checked == set(gen), (
        f"{doc} restates {sorted(checked)} but {RESULTS_TABLE.name} reports "
        f"{sorted(gen)}; every reported arm should appear, and any label not "
        "in LABEL_TO_ARM is silently skipped — add it there"
    )


def cited_result_paths(writeup: str) -> set[str]:
    """results/ paths the write-up references, in backticks or as image links."""
    cited = set(re.findall(r"`(results/[^`]+?)`", writeup))
    cited |= set(re.findall(r"\]\(\.\./(results/[^)]+?)\)", writeup))
    return cited


def test_cited_result_files_exist(writeup):
    missing = sorted(c for c in cited_result_paths(writeup) if not (ROOT / c).exists())
    assert not missing, f"write-up cites files that do not exist: {missing}"


def test_cited_result_files_are_committed(writeup):
    """Existing on this laptop is not enough — they must survive a clone.

    The write-up cited results/push_failure_seed2081.mp4 while .gitignore
    carried `results/**/*.mp4`, so the file was here and absent from every
    clone. Filesystem checks cannot catch that; git has to be asked.
    """
    import subprocess

    tracked = set(
        subprocess.run(
            ["git", "ls-files", "results"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
    )
    # Directories are cited too (e.g. results/archive_broken_config/); a
    # directory counts as tracked if anything under it is.
    untracked = sorted(
        c for c in cited_result_paths(writeup)
        if c.rstrip("/") not in tracked
        and not any(t.startswith(c.rstrip("/") + "/") for t in tracked)
    )
    assert not untracked, (
        f"write-up cites paths that are not committed, so a judge's clone "
        f"will not have them: {untracked}"
    )
