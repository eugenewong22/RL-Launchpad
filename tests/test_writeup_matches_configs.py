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
