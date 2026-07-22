import csv

from src.baseline.train_sb3 import BaselineConfig, run_baseline


def test_short_baseline_run_writes_same_log_schema(tmp_path):
    cfg = BaselineConfig(
        env_id="FetchReach-v4",
        total_env_steps=300,
        eval_every=150,
        n_eval_episodes=2,
        learning_starts=100,
        run_dir=str(tmp_path / "run"),
        seed=0,
    )
    run_baseline(cfg)

    with open(tmp_path / "run" / "progress.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    # Identical schema to the from-scratch runs -> shared plotting code.
    assert {"env_steps", "wall_clock_s", "success_rate", "critic_loss"} <= set(rows[0])
    assert (tmp_path / "run" / "config.yaml").exists()
    assert (tmp_path / "run" / "checkpoint_latest.zip").exists()
