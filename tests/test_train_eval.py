import csv

import numpy as np

from src.agent.evaluate import evaluate
from src.agent.train import TrainConfig, run_training


def scripted_policy(obs_dict):
    """Deterministic stand-in: returns zeros (gripper stays put)."""
    assert "observation" in obs_dict and "desired_goal" in obs_dict
    return np.zeros(4, dtype=np.float32)


def test_evaluate_is_deterministic_under_fixed_seeds():
    r1 = evaluate(scripted_policy, "FetchReach-v4", n_episodes=3, eval_seed_base=10_000)
    r2 = evaluate(scripted_policy, "FetchReach-v4", n_episodes=3, eval_seed_base=10_000)
    assert 0.0 <= r1["success_rate"] <= 1.0
    assert r1 == r2
    r3 = evaluate(scripted_policy, "FetchReach-v4", n_episodes=3, eval_seed_base=20_000)
    assert r3["episode_seeds"] != r1["episode_seeds"]


def test_short_training_run_writes_log_and_checkpoint(tmp_path):
    cfg = TrainConfig(
        env_id="FetchReach-v4",
        total_env_steps=300,  # 6 episodes of 50 steps
        warmup_steps=100,
        eval_every=150,
        n_eval_episodes=2,
        normalize_obs=True,
        action_l2=1.0,
        run_dir=str(tmp_path / "run"),
        seed=0,
    )
    run_training(cfg)

    log_path = tmp_path / "run" / "progress.csv"
    assert log_path.exists()
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 2  # evals at 150 and 300 steps
    assert {"env_steps", "wall_clock_s", "success_rate", "critic_loss"} <= set(rows[0])
    assert int(rows[-1]["env_steps"]) == 300

    assert (tmp_path / "run" / "checkpoint_latest.pt").exists()
    assert (tmp_path / "run" / "config.yaml").exists()
    # normalization stats are part of the learned artifact (eval needs them)
    assert (tmp_path / "run" / "normalizer.npz").exists()
