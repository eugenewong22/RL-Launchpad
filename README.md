# From-Scratch TD3 + HER on Fetch Manipulation

LaunchPad 2026 submission — Griffin Labs "RL From Scratch" track.

A TD3 agent with Hindsight Experience Replay, written from first
principles in PyTorch (`src/agent/` — training loop, algorithm, and
networks; no RL-library code), trained on sparse-reward Fetch
manipulation and measured against the Stable-Baselines3 TD3+HER
baseline under an identical evaluation protocol.

## Judge quickstart (< 15 min)

Requires [uv](https://docs.astral.sh/uv/) (any platform; CPU is enough).

```bash
git clone <this-repo> && cd launchpad-rl
uv sync                                    # installs pinned deps from uv.lock (~2 min)
uv run python scripts/check_env.py         # simulator sanity check (~30 s)
uv run pytest                              # unit + integration tests (~30 s)

# Evaluate the reported checkpoint on the R4 protocol (50 fixed eval seeds):
uv run python -m src.agent.evaluate \
    --checkpoint results/push_td3_her_seed0/checkpoint_best.pt \
    --env-id FetchPush-v4 --episodes 50
```

To watch the policy, render episodes from the same checkpoint and seeds:

```bash
uv run python scripts/record_video.py \
    --checkpoint results/push_td3_her_seed0/checkpoint_best.pt \
    --env-id FetchPush-v4 --episodes 5 --out results/demo.mp4
```

## Rules mapping

| Rule | Where |
|---|---|
| R1 from-scratch algorithm/networks | `src/agent/` only; `src/baseline/` is SB3 and clearly separated |
| R2 published baseline, same protocol | `src/baseline/train_sb3.py` evaluates through the same `src/agent/evaluate.evaluate` |
| R3 reproducibility | `uv.lock` pins exact versions; configs + seeds committed; this quickstart |
| R4 standardized eval | 50 episodes, eval seeds `10000+i`, disjoint from training seeds; videos from reported checkpoints |
| R5 simulation only, stock tasks | Unmodified `FetchReach-v4` / `FetchPush-v4` — no reward, observation, or terrain changes |
| R6 compute honesty | every run logs env steps + wall-clock to `progress.csv`; `results/compute_table.md` aggregates |

## Reproduce the experiments

```bash
# Smoke test (2 min CPU): from-scratch TD3+HER on FetchReach -> ~100% success
uv run python -m src.agent.train --config configs/td3_her_reach.yaml --seed 0

# Full matrix: 3 seeds x {from-scratch TD3+HER, no-HER ablation, SB3 baseline} on FetchPush
bash scripts/run_all_seeds.sh

# Figures + compute table from committed CSVs
uv run python scripts/make_plots.py
```

## Layout

```
src/agent/          from-scratch code (R1): networks, HER buffer, TD3, train loop, eval
src/baseline/       SB3 TD3+HER baseline runner (R2)
configs/            one YAML per experiment arm
scripts/            env check, run matrix, plots, video
tests/              unit + integration tests (written first; see git history)
results/            committed progress.csv per run + checkpoints + figures
```
