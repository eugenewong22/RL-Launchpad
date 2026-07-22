#!/usr/bin/env bash
# Full experiment matrix (plan Phase 4): 3 seeds x 3 arms on FetchPush.
# Arms: from-scratch TD3+HER / ablation TD3 no-HER / SB3 TD3+HER baseline.
# Runs sequentially; expect several hours per arm per seed on CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(0 1 2)

for seed in "${SEEDS[@]}"; do
  uv run python -m src.agent.train --config configs/td3_her_push.yaml --seed "$seed"
done

for seed in "${SEEDS[@]}"; do
  uv run python -m src.agent.train --config configs/td3_noher_push.yaml --seed "$seed"
done

for seed in "${SEEDS[@]}"; do
  uv run python -m src.baseline.train_sb3 --config configs/sb3_td3_her_push.yaml --seed "$seed"
done

uv run python scripts/make_plots.py
