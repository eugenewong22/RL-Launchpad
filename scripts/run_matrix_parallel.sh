#!/usr/bin/env bash
# Launch every missing FetchPush run concurrently with nohup (for many-core
# servers). A run is skipped if its results directory already exists —
# delete a partial run's directory to relaunch it.
set -euo pipefail
cd "$(dirname "$0")/.."

launch() { # launch <run_dir> <command...>
  local dir="$1"; shift
  if [ -d "$dir" ]; then
    echo "skip $dir (exists)"
  else
    mkdir -p logs
    local name; name="$(basename "$dir")"
    nohup "$@" > "logs/$name.log" 2>&1 &
    echo "launched $name (pid $!)"
  fi
}

for seed in 0 1 2; do
  launch "results/push_td3_her_seed$seed" \
    uv run python -m src.agent.train --config configs/td3_her_push.yaml --seed "$seed"
  launch "results/push_td3_noher_seed$seed" \
    uv run python -m src.agent.train --config configs/td3_noher_push.yaml --seed "$seed"
  launch "results/push_sb3_her_seed$seed" \
    uv run python -m src.baseline.train_sb3 --config configs/sb3_td3_her_push.yaml --seed "$seed"
done
wait
