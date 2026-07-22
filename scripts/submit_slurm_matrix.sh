#!/usr/bin/env bash
# Submit every missing FetchPush run as its own SLURM job. Run from the
# repo root on the LOGIN node, after `uv sync` has already fetched deps
# there (compute nodes on most clusters, incl. NUS SoC, have no internet).
#
# Usage: bash scripts/submit_slurm_matrix.sh
# Override any default: PARTITION=normal TIME=02:30:00 bash scripts/submit_slurm_matrix.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Default 'long': the 8h default TIME below exceeds 'normal' partition's
# 3h cap on this cluster (both share the same xcn* CPU nodes, 'long'
# just allows longer walltime). Run `sinfo` to see current node states.
PARTITION="${PARTITION:-long}"
CPUS="${CPUS:-2}"
MEM="${MEM:-4G}"
TIME="${TIME:-08:00:00}"

submit() { # submit <run_dir> <cmd...>
  local dir="$1"; shift
  if [ -d "$dir" ]; then
    echo "skip $dir (exists)"
    return
  fi
  local name; name="$(basename "$dir")"
  mkdir -p logs
  sbatch \
    --job-name="$name" \
    --partition="$PARTITION" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --chdir="$PWD" \
    --output="logs/${name}.out" \
    --export=ALL,OMP_NUM_THREADS="$CPUS" \
    --wrap="export PATH=\"\$HOME/.venvs/bootstrap/bin:\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH\"; $*"
  echo "submitted $name"
}

for seed in 0 1 2; do
  submit "results/push_td3_her_seed$seed" \
    uv run python -m src.agent.train --config configs/td3_her_push.yaml --seed "$seed"
  submit "results/push_td3_noher_seed$seed" \
    uv run python -m src.agent.train --config configs/td3_noher_push.yaml --seed "$seed"
  submit "results/push_sb3_her_seed$seed" \
    uv run python -m src.baseline.train_sb3 --config configs/sb3_td3_her_push.yaml --seed "$seed"
done
