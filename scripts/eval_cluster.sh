#!/usr/bin/env bash
# Final R4 evaluation, meant to run on a COMPUTE node via sbatch.
#
# The login node is older hardware than the xcn* compute nodes: importing
# torch/numpy there dies with "Illegal instruction (core dumped)" (SIGILL —
# the wheels use CPU instructions the login node lacks), and it also has a
# per-process memory cap that aborts large imports. Every training run
# happened on a compute node, so scoring those checkpoints must too.
#
# Usage, from the repo root on the login node:
#   sbatch --partition=long --cpus-per-task=4 --mem=8G --time=01:00:00 \
#          --chdir="$PWD" --output=logs/eval.out scripts/eval_cluster.sh
# Then: cat logs/eval.out
set -euo pipefail
cd "$(dirname "$0")/.."

# Call the venv's python directly — uv is not needed once .venv exists, and
# sbatch jobs don't source ~/.bashrc so uv may not be on PATH anyway.
PY=".venv/bin/python"
[ -x "$PY" ] || { echo "no $PY — run scripts/setup_cluster.sh first" >&2; exit 1; }

echo "### host: $(hostname)"
echo
echo "### FetchPush-v4 (headline result)"
"$PY" scripts/final_eval.py --env-id FetchPush-v4
echo
cat results/final_eval.md
cp results/final_eval.md results/final_eval_push.md

echo
echo "### FetchPickAndPlace-v4 (stretch task)"
"$PY" scripts/final_eval.py --env-id FetchPickAndPlace-v4
echo
cat results/final_eval.md
cp results/final_eval.md results/final_eval_pickplace.md

echo
echo "### base64 of both tables (paste this back)"
tar czf - results/final_eval_push.md results/final_eval_pickplace.md | base64 | fold -w 200
