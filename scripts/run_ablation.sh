#!/usr/bin/env bash
# Single-factor ablation of the stabilizer package (docs/writeup.md §5).
#
# Five 1M-step FetchPush runs, each reverting ONE stabilizer to the value the
# original failed campaign used. ~42 min per run on an M-series laptop, so
# ~3.5 h sequential. Safe to leave overnight; each run checkpoints and logs
# to its own results/ablate_<key>/ directory.
#
#   bash scripts/run_ablation.sh            # sequential (default)
#   bash scripts/run_ablation.sh --jobs 3   # 3 at a time
#   bash scripts/run_ablation.sh --resume   # skip arms that already finished
#
# Sequential is the default deliberately: run in parallel and the per-run
# wall-clock in progress.csv stops being comparable to the reported runs in
# results/compute_table.md, because the arms contend for cores. Success rates
# are unaffected either way -- only the R6 timing numbers are.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

JOBS=1
RESUME=0
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PY="uv run python"
KEYS=(action_l2 normalize_obs tau expl_noise random_eps)

$PY scripts/make_ablation_configs.py

mkdir -p logs
started=0
for key in "${KEYS[@]}"; do
  # train.py appends _seed0 when --seed is passed without --run-dir.
  out="results/ablate_${key}_seed0"
  # An arm counts as done only if it reached the full budget -- a killed run
  # leaves a short progress.csv that would otherwise look complete.
  if [ "$RESUME" = 1 ] && [ -f "$out/progress.csv" ] \
     && tail -1 "$out/progress.csv" | grep -q "^1000000,"; then
    echo "skip $key: already ran to 1M steps"
    continue
  fi

  echo "=== ablate $key -> $out (log: logs/ablate_${key}.log) ==="
  $PY -m src.agent.train --config "configs/td3_her_push_ablate_${key}.yaml" \
      --seed 0 > "logs/ablate_${key}.log" 2>&1 &

  started=$((started + 1))
  # Throttle to $JOBS concurrent trainings. Polled rather than `wait -n`:
  # macOS ships bash 3.2, where `wait -n` does not exist and, under set -e,
  # takes the whole script down on the first arm. 5s granularity is nothing
  # against ~46-minute runs.
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
done
wait
echo "=== $started arm(s) finished ==="

$PY scripts/analyze_ablation.py
