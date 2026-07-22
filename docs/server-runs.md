# Running the experiment matrix on a remote (school) server

Training is CPU-only and headless — no GPU, no display, no rendering
libraries needed. The repo is self-contained: `uv` fetches the pinned
Python and all locked dependencies. Wall-clock differs across hardware;
that is fine under R6 as long as it's reported, and `progress.csv`
records it per run — just note the server's CPU model alongside.

## One-time setup

```bash
ssh <you>@<server>
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs to ~/.local/bin
git clone <repo-url> launchpad-rl && cd launchpad-rl
uv sync            # fetches Python 3.11 + pinned deps into .venv
uv run pytest      # 17 tests should pass before burning compute
```

## Option A: plain server (tmux/nohup)

Servers usually have far more than 4 cores, so run all 9 remaining runs
concurrently (each uses ~1 core):

```bash
bash scripts/run_matrix_parallel.sh   # nohup-launches every missing run
tail -f results/push_td3_her_seed1/progress.csv   # watch any run
```

## Option B: SLURM cluster

```bash
for seed in 0 1 2; do
  for cfg in td3_her_push td3_noher_push; do
    sbatch --job-name="$cfg-$seed" --cpus-per-task=2 --mem=4G --time=06:00:00 \
      --wrap="cd $PWD && uv run python -m src.agent.train --config configs/$cfg.yaml --seed $seed"
  done
  sbatch --job-name="sb3-$seed" --cpus-per-task=2 --mem=4G --time=08:00:00 \
    --wrap="cd $PWD && uv run python -m src.baseline.train_sb3 --config configs/sb3_td3_her_push.yaml --seed $seed"
done
```

## Bringing results home

Each finished run is one directory: `progress.csv`, `config.yaml`, and
checkpoints. Sync them into the local repo and regenerate everything:

```bash
rsync -av <you>@<server>:launchpad-rl/results/ results/
uv run python scripts/make_plots.py
uv run python scripts/final_eval.py
```

Skip runs that already exist locally — arms/seeds are independent, so
local and server runs mix freely (hardware per run goes in the compute
table).
