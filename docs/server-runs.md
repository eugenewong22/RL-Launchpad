# Running the experiment matrix on the school (SLURM) cluster

Training is CPU-only and headless — the tiny 256-256 MLPs and MuJoCo's
CPU-bound stepping don't benefit from a GPU allocation; what the
cluster buys you is **cores**, so all 9 remaining runs go in parallel
instead of 3-at-a-time on a laptop. Request a plain CPU partition, not
a GPU one, unless a GPU partition is the only place with free/fast
cores.

## 1. Connect

```bash
# Connect to the school VPN first if required, then:
ssh <your-username>@xlogin1.comp.nus.edu.sg   # adjust hostname if different
```

## 2. Install uv and clone

`uv` itself (and cloning) are lightweight enough for the login node:

```bash
python3 -m venv ~/.venvs/bootstrap        # if `pip install --user uv` errors with
~/.venvs/bootstrap/bin/pip install uv     # "externally-managed-environment"
echo 'export PATH="$HOME/.venvs/bootstrap/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
uv --version

# Public repo — plain HTTPS clone, no token/SSH-key setup needed.
git clone https://github.com/eugenewong22/RL-Launchpad.git launchpad-rl
cd launchpad-rl
```

## 3. Find your partition name

```bash
sinfo   # lists partitions; pick a CPU partition (e.g. "cpu", "long", "normal")
```

## 4. Run `uv sync` + tests — NOT directly on the login node

`uv sync` (dependency resolution) and `pytest` (imports torch) need more
memory than shared login nodes typically allow per process — symptom:
`memory allocation of N bytes failed` / `Aborted (core dumped)`. Submit
this as its own compute-node job instead:

```bash
mkdir -p logs
sbatch --job-name=setup --partition=<name-from-sinfo> --cpus-per-task=2 --mem=4G \
    --time=00:30:00 --chdir="$PWD" --output=logs/setup.out \
    --wrap="bash scripts/setup_cluster.sh"
squeue -u $USER            # wait for it to finish
cat logs/setup.out         # should end with "17 passed"
```

If `logs/setup.out` shows a network error instead (compute nodes with no
internet), reuse the package cache your laptop already built locally
rather than fetching on the cluster at all:

```bash
# from your laptop:
rsync -av ~/.cache/uv/ <your-username>@xlogin1.comp.nus.edu.sg:~/.cache/uv/
# then on the cluster:
uv sync --offline && uv run pytest
```

## 5. Submit the matrix

`scripts/submit_slurm_matrix.sh` submits one `sbatch` job per missing
run (skips any `results/<run>/` that already exists, so it's safe to
re-run after partial failures):

```bash
PARTITION=<name-from-sinfo> bash scripts/submit_slurm_matrix.sh
squeue -u $USER                              # watch job states
tail -f results/push_td3_her_seed1/progress.csv   # watch any run's progress
```

Override `CPUS`, `MEM`, or `TIME` as env vars if the defaults (2 cores,
4G, 8h) don't fit the queue's limits — a 1M-step FetchPush run took
about 1 hour on a laptop core, so 8h is generous headroom for a shared
cluster core.

## Bringing results home

Each finished run is one self-contained directory: `progress.csv`,
`config.yaml`, and checkpoints. From your laptop:

```bash
rsync -av <your-username>@xlogin1.comp.nus.edu.sg:launchpad-rl/results/ ~/dev/launchpad-rl/results/
cd ~/dev/launchpad-rl
uv run python scripts/make_plots.py
uv run python scripts/final_eval.py
```

Local and cluster runs mix freely — arms/seeds are independent, and
each run's own `progress.csv` records its wall-clock, so note the
cluster's CPU model in the write-up's compute table alongside the
laptop's.
