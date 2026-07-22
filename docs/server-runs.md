# Running the experiment matrix on the school (SLURM) cluster

Training is CPU-only and headless — the tiny 256-256 MLPs and MuJoCo's
CPU-bound stepping don't benefit from a GPU allocation; what the
cluster buys you is **cores**, so all 9 remaining runs go in parallel
instead of 3-at-a-time on a laptop. Use the `normal`/`long` CPU
partitions (same `xcn*` nodes), not `gpu`/`gpu-long`.

Replace `YOUR_NUS_USER` below with your actual username — do it as a
plain find-and-replace, not by pasting a placeholder wrapped in `<...>`:
bash treats bare `<`/`>` as redirection operators even with no spaces
around them, so a literal `<name>` in a command tries to open a file
called `name` and fails instead of erroring obviously.

## 1. Connect

```bash
# Connect to the school VPN first if required, then:
ssh YOUR_NUS_USER@xlogin1.comp.nus.edu.sg
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

## 3. Run `uv sync` + tests — NOT directly on the login node

`uv sync` (dependency resolution) and `pytest` (imports torch) need more
memory than shared login nodes typically allow per process — symptom:
`memory allocation of N bytes failed` / `Aborted (core dumped)`. Submit
this as its own compute-node job instead. 30 minutes comfortably fits
either CPU partition's time cap, so `normal` (more idle nodes) is fine
here:

```bash
mkdir -p logs
sbatch --job-name=setup --partition=normal --cpus-per-task=2 --mem=4G \
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
rsync -av ~/.cache/uv/ YOUR_NUS_USER@xlogin1.comp.nus.edu.sg:~/.cache/uv/
# then on the cluster:
uv sync --offline && uv run pytest
```

## 4. Submit the matrix

`scripts/submit_slurm_matrix.sh` submits one `sbatch` job per missing
run (skips any `results/<run>/` that already exists, so it's safe to
re-run after partial failures). It defaults to `partition=long` because
its 8h-per-run time request exceeds `normal`'s 3h cap (both partitions
share the same nodes — `long` just allows longer walltime):

```bash
bash scripts/submit_slurm_matrix.sh
squeue -u $USER                                   # watch job states
tail -f results/push_td3_her_seed1/progress.csv   # watch any run's progress
```

Override defaults via env vars if needed, e.g. a shorter, `normal`-partition
run: `PARTITION=normal TIME=02:30:00 bash scripts/submit_slurm_matrix.sh`
(a 1M-step FetchPush run took about 1 hour on a laptop core, so even 2.5h
has margin — but cluster cores may be slower, so `long`'s default 8h is
the safer choice unless the queue is backed up).

## Bringing results home

Each finished run is one self-contained directory: `progress.csv`,
`config.yaml`, and checkpoints. From your laptop:

```bash
rsync -av YOUR_NUS_USER@xlogin1.comp.nus.edu.sg:launchpad-rl/results/ ~/dev/launchpad-rl/results/
cd ~/dev/launchpad-rl
uv run python scripts/make_plots.py
uv run python scripts/final_eval.py
```

Local and cluster runs mix freely — arms/seeds are independent, and
each run's own `progress.csv` records its wall-clock, so note the
cluster's CPU model in the write-up's compute table alongside the
laptop's.
