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

## 2. One-time setup (on the login node — it's the one with internet)

Compute nodes on most SLURM clusters (NUS SoC included) have **no
internet access**, so `uv sync` must run on the login node first; the
resulting `.venv` is then reused by every submitted job.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # installs uv to ~/.local/bin
source ~/.bashrc                                    # or restart the shell, so `uv` is on PATH

# The repo is public — plain HTTPS clone, no token/SSH-key setup needed.
# (If it's ever made private: a fine-grained PAT scoped to this repo with
# Contents: Read-only is enough — nothing in this workflow pushes from
# the cluster, results come home via rsync.)
git clone https://github.com/eugenewong22/RL-Launchpad.git launchpad-rl
cd launchpad-rl

uv sync        # fetches pinned Python 3.11 + all deps into .venv (needs internet)
uv run pytest  # 17 tests should pass before spending any cluster-hours
```

## 3. Find your partition name

```bash
sinfo   # lists partitions; pick a CPU partition (e.g. "cpu", "long", "normal")
```

## 4. Submit the matrix

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
