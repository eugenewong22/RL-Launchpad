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
cat logs/setup.out         # should end with "N passed", no failures
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
Measured, not guessed: a 1M-step FetchPush run takes **42 min** on an
M-series laptop core and **~4.8 h** on a cluster Xeon core — 6.7x slower
per core (`results/compute_table.md`). That is why `long`'s 8 h default is
the right choice here, and why the cluster's value is running 12 arms
*concurrently*, not finishing any one of them faster.

## Bringing results home

Each finished run is one self-contained directory: `progress.csv`,
`config.yaml`, checkpoints, and `normalizer.npz`.

**The obvious route often does not work.** `rsync` from the laptop needs
laptop -> cluster reachability, and the NUS VPN's Student profile installs
a `192.168.0/16` route that collides with a home router's subnet, so `ssh`
to `xlogin1` times out even while the VPN reports "connected":

```bash
rsync -av YOUR_NUS_USER@xlogin1.comp.nus.edu.sg:launchpad-rl/results/ ~/dev/launchpad-rl/results/
```

**Use GitHub as the rendezvous instead.** The cluster can reach GitHub even
when the laptop cannot reach the cluster — the direction that matters is
cluster -> internet, which works:

```bash
# On the cluster. Checkpoints are ~1.7 MB each; ~24 MB for the full matrix.
cd ~/launchpad-rl
git add results/*/checkpoint_best.pt results/*/checkpoint_latest.zip \
        results/*/normalizer.npz results/*/progress.csv results/*/config.yaml
git commit -m "cluster: checkpoints for all reported runs"
git pull --no-rebase --no-edit && git push
```

Pushing needs credentials, and both SSH routes may be firewalled — port 22
and GitHub's port-443 SSH endpoint (`ssh.github.com`) were both blocked from
this cluster. HTTPS works, so use a **fine-grained** personal access token
scoped to this one repo with `Contents: read and write`, held in memory
only:

```bash
git config --local credential.helper 'cache --timeout=3600'
git push          # username, then paste the token at the password prompt
git credential-cache exit && git config --local --unset credential.helper
```

Do not put the token in the remote URL — that writes it in cleartext into
`.git/config` on shared cluster storage.

Then, on the laptop:

```bash
cd ~/dev/launchpad-rl
git pull --no-rebase          # merge, not rebase; see the gotcha below
uv run python scripts/make_plots.py
uv run python scripts/final_eval.py --env-id FetchPush-v4
```

Local and cluster runs mix freely — arms/seeds are independent, and
each run's own `progress.csv` records its wall-clock, so note the
cluster's CPU model in the write-up's compute table alongside the
laptop's.

## Gotchas that cost real time

**`sbatch` scripts must locate the repo via `$SLURM_SUBMIT_DIR`, not `$0`.**
SLURM ships the script's *contents* to the compute node and runs a spooled
copy at `/var/spool/slurmd/job<N>/slurm_script`. So `$0` is not your file,
and the usual `cd "$(dirname "$0")/.."` lands in `/var/spool` — where there
is no `.venv`, so the job dies in seconds looking like a mystery. Every
script here starts with:

```bash
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # fallback for `bash scripts/...`
```

**Never run torch on the login node.** It is older hardware than the `xcn*`
compute nodes, so importing torch/numpy dies with `Illegal instruction
(core dumped)` — SIGILL, i.e. the wheels use CPU instructions the login node
lacks. It also has a per-process memory cap that aborts `uv sync`. Anything
that imports torch goes through `sbatch`.

**`sacct` distinguishes the two failure modes** that look identical in
`squeue`:

```bash
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode,Reason
```

`ExitCode 1:0` means your script exited nonzero. `0:9` with `CANCELLED`
means SLURM killed it — `Reason` says why (usually walltime or memory).

**`git pull` refuses to overwrite untracked files.** If the cluster has
untracked `results/<run>/` directories that were later committed upstream,
the pull aborts — loudly, but a `cmd1 && cmd2` chain then continues
quietly and the next step runs against stale code. Check `git status`
before assuming a pull landed.

**Merge, do not rebase, when both sides added the same paths.** Both the
laptop and the cluster commit into `results/`. Merge compares against the
common ancestor and resolves silently; rebase replays "add X" onto a tree
where X already exists and conflicts once per file.

**Diagnostic scripts must not use `set -e`.** Fail-fast collapses a
multi-boundary investigation into whichever boundary broke first. When a
job dies for unknown reasons, print every boundary — `$0`, `$PWD`,
`hostname`, `ls .venv`, `which python` — and let it run to the end.

**Quote the heredoc delimiter** (`<<'EOF'`) when writing a script from the
login node, or the submitting shell expands `$(hostname)`, `$PY` and `$0`
into the file before SLURM ever sees it.
