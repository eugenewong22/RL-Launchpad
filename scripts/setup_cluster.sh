#!/usr/bin/env bash
# One-time env setup, meant to run on a COMPUTE node via sbatch — the
# login node's tight per-process memory limit aborts `uv sync` /
# `pytest` (torch import + dependency resolution need more than it
# allows; symptom: "memory allocation of N bytes failed / Aborted").
set -euo pipefail
# Under sbatch, $0 is SLURM's spooled copy of this script, not this file, so
# `dirname $0`/.. lands in /var/spool. Use the submitting directory instead;
# fall back to $0 for direct `bash scripts/...` runs.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# sbatch jobs don't source ~/.bashrc, so PATH edits there don't apply —
# add every place `uv` might have been installed explicitly.
export PATH="$HOME/.venvs/bootstrap/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || { echo "uv not found on PATH: $PATH" >&2; exit 1; }
uv sync
uv run pytest
