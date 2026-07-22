#!/usr/bin/env bash
# One-time env setup, meant to run on a COMPUTE node via sbatch — the
# login node's tight per-process memory limit aborts `uv sync` /
# `pytest` (torch import + dependency resolution need more than it
# allows; symptom: "memory allocation of N bytes failed / Aborted").
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync
uv run pytest
