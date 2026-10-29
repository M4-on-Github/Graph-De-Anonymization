#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# E2 ignition grid — one ignition_grid.py array task per SLURM array task (one
# instance and one n group of a block: 1 to 9 runs of `propagate` from a
# synthetic anchor set; see src/ignition_grid/ignition_plan.py).
#
# CPU only — `propagate` is pure python and the runner builds no embedding.
# Resources are 31_baselines_job.sh's: 2 CPUs, 24G, 2 h. A task of 9 runs at the
# 600 s per-run cap is 90 min (the spec's Cost item), inside the limit.
# The task's content is rebuilt deterministically from --block and the task id,
# so nothing is frozen at submission (the same design as sweep.py).
#
# Do not call directly — submit_ignition_grid.sh sets --array, --output, --error
# and appends --block, --pair-root, --out, --status and --note.
# To run one task by hand on an interactive node:
#   bash scripts/moes-cluster-scripts/32_ignition_grid_job.sh 2 --block base
# To run part of one: add --cells 1000:5,1000:200
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH -J gda_ig

set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "$SLURM_SUBMIT_DIR/scripts/moes-cluster-scripts/common.sh"
else
    source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:?Usage: 32_ignition_grid_job.sh [TASK_ID] <ignition_grid.py flags...> (or run as a SLURM array task)}}"
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || shift

banner "E2 ignition grid — task $TASK_ID"
echo " Args: $*"
echo "-------------------------------------------"

export PYTHONHASHSEED=0
in_container "$REPO/src/ignition_grid/ignition_grid.py" --task-id "$TASK_ID" "$@"

echo "==========================================="
echo " Task $TASK_ID done: $(date)"
echo "==========================================="
