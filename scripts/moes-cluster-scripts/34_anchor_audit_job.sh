#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# E2.0 anchor audit — one anchor_audit.py array task per SLURM array task (one
# pair, five centrality measures, one anchor_audit record each; no propagate).
#
# CPU only. Resources are 31_baselines_job.sh's (2 CPUs, 24G, 2 h): the cost is
# the five centrality computations, about 85 s per record (spec, Cost item).
#
# Do not call directly — submit_ignition_grid.sh sets --array, --output, --error
# and appends --group, --pair-root, --out, --status and --note.
# To run one pair by hand on an interactive node:
#   bash scripts/moes-cluster-scripts/34_anchor_audit_job.sh 0 --group high
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH -J gda_aud

set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "$SLURM_SUBMIT_DIR/scripts/moes-cluster-scripts/common.sh"
else
    source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:?Usage: 34_anchor_audit_job.sh [TASK_ID] <anchor_audit.py flags...> (or run as a SLURM array task)}}"
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || shift

banner "E2.0 anchor audit — task $TASK_ID"
echo " Args: $*"
echo "-------------------------------------------"

export PYTHONHASHSEED=0
in_container "$REPO/src/ignition_grid/anchor_audit.py" --task-id "$TASK_ID" "$@"

echo "==========================================="
echo " Task $TASK_ID done: $(date)"
echo "==========================================="
