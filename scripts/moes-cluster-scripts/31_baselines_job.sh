#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Adv-GD suite — one sweep_baselines.py cell per array task (structural
# fingerprint, IsoRank, centrality rank-matching or composite centrality,
# selected by the --method flag submit_adv_gd_suite.sh passes through as $@).
#
# CPU only — none of the four baselines touches a GPU. Plan is rebuilt
# deterministically from CLI flags per task, same as 30_adv_gd_job.sh.
#
# Do not call directly — submit_adv_gd_suite.sh sets --array, --output,
# --error and appends the sweep flags (--method, --betas, --pair-seeds, etc.).
# To run one cell by hand on an interactive node:
#   bash scripts/moes-cluster-scripts/31_baselines_job.sh 0 --method isorank --betas 0.0
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH -J gda_bl

set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "$SLURM_SUBMIT_DIR/scripts/moes-cluster-scripts/common.sh"
else
    source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:?Usage: 31_baselines_job.sh [TASK_ID] <sweep_baselines.py flags...> (or run as a SLURM array task)}}"
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || shift

banner "Baselines suite — cell $TASK_ID"
echo " Args: $*"
echo "-------------------------------------------"

export PYTHONHASHSEED=0
in_container "$REPO/src/baselines/sweep_baselines.py" --task-id "$TASK_ID" "$@"

echo "==========================================="
echo " Cell $TASK_ID done: $(date)"
echo "==========================================="
