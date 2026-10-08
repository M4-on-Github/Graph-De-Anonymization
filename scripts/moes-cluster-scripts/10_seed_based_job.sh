#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — one seed-based cell (Report 1's Algorithm 1) per array task.
#
# CPU only. The cell is read out of the manifest frozen at submit time, so
# editing grid.py's defaults cannot retarget a job already in the queue.
#
# Do not call directly — submit_report1.sh sets --array, --output and --error.
# To run one cell by hand on an interactive node:
#   bash scripts/moes-cluster-scripts/10_seed_based_job.sh MANIFEST 0
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -J gda_sb

set -euo pipefail
# sbatch copies this script into /var/spool/slurmd/jobNNN/ before running it,
# so dirname "$0" is the spool dir, not this directory. Under a real job the
# repo is SLURM_SUBMIT_DIR (set by sbatch to the submitter's cwd, which
# submit_report1.sh guarantees is the repo root); only fall back to the
# dirname trick when run directly, outside SLURM.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "$SLURM_SUBMIT_DIR/scripts/moes-cluster-scripts/common.sh"
else
    source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"
fi

MANIFEST="${1:?Usage: 10_seed_based_job.sh MANIFEST [TASK_ID]}"
TASK_ID="${2:-${SLURM_ARRAY_TASK_ID:-}}"
: "${TASK_ID:?no task id: pass one, or run this as a SLURM array task}"

[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST" >&2; exit 1; }

banner "Report 1 — seed-based cell $TASK_ID"
echo " Manifest: $MANIFEST"
python3 "$SCRIPTS/grid.py" --kind seed_based --count >/dev/null 2>&1 || true
echo "-------------------------------------------"

# The whole point of one process per cell: a failure here costs this cell and
# nothing else, and report1_cell.py writes the record either way.
in_container "$SCRIPTS/report1_cell.py" \
    --manifest "$MANIFEST" \
    --task-id "$TASK_ID" \
    --status exploratory

echo "==========================================="
echo " Cell $TASK_ID done: $(date)"
echo "==========================================="
