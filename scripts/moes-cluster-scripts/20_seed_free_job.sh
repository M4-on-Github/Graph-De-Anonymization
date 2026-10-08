#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — one seed-free cell (a ranking, plus its integrations) per array task.
#
# CPU only, and slower than a seed-based cell for two reasons worth knowing
# before you shorten --time: k-truss and k-core decompose the graph by repeated
# subgraph extraction, and every cell also scores the *full* ranking (the
# report's Table 2) before any top-k is integrated.
#
# One task computes the centrality once and writes 1 + len(top_ks) records.
#
# Do not call directly — submit_report1.sh sets --array, --output and --error.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -J gda_sf

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

MANIFEST="${1:?Usage: 20_seed_free_job.sh MANIFEST [TASK_ID]}"
TASK_ID="${2:-${SLURM_ARRAY_TASK_ID:-}}"
: "${TASK_ID:?no task id: pass one, or run this as a SLURM array task}"

[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST" >&2; exit 1; }

banner "Report 1 — seed-free cell $TASK_ID"
echo " Manifest: $MANIFEST"
echo "-------------------------------------------"

in_container "$SCRIPTS/report1_cell.py" \
    --manifest "$MANIFEST" \
    --task-id "$TASK_ID" \
    --status exploratory

echo "==========================================="
echo " Cell $TASK_ID done: $(date)"
echo "==========================================="
