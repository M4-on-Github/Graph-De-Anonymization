#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Adv-GD suite — one sweep.py cell per array task (full Adv-GD or the
# skip_adversarial ablation, selected by which flags submit_adv_gd_suite.sh
# passes through as $@).
#
# GPU (GAE training in Phase 1 needs it). The plan is rebuilt deterministically
# from the same CLI flags every task receives (sweep.py's own design, not a
# frozen manifest like Report 1's grid.py/report1_cell.py) -- see sweep.py's
# docstring for why that is safe across array tasks.
#
# --constraint excludes pleiades-1-3 (feature "1080Ti"): its Pascal-era GPUs
# are compute capability 6.1, and the container's torch wheel (cu130) ships
# kernels starting at sm_75 -- a task landing there dies immediately with
# `CUDA error: no kernel image is available for execution on the device`.
# Confirmed via `sinfo -N -o "%N %f %G" -p pleiades`: pleiades-0-17/-0-23 are
# RTX6000ADA, pleiades-1-10 is QUADRORTX6000 (both new enough), pleiades-1-3
# is the lone 1080Ti node -- this whitelists the two known-good families
# rather than excluding by hostname, so it stays correct if the cluster's
# node set changes.
#
# Do not call directly — submit_adv_gd_suite.sh sets --array, --output,
# --error and appends the sweep flags (--betas, --pair-seeds, --align-seeds,
# --skip-adversarial, etc.).
# To run one cell by hand on an interactive node:
#   GDA_NV=1 bash scripts/moes-cluster-scripts/30_adv_gd_job.sh --task-id 0 --betas 0.0
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX6000ADA|QUADRORTX6000
#SBATCH --time=04:00:00
#SBATCH -J gda_adv

set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "$SLURM_SUBMIT_DIR/scripts/moes-cluster-scripts/common.sh"
else
    source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:?Usage: 30_adv_gd_job.sh [TASK_ID] <sweep.py flags...> (or run as a SLURM array task)}}"
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || shift

banner "Adv-GD suite — cell $TASK_ID"
echo " Args: $*"
echo "-------------------------------------------"

export GDA_NV=1
export PYTHONHASHSEED=0
in_container "$REPO/src/adv_gd/sweep.py" --task-id "$TASK_ID" "$@"

echo "==========================================="
echo " Cell $TASK_ID done: $(date)"
echo "==========================================="
