#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# E3 non-adversarial alignment control (Experiment Log 3, section E3) — one
# w_control.py cell per array task: arm B (W_0 from composite_centrality's
# anchors) or arm C (oracle W_0), on one cached embedding pair. Each task
# writes one adv_gd_w_control record.
#
# GPU, like 30_adv_gd_job.sh, so that the CSLS steps run on the same kind of
# device as the arm-A records they are compared with. No GAE is trained and no
# adversarial step runs: w_control.py only READS the cached Z1.npz/Z2.npz, hashes
# them, applies the Log's hash rule and aborts on a mismatch. The cost is the
# fingerprint (arm B) and Phase 3, assumed under the 255-737 s of a full Adv-GD
# cell (the Log takes 8 min; not measured). Without a GPU the driver falls back
# to the CPU by itself; that wall time is not measured either (Phase 3 alone
# took 200-430 s in the existing b10_s0 and b50_s0 Adv-GD logs). Memory: the
# candidate-set step for M = 10000 holds a 10000 x 27k float32 matrix (about
# 1.1 GB, about 2 GB at its peak), hence --mem=32G rather than a small request.
#
# --constraint excludes pleiades-1-3 for the reason given in 30_adv_gd_job.sh
# (its 1080 Ti is sm_61, which the container's torch cannot run).
#
# Do not call directly — submit_w_control.sh sets --array, --output, --error
# and appends the plan flags (--arms, --betas, --pair-seeds, --pair-root,
# --gae-seed, --e1, --status, ...). To run one cell by hand on an interactive
# node:
#   GDA_NV=1 bash scripts/moes-cluster-scripts/33_adv_gd_w_control_job.sh 0 \
#       --arms B --betas 0.1 --pair-seeds 0
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX6000ADA|QUADRORTX6000
#SBATCH --time=02:00:00
#SBATCH -J gda_wc

set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "$SLURM_SUBMIT_DIR/scripts/moes-cluster-scripts/common.sh"
else
    source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:?Usage: 33_adv_gd_w_control_job.sh [TASK_ID] <w_control.py flags...> (or run as a SLURM array task)}}"
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || shift

banner "E3 W-control — cell $TASK_ID"
echo " Args: $*"
echo "-------------------------------------------"

export GDA_NV=1
export PYTHONHASHSEED=0
in_container "$REPO/src/adv_gd/w_control.py" --task-id "$TASK_ID" "$@"

echo "==========================================="
echo " Cell $TASK_ID done: $(date)"
echo "==========================================="
