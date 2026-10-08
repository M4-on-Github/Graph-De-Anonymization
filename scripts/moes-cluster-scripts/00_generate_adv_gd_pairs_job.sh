#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Adv-GD suite — step 0: cut the HepTh pairs the whole suite runs on.
#
# Separate from Report 1's 00_generate_pairs_job.sh / grid.py on purpose: this
# suite's pairs go into runs/sweep_divergence/ (flat layout, adv_gd/sweep.py's
# root), not Report 1's runs/report1_*/ (--compat-layout, grid.py's root).
# Same generator, same beta math underneath, different directory tree.
#
# One process for the whole beta x pair_seed grid (adv_gd/sweep.py --generate
# shells out to the generator once per beta, same reasoning as Report 1's
# version: the parent graph is loaded once per invocation of this script
# rather than once per cell). Idempotent: already-present pairs are skipped.
#
# Do not call directly — submit_adv_gd_suite.sh runs it as step 0. To run it
# alone:
#   sbatch -p pleiades --cpus-per-task=2 --mem=8G --time=1:00:00 \
#          scripts/moes-cluster-scripts/00_generate_adv_gd_pairs_job.sh \
#          --betas 0.0 0.1 0.2 --pair-seeds 0 --beta-mode divergence \
#          --out runs/sweep_divergence
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -J gda_gen

set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "$SLURM_SUBMIT_DIR/scripts/moes-cluster-scripts/common.sh"
else
    source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"
fi
banner "Adv-GD suite — generate HepTh pairs"
echo " Args: $*"
echo "-------------------------------------------"

in_container "$REPO/src/adv_gd/sweep.py" --generate "$@"

echo "==========================================="
echo " Generation done: $(date)"
echo "==========================================="
