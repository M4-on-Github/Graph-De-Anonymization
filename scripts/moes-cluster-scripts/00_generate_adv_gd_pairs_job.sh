#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Adv-GD suite — step 0: cut the HepTh pairs the whole suite runs on.
#
# Separate from Report 1's 00_generate_pairs_job.sh / grid.py on purpose: this
# suite's pairs go into runs/sweep_<mode>/ (flat layout, adv_gd/sweep.py's
# root; runs/sweep_nominal/ by default), not Report 1's runs/report1_*/
# (--compat-layout, grid.py's root).
# Same generator, same beta math underneath, different directory tree.
#
# One SLURM job for the whole beta x pair_seed grid (adv_gd/sweep.py --generate
# loops over the pairs and shells out to the generator once per pair, so the
# parent graph is parsed once per pair, not once per array cell that later
# runs on it). Idempotent: already-present pairs are skipped.
#
# Do not call directly — submit_adv_gd_suite.sh runs it as step 0. To run it
# alone:
#   sbatch -p pleiades --cpus-per-task=2 --mem=8G --time=1:00:00 \
#          scripts/moes-cluster-scripts/00_generate_adv_gd_pairs_job.sh \
#          --betas 0.0 0.1 0.2 --pair-seeds 0 --beta-mode nominal \
#          --out runs/sweep_nominal
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
