#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — cut the HepTh noise pairs the whole grid runs on.
#
# One process for every pair: the parent graph is loaded once and 18 pairs are
# cut from it, which is strictly cheaper than 18 processes that each parse
# 350k edges. Downloads the parent first if it is not on disk.
#
# Idempotent in the sense that matters: re-running regenerates each pair from
# the same (beta, seed), and the generator reports on stderr if it would be
# replacing a pair made with a *different* configuration.
#
# Do not call directly — submit_report1.sh runs it as step 0. To run it alone:
#   sbatch -p pleiades --cpus-per-task=2 --mem=32G --time=2:00:00 \
#          --output=/data/$USER/Graph-De-Anonymization/logs/gda_generate_%j.out \
#          scripts/moes-cluster-scripts/00_generate_pairs_job.sh --beta-mode nominal
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -J gda_generate

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
banner "Report 1 — generate HepTh noise pairs"

# Everything after the script name is passed through to grid.py, so the submit
# script's axis flags reach the generator unchanged.
GRID_ARGS=("$@")

# ── The parent graph ─────────────────────────────────────────────────────────
# Only cit-HepTh.txt.gz is fetched. The full SNAP archive also unpacks ~29,000
# tiny .abs abstract files, and globbing or du-ing that directory on the NAS is
# slow enough to look like a hang (see data/README.md).
if [ ! -f "$PARENT_GRAPH" ]; then
    echo "[$(date)] Parent graph absent; fetching cit-HepTh.txt.gz from SNAP ..."
    mkdir -p "$(dirname "$PARENT_GRAPH")"
    TMP_GZ="$(mktemp "$DATA_DIR/cit-HepTh.XXXXXX.txt.gz")"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$TMP_GZ" https://snap.stanford.edu/data/cit-HepTh.txt.gz
    else
        wget -qO "$TMP_GZ" https://snap.stanford.edu/data/cit-HepTh.txt.gz
    fi
    gunzip -c "$TMP_GZ" > "$PARENT_GRAPH"
    rm -f "$TMP_GZ"
    echo "  wrote $PARENT_GRAPH ($(wc -l < "$PARENT_GRAPH") lines)"
else
    echo "Parent graph present: $PARENT_GRAPH"
fi

# ── Generate ─────────────────────────────────────────────────────────────────
# grid.py composes the argument list so that the betas generated here and the
# pair names the cells look for come from one place. It runs on the head node's
# python 3.6 as well as in the container, hence the host python3 here.
GEN_ARGV="$(python3 "$SCRIPTS/grid.py" --kind pairs --gen-argv "${GRID_ARGS[@]}")"
echo "Generator arguments:"
echo "  $GEN_ARGV"
echo

# shellcheck disable=SC2086  # GEN_ARGV is shell-quoted by grid.py on purpose
eval set -- $GEN_ARGV
in_container "$REPO/src/beta_noise_generator/perturbation.py" "$@"

echo
echo "Pairs on disk:"
python3 "$SCRIPTS/grid.py" --kind pairs --plan "${GRID_ARGS[@]}" | \
    while read -r line; do echo "  $line"; done
echo "==========================================="
echo " Generation done: $(date)"
echo "==========================================="
