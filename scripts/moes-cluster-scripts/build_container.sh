#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build the Graph-De-Anonymization Apptainer container.
#
# Run once before submit_report1.sh. Safe to re-run: the build is skipped when
# the .def file's hash matches the one recorded beside the SIF.
#
# Submit it as a batch job from the repo root — 64G is not generous, it is
# required. A build in a 4G interactive session is killed (exit 137) while pip
# unpacks the 550 MB cudnn wheel, and mksquashfs then needs room for the image
# on top of that:
#
#   sbatch --partition=pleiades --cpus-per-task=8 --mem=64G --time=1:30:00 \
#          --job-name=build_gda --output=/data/$USER/Graph-De-Anonymization/logs/build_gda_%j.out \
#          scripts/moes-cluster-scripts/build_container.sh
#
# An interactive node works too if the session was allocated the memory:
#
#   srun -p pleiades --cpus-per-task=8 --mem=64G --time=1:30:00 --pty bash
#   bash scripts/moes-cluster-scripts/build_container.sh
#
#   --force   Rebuild even when the hash matches.
#
# Writes: /data/$USER/Graph-De-Anonymization/graph_deanon.sif
#         (+ .def.sha256 beside it, and logs/ under the same directory)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# sbatch stages this script to /var/spool/slurmd/jobNNN/, where the .def file
# next to it does not exist. Inside a real job the repo is SLURM_SUBMIT_DIR;
# the SLURM_JOB_ID test is the guard against a stale SLURM_SUBMIT_DIR leaking
# into an interactive shell.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}/scripts/moes-cluster-scripts" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi

# One project directory on the NAS; see common.sh.
DATA_DIR="${GDA_DATA_DIR:-/data/$USER/Graph-De-Anonymization}"
SIF="${GDA_SIF:-$DATA_DIR/graph_deanon.sif}"
DEF="$SCRIPT_DIR/container_report1.def"
HASH_FILE="$SIF.def.sha256"

FORCE=false
[ "${1:-}" = "--force" ] && FORCE=true

mkdir -p "$DATA_DIR/logs"

echo "==========================================="
echo " Graph-De-Anonymization container build"
echo " Node    : $(hostname)"
echo " Started : $(date)"
echo " DEF     : $DEF"
echo " SIF     : $SIF"
echo "==========================================="

[ -f "$DEF" ] || { echo "ERROR: $DEF not found" >&2; exit 1; }
DEF_HASH="$(sha256sum "$DEF" | cut -d' ' -f1)"

if [ -f "$SIF" ] && [ -f "$HASH_FILE" ] && \
   [ "$DEF_HASH" = "$(cat "$HASH_FILE")" ] && [ "$FORCE" = false ]; then
    echo "SIF is up to date with the .def (hash ${DEF_HASH:0:12}); nothing to do."
    echo "Pass --force to rebuild anyway."
    exit 0
fi

# mksquashfs assembles the image in APPTAINER_TMPDIR, which defaults to /tmp --
# RAM-backed on these nodes. Point it at the NAS so a multi-GB image does not
# have to fit in memory.
export APPTAINER_TMPDIR="$DATA_DIR/apptainer_tmp"
export APPTAINER_CACHEDIR="$DATA_DIR/apptainer_tmp/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "[$(date)] Building (this pulls torch; expect 10-20 minutes) ..."
apptainer build --fakeroot --force "$SIF" "$DEF"

printf '%s\n' "$DEF_HASH" > "$HASH_FILE"

echo "[$(date)] Verifying ..."
apptainer exec --containall --env PYTHONNOUSERSITE=1 "$SIF" python3 -c \
    "import networkx, numpy, matplotlib, torch, subprocess; \
print('python ok;', 'git', subprocess.run(['git','--version'],capture_output=True,text=True).stdout.strip())"

echo "==========================================="
echo " Built  : $SIF"
echo " Size   : $(du -h "$SIF" | cut -f1)"
echo " Hash   : $DEF_HASH"
echo " Next   : bash scripts/moes-cluster-scripts/submit_report1.sh --dry-run"
echo "==========================================="
