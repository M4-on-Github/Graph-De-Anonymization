# ─────────────────────────────────────────────────────────────────────────────
# Shared setup for every script in this directory. Source it, do not run it.
#
#   source "$(dirname "$0")/common.sh"
#
# Defines: REPO, DATA_DIR, LOGS_DIR, SIF, PARENT_GRAPH, and in_container().
# ─────────────────────────────────────────────────────────────────────────────

# sbatch copies the submitted script into /var/spool/slurmd/jobNNN/, so
# BASH_SOURCE resolves to the spool copy and the repo is not above it. Inside a
# real job the repo is SLURM_SUBMIT_DIR; the SLURM_JOB_ID test guards against a
# stale SLURM_SUBMIT_DIR inherited by an interactive shell.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
fi
SCRIPTS="$REPO/scripts/moes-cluster-scripts"

# Everything this project writes outside the repo lives under one directory on
# the NAS, mirroring the repo's own name, so that the project can be found or
# deleted in one place rather than being spread through /data/$USER alongside
# every other project's logs and images.
DATA_DIR="${GDA_DATA_DIR:-/data/$USER/Graph-De-Anonymization}"
LOGS_DIR="$DATA_DIR/logs"
SIF="${GDA_SIF:-$DATA_DIR/graph_deanon.sif}"
PARENT_GRAPH="$REPO/data/cit-HepTH/cit-HepTh.txt"

if [ ! -d "$REPO/src/adv_gd" ]; then
    echo "ERROR: $REPO does not look like the repo (no src/adv_gd)." >&2
    echo "       Submit from the repo root so SLURM_SUBMIT_DIR points at it." >&2
    exit 1
fi

# in_container <python args...>
#
# Only the repo and this project's NAS directory are bound, not the whole of
# /data/$USER: a job has no business reading another project's data.
#
# --containall is deliberate: without it the host's PYTHONPATH, pip --user
# site-packages under ~/.local and a stale MPLBACKEND all leak in, and the run
# record would name library versions the job did not actually use. Everything
# the job needs is bound explicitly.
#
# PYTHONHASHSEED is set, not inherited. The inherited baseline is hash-order
# sensitive; report1_cell.py refuses to run without it and the value goes into
# the record.
in_container() {
    if [ ! -f "$SIF" ]; then
        echo "ERROR: container not found: $SIF" >&2
        echo "       Build it: bash $SCRIPTS/build_container.sh" >&2
        echo "       Or point GDA_SIF at an existing image." >&2
        exit 1
    fi
    apptainer exec \
        --containall \
        ${GDA_NV:+--nv} \
        --pwd "$REPO" \
        --bind "$REPO:$REPO" \
        --bind "$DATA_DIR:$DATA_DIR" \
        --bind /tmp:/tmp \
        --env USER="$USER" \
        --env PYTHONUNBUFFERED=1 \
        --env PYTHONHASHSEED="${PYTHONHASHSEED:-0}" \
        --env MPLBACKEND=Agg \
        "$SIF" \
        python3 "$@"
}

# Printed by every job so that a log says which code and which image produced it.
banner() {
    echo "==========================================="
    echo " $1"
    echo " Job     : ${SLURM_JOB_ID:-interactive}${SLURM_ARRAY_TASK_ID:+ (task $SLURM_ARRAY_TASK_ID)}"
    echo " Node    : $(hostname)"
    echo " Started : $(date)"
    echo " Repo    : $REPO"
    echo " Commit  : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)$([ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] && echo ' (dirty)')"
    echo " SIF     : $SIF"
    echo "==========================================="
}
