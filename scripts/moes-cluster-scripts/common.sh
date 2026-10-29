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
# Report 1's parent (cit-HepTh), read only by 00_generate_pairs_job.sh. The Adv-GD
# suite does not use it: its parent comes from sweep.py's PARENTS, chosen with
# --dataset, so this does not need to follow the dataset.
PARENT_GRAPH="$REPO/data/cit-HepTH/cit-HepTh.txt"

if [ ! -d "$REPO/src/adv_gd" ]; then
    echo "ERROR: $REPO does not look like the repo (no src/adv_gd)." >&2
    echo "       Submit from the repo root so SLURM_SUBMIT_DIR points at it." >&2
    exit 1
fi

# stage_sif_locally
#
# apptainer mmaps the .sif's squashfs directly; with --limit 24 concurrent
# tasks on a node all mmapping the same copy over NFS from $DATA_DIR, an NFS
# hiccup under that load surfaces as SIGBUS (signal 7, exit 135) on the mmap'd
# page fault -- observed in production on job 49921, always *after* the cell's
# own python had already returned 0 and written its record (confirmed against
# results/runs/index.jsonl), never during computation. Fix: copy the image
# once per node onto node-local tmpfs (/tmp here is tmpfs, not NFS -- checked
# via `mount`) and mmap that copy instead. flock serializes the first task on
# a node that finds no local copy yet; every task after it just stat()s and
# moves on.
LOCAL_SIF_DIR="/tmp/${USER}-gda-sif"
LOCAL_SIF="$LOCAL_SIF_DIR/$(basename "$SIF")"

stage_sif_locally() {
    mkdir -p "$LOCAL_SIF_DIR"
    (
        flock -x 200
        local want
        want="$(stat -c%s "$SIF")"
        if [ ! -f "$LOCAL_SIF" ] || [ "$(stat -c%s "$LOCAL_SIF" 2>/dev/null)" != "$want" ]; then
            cp "$SIF" "$LOCAL_SIF.tmp"
            mv "$LOCAL_SIF.tmp" "$LOCAL_SIF"
        fi
    ) 200>"$LOCAL_SIF_DIR/.stage.lock"
}

# in_container <python args...>
#
# Only the repo and this project's NAS directory are bound, not the whole of
# /data/$USER: a job has no business reading another project's data.
#
# --containall is deliberate: without it the host's PYTHONPATH and a stale
# MPLBACKEND leak in, and the run record would name library versions the job
# did not actually use. Everything the job needs is bound explicitly.
#
# --containall does NOT stop pip --user site-packages under ~/.local: the
# cluster's apptainer.conf has `mount hostfs = yes`, so the real home is visible
# and its site-packages sorts ahead of the container's own pinned ones.
# PYTHONNOUSERSITE=1 is what closes that.
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
    stage_sif_locally
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
        --env PYTHONNOUSERSITE=1 \
        "$LOCAL_SIF" \
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
