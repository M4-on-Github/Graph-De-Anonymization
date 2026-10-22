#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Report 1 on cit-HepTh — submit the whole thing.
#
# Three stages, chained with --dependency=afterok so the arrays cannot start on
# pairs that do not exist yet:
#
#   0  generate the beta x noise-draw pairs from the HepTh parent   (1 job)
#   1  seed-based cells: Algorithm 1 at each (beta, train %, theta) (array)
#   2  seed-free cells: each ranking, alone and integrated          (array)
#
# Run from the repo root on the head node:
#
#   bash scripts/moes-cluster-scripts/submit_report1.sh --dry-run   # see the plan
#   bash scripts/moes-cluster-scripts/submit_report1.sh
#
# Submit-specific flags:
#   --dry-run            Freeze the manifests and print the sbatch commands only.
#   --no-generate        Assume the pairs exist; skip stage 0 and the dependency.
#   --only KIND          One of: pairs | seed_based | seed_free.
#   --limit N            Max concurrent array tasks (default 24). This is a
#                        shared cluster; the default is deliberately not
#                        "all of them".
#   --sf-pair-seeds "L"  Noise draws for the seed-free arrays only (default 0).
#                        The rankings are the slow half, and replicating a
#                        deterministic ranking over noise draws costs a k-truss
#                        decomposition per draw.
#   --sb-time / --sf-time    Override the per-cell walltimes.
#
# Everything written outside the repo goes under
# /data/$USER/Graph-De-Anonymization/ -- manifests and logs in logs/, the
# container image beside them. Override the root with GDA_DATA_DIR.
#
# Anything else is passed straight through to grid.py, so the axes are its
# flags and not a second copy of them here:
#
#   --betas 0.0 0.2 0.5 --pair-seeds 0 1 --thetas 0.5 --train-portions 0.2
#   --methods pagerank degree --top-ks 50
#
# Prerequisite: bash scripts/moes-cluster-scripts/build_container.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"

DRY_RUN=false
GENERATE=true
ONLY=""
LIMIT=24
SF_PAIR_SEEDS="0"
SB_TIME=""
SF_TIME=""
GRID_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=true; shift ;;
        --no-generate)   GENERATE=false; shift ;;
        --only)          ONLY="$2"; shift 2 ;;
        --limit)         LIMIT="$2"; shift 2 ;;
        --sf-pair-seeds) SF_PAIR_SEEDS="$2"; shift 2 ;;
        --sb-time)       SB_TIME="$2"; shift 2 ;;
        --sf-time)       SF_TIME="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)               GRID_ARGS+=("$1"); shift ;;
    esac
done

case "$ONLY" in ""|pairs|seed_based|seed_free) ;;
    *) echo "ERROR: --only must be pairs, seed_based or seed_free" >&2; exit 1 ;;
esac

mkdir -p "$LOGS_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SB_MANIFEST="$LOGS_DIR/report1_sb_${STAMP}.json"
SF_MANIFEST="$LOGS_DIR/report1_sf_${STAMP}.json"

if [ ! -f "$SIF" ] && [ "$DRY_RUN" = false ]; then
    echo "ERROR: container not found: $SIF" >&2
    echo "       bash $SCRIPTS/build_container.sh   (needs --mem=64G)" >&2
    exit 1
fi

# ── Freeze the plans ─────────────────────────────────────────────────────────
# The manifest is the contract between this script and the array tasks. Writing
# it here, once, is what stops a later edit to grid.py from silently changing
# what a queued task will do. The trailing --pair-seeds for the seed-free plan
# wins over any earlier one: argparse keeps the last occurrence.
echo "==========================================="
echo " Report 1 on cit-HepTh — submission plan"
echo " Repo     : $REPO"
echo " Stamp    : $STAMP"
echo " Grid     : ${GRID_ARGS[*]:-<defaults>}"
echo "==========================================="

N_PAIRS="$(python3 "$SCRIPTS/grid.py" --kind pairs --count "${GRID_ARGS[@]+"${GRID_ARGS[@]}"}")"

N_SB=0
if [ -z "$ONLY" ] || [ "$ONLY" = "seed_based" ]; then
    python3 "$SCRIPTS/grid.py" --kind seed_based \
        "${GRID_ARGS[@]+"${GRID_ARGS[@]}"}" --manifest "$SB_MANIFEST" >/dev/null
    N_SB="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["n_cells"])' "$SB_MANIFEST")"
fi

N_SF=0
if [ -z "$ONLY" ] || [ "$ONLY" = "seed_free" ]; then
    # shellcheck disable=SC2086  # SF_PAIR_SEEDS is a deliberate word list
    python3 "$SCRIPTS/grid.py" --kind seed_free \
        "${GRID_ARGS[@]+"${GRID_ARGS[@]}"}" --pair-seeds $SF_PAIR_SEEDS \
        --manifest "$SF_MANIFEST" >/dev/null
    N_SF="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["n_cells"])' "$SF_MANIFEST")"
fi

echo " Pairs    : $N_PAIRS"
echo " Seed-based cells : $N_SB   -> $SB_MANIFEST"
echo " Seed-free cells  : $N_SF   -> $SF_MANIFEST"
echo "   (each seed-free cell writes 1 ranking record + one per top-k)"
echo " Concurrency      : $LIMIT tasks"
echo "==========================================="

submit() {   # submit <description> <sbatch args...>
    local desc="$1"; shift
    if [ "$DRY_RUN" = true ]; then
        # To stderr, not stdout: callers that only want the job id redirect
        # stdout to /dev/null, and a dry run that prints nothing is worse than
        # no dry run at all.
        {
            echo "  [dry-run] $desc"
            echo "            sbatch $*"
        } >&2
        echo "DRYRUN"
        return 0
    fi
    local id
    id="$(sbatch --parsable "$@")"
    echo "  $desc -> job $id" >&2
    echo "$id"
}

# ── Stage 0: pairs ───────────────────────────────────────────────────────────
GEN_JOB=""
if [ "$GENERATE" = true ] && { [ -z "$ONLY" ] || [ "$ONLY" = "pairs" ]; }; then
    GEN_JOB="$(submit "generate $N_PAIRS pairs" \
        -p pleiades \
        --output="$LOGS_DIR/gda_generate_%j.out" \
        --error="$LOGS_DIR/gda_generate_%j.err" \
        "$SCRIPTS/00_generate_pairs_job.sh" "${GRID_ARGS[@]+"${GRID_ARGS[@]}"}")"
fi
[ "$ONLY" = "pairs" ] && { echo; echo "Only stage 0 requested; done."; exit 0; }

DEP=()
[ -n "$GEN_JOB" ] && [ "$GEN_JOB" != "DRYRUN" ] && DEP=(--dependency="afterok:$GEN_JOB")

# ── Stages 1 and 2: the arrays ───────────────────────────────────────────────
if [ "$N_SB" -gt 0 ]; then
    submit "seed-based array (0-$((N_SB - 1))%$LIMIT)" \
        -p pleiades \
        "${DEP[@]+"${DEP[@]}"}" \
        --array="0-$((N_SB - 1))%$LIMIT" \
        ${SB_TIME:+--time="$SB_TIME"} \
        --output="$LOGS_DIR/gda_sb_%A_%a.out" \
        --error="$LOGS_DIR/gda_sb_%A_%a.err" \
        "$SCRIPTS/10_seed_based_job.sh" "$SB_MANIFEST" >/dev/null
fi

if [ "$N_SF" -gt 0 ]; then
    submit "seed-free array (0-$((N_SF - 1))%$LIMIT)" \
        -p pleiades \
        "${DEP[@]+"${DEP[@]}"}" \
        --array="0-$((N_SF - 1))%$LIMIT" \
        ${SF_TIME:+--time="$SF_TIME"} \
        --output="$LOGS_DIR/gda_sf_%A_%a.out" \
        --error="$LOGS_DIR/gda_sf_%A_%a.err" \
        "$SCRIPTS/20_seed_free_job.sh" "$SF_MANIFEST" >/dev/null
fi

echo
echo "==========================================="
echo " Monitor:"
echo "   squeue -u $USER"
echo "   tail -f $LOGS_DIR/gda_sb_<ARRAY_JOBID>_0.out"
echo
echo " Results (records are tracked in git, bytes are not):"
echo "   wc -l $REPO/results/runs/index.jsonl"
echo "   python3 $SCRIPTS/collect.py --kind report1_seed_based"
echo
echo " Which task was which:"
echo "   python3 -c \"import json;[print(c['task_id'],c['pair'],c.get('train_portion'),c.get('theta'),c.get('method')) for c in json.load(open('$SB_MANIFEST'))['cells']]\""
echo "==========================================="
