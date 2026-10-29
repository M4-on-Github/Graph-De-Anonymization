#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# E2 (Experiment Log 3): the Phase-3 ignition-threshold grid and its anchor
# audit — submit.
#
# One CPU array per block, over pairs that already exist (nothing here makes a
# pair). `src/ignition_grid/ignition_plan.py` owns the grid, so an array's size
# and a task's content cannot disagree; this script asks it for the counts.
#
#   block           runs  tasks  what                                   label
#   base             432     54  18 instances x 24 cells, by n group    confirmatory
#   scheme            54      9  degree-matched wrong anchors, n=1000   confirmatory
#   control            6      2  singleton candidates                   confirmatory
#   placement         54      9  uniform placement, n=1000              exploratory
#   lowoverlap        12      3  o=0.36 probe, n=1000                   exploratory
#   control_rerun      6      2  NOT in the 558: the empty-candidate     (--only only)
#                               reruns behind P2.6's "a rerun ... is also identical"
#   audit_high        45      9  E2.0, 9 pairs with overlap >= 0.7985   confirmatory
#   audit_low         65     13  E2.0, 13 pairs with overlap <= 0.6985  exploratory
#
# The label is the spec's "Covered records". It goes on every array of the call,
# so submit the two kinds separately, which is what the defaults do:
#
#   bash scripts/moes-cluster-scripts/submit_ignition_grid.sh --dry-run --status confirmatory
#   bash scripts/moes-cluster-scripts/submit_ignition_grid.sh --status confirmatory
#   bash scripts/moes-cluster-scripts/submit_ignition_grid.sh --status exploratory
#
# Run from the repo root on the head node. Submit-specific flags:
#   --dry-run           Print the sbatch commands only, submit nothing, write nothing.
#   --status S          exploratory | confirmatory (default exploratory). Pass
#                        confirmatory explicitly, and only for blocks the spec
#                        covers: it is refused for placement, lowoverlap and
#                        audit_low, and with --only naming any of them. Every
#                        record of the call gets this label
#                        (docs/experiment_logging_policy.md).
#   --only "B ..."      Blocks to submit, space-separated, in quotes. Default: the
#                        blocks the spec gives the --status label, i.e.
#                        confirmatory = base scheme control audit_high,
#                        exploratory  = placement lowoverlap audit_low.
#                        control_rerun is never a default.
#   --limit N           Max concurrent array tasks per block (default 12, as
#                        submit_adv_gd_suite.sh). About 4 h at 12, 2 h at 24.
#   --pair-root "D ..." The existing pair roots, space-separated in quotes
#                        (default "runs/sweep_nominal runs/sweep_divergence").
#   --out DIR           Root for output mappings and anchor lists (default
#                        runs/ignition_grid).
#   --note TEXT         Prefix for the records' note.
#   --time T            Per-task walltime, passed to sbatch (default: the job
#                        script's 02:00:00). A 9-run task leaves 800 s per run
#                        in 2 h, and the logs hold propagate times up to 1,369 s
#                        (spec, Cost); a task killed at the limit loses its
#                        remaining runs, so raise this if the first tasks run long.
#
# Before submitting, every pair the blocks need is resolved under --pair-root and
# its overlap checked (a failure stops a real submission; a dry run only warns).
# Records land in results/runs/ (kinds ignition_grid and anchor_audit); logs under
# /data/$USER/Graph-De-Anonymization/logs/, as for the rest of the suite.
#
# Prerequisite: bash scripts/moes-cluster-scripts/build_container.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"

DRY_RUN=false
ONLY=""
LIMIT=12
STATUS="exploratory"
PAIR_ROOT="runs/sweep_nominal runs/sweep_divergence"
OUT="runs/ignition_grid"
TIME=""
NOTE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --only)         ONLY="$2"; shift 2 ;;
        --limit)        LIMIT="$2"; shift 2 ;;
        --status)       STATUS="$2"; shift 2 ;;
        --pair-root)    PAIR_ROOT="$2"; shift 2 ;;
        --out)          OUT="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --note)         NOTE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: unrecognized argument: $1" >&2; exit 1 ;;
    esac
done

KNOWN_BLOCKS="base scheme control placement lowoverlap control_rerun audit_high audit_low"
case "$STATUS" in exploratory|confirmatory) ;;
    *) echo "ERROR: --status must be exploratory or confirmatory" >&2; exit 1 ;;
esac
if ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || [ "$LIMIT" -lt 1 ]; then
    echo "ERROR: --limit must be a positive integer" >&2; exit 1
fi

# The plan runs on the head node's python3 (3.6): the module is standard library only.
PLAN=(python3 "$REPO/src/ignition_grid/ignition_plan.py")
if ! "${PLAN[@]}" --summary >/dev/null; then
    echo "ERROR: could not run ${PLAN[*]} (needs python3)" >&2; exit 1
fi

if [ -z "$ONLY" ]; then
    BLOCKS="$("${PLAN[@]}" --default-blocks "$STATUS")"
else
    BLOCKS="$ONLY"
fi
for b in $BLOCKS; do
    case " $KNOWN_BLOCKS " in *" $b "*) ;;
        *) echo "ERROR: unknown block '$b'; one of: $KNOWN_BLOCKS" >&2; exit 1 ;;
    esac
done

# The label must be one the spec gives the block (Covered records).
for b in $BLOCKS; do
    cov="$("${PLAN[@]}" --block "$b" --coverage)"
    if [ "$STATUS" = confirmatory ] && [ "$cov" != confirmatory ]; then
        echo "ERROR: block '$b' is exploratory in the spec (no prediction covers it);" >&2
        echo "       refusing --status confirmatory for it." >&2
        exit 1
    fi
    if [ "$STATUS" = exploratory ] && [ "$cov" = confirmatory ]; then
        echo "NOTE: block '$b' is covered by confirmatory predictions; submitting it with" >&2
        echo "      --status exploratory means its records will not count as confirmatory." >&2
    fi
done

# Every pair must exist and be the one the plan means. Fatal on a real submission.
for b in $BLOCKS; do
    # shellcheck disable=SC2086  # PAIR_ROOT is a deliberate word list
    if ! "${PLAN[@]}" --check-pairs --block "$b" --pair-root $PAIR_ROOT; then
        if [ "$DRY_RUN" = true ]; then
            echo "WARNING: pairs for block '$b' not found under: $PAIR_ROOT (dry run goes on)" >&2
        else
            echo "ERROR: pairs for block '$b' not found under: $PAIR_ROOT" >&2; exit 1
        fi
    fi
done

if [ "$DRY_RUN" = false ]; then
    if [ ! -f "$SIF" ]; then
        echo "ERROR: container not found: $SIF" >&2
        echo "       Build it: bash $SCRIPTS/build_container.sh" >&2
        exit 1
    fi
    mkdir -p "$LOGS_DIR"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

run() {
    # The preview line goes to stderr: every call site captures this function's
    # stdout via $(...) to get sbatch's --parsable job id. %q, so that the line
    # can be pasted back into a shell as it is (the --note has spaces).
    { printf '+'; printf ' %q' "$@"; printf '\n'; } >&2
    if [ "$DRY_RUN" = false ]; then "$@"; fi
}

echo "==========================================="
echo " E2 ignition grid — submission plan"
echo " Repo       : $REPO"
echo " Status     : $STATUS"
echo " Blocks     : $BLOCKS"
echo " Pair roots : $PAIR_ROOT"
echo " Output     : $OUT"
echo " Concurrency: $LIMIT tasks per block"
[ -n "$TIME" ] && echo " Walltime   : $TIME per task"
[ "$DRY_RUN" = true ] && echo " DRY RUN    : nothing is submitted"
echo "==========================================="

TOTAL_TASKS=0
TOTAL_RUNS=0
for b in $BLOCKS; do
    n_tasks="$("${PLAN[@]}" --block "$b" --count)"
    n_runs="$("${PLAN[@]}" --block "$b" --runs)"
    out="$LOGS_DIR/ignition_grid_${b}_${STAMP}_%A_%a.out"
    case "$b" in
        audit_*)
            what="records"
            JOBID=$(run sbatch --parsable -J "gda_aud_${b#audit_}" \
                    --array="0-$((n_tasks - 1))%$LIMIT" ${TIME:+--time="$TIME"} -o "$out" -e "$out" \
                    "$SCRIPTS/34_anchor_audit_job.sh" \
                    --group "${b#audit_}" --pair-root $PAIR_ROOT --out "$OUT" \
                    --status "$STATUS" --note "${NOTE:+$NOTE }E2.0 anchor audit: $b") ;;
        *)
            what="runs"
            JOBID=$(run sbatch --parsable -J "gda_ig_${b}" \
                    --array="0-$((n_tasks - 1))%$LIMIT" ${TIME:+--time="$TIME"} -o "$out" -e "$out" \
                    "$SCRIPTS/32_ignition_grid_job.sh" \
                    --block "$b" --pair-root $PAIR_ROOT --out "$OUT" \
                    --status "$STATUS" --note "${NOTE:+$NOTE }E2 ignition grid: $b") ;;
    esac
    echo "$b: $n_tasks tasks, $n_runs $what, job ${JOBID:-<dry-run>}"
    TOTAL_TASKS=$((TOTAL_TASKS + n_tasks))
    TOTAL_RUNS=$((TOTAL_RUNS + n_runs))
done
echo "-------------------------------------------"
echo "total: $TOTAL_TASKS tasks, $TOTAL_RUNS runs and records ($STATUS)"
