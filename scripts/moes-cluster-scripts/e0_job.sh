#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# E0 — Report 1 reproduction (Log 3, section "E0"): the 11-task array, and the
# submitter for it, in one file. CPU only.
#
#   task  arm  run                                         PYTHONHASHSEED
#     0   a    integrated: inherited seed_free.py, top 50         0   <- first, alone
#     1   a    integrated                                         1
#     2   a    integrated                                         2
#     3   a    integrated                                         3
#     4   a    integrated                                         4
#     5   a    integrated, repeat of task 0                       0
#     6   a    pure: the PageRank ranking scored alone            0
#     7   a    pure                                               1
#     8   b    centrality_rank_match.py, pagerank, 50 anchors     0
#     9   b    the same                                           1
#    10   b    the same, --eccen-support nonzero (diagnostic)     0
#
# Arm (a) is Report 1's own code, unmodified, behind e0_arm_a_record.py; arm (b)
# is our reimplementation (--n-anch 50) on the pair staged by e0_stage_pair.py.
# One task is one run, with its own PYTHONHASHSEED and its own output directory.
#
# Run from the repo root on the head node, like submit_*.sh:
#
#   bash scripts/moes-cluster-scripts/e0_job.sh --dry-run
#   bash scripts/moes-cluster-scripts/e0_job.sh --status confirmatory           # task 0 only
#   bash scripts/moes-cluster-scripts/e0_job.sh --status confirmatory --tasks 1-10
#
# Task 0 goes first and alone, by default, because Log 3 asks for it: arm (a)'s
# cost is a guess (0.5-2 h per integrated run) and so is its memory. Read its
# `seconds_main` (the record) and MaxRSS (sacct) before submitting the other ten.
#
# Flags (submit mode):
#   --status S        exploratory | confirmatory (default exploratory). Applies
#                     to tasks 0-9. Task 10, the sigma-over-nonzero diagnostic, is
#                     always exploratory: Log 3 does not cover it. Pass
#                     confirmatory only once the approval line is filled in.
#   --tasks LIST      Which tasks: 0, 1-10, 0,2,5-7 or all (default 0).
#   --limit N         Max concurrent tasks (default 11).
#   --time HH:MM:SS   Walltime per task (default 24:00:00, the README's seed-free
#                     default; the pure and arm-(b) tasks need minutes).
#   --pair-dir DIR    Staged arm-(b) pair (default runs/e0_labeled_dev).
#   --out-root DIR    Arm (a) output directories go here (default runs/e0_arm_a).
#   --dry-run         Print the plan and the sbatch command; submit nothing.
#   --task N          Run task N here, in the foreground, instead of submitting
#                     (an interactive node, never the login node). With --dry-run
#                     it prints the commands without running them.
#
# Records go to results/runs/ as for every run; the bytes stay under runs/ and
# /data/$USER/Graph-De-Anonymization/logs/. Tasks 5 and 9 repeat the
# configuration of tasks 0 and 8, so they share a run-id digest; each waits 75 s
# before it starts so that the two records get different minutes in their
# run id (otherwise provenance.py writes the second as record.<epoch>.json,
# which collect.py does not read).
#
# Prerequisite: bash scripts/moes-cluster-scripts/build_container.sh
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -J gda_e0

set -euo pipefail
# Under sbatch this file is copied into /var/spool/slurmd/jobNNN/, so dirname
# "$0" is the spool dir; the repo is SLURM_SUBMIT_DIR (the submit step below
# passes --chdir, so it is). See 20_seed_free_job.sh. Everywhere else -- the
# submit step, or a task run by hand -- the repo is where this file is: an
# interactive srun session also has SLURM_JOB_ID, and a SLURM_SUBMIT_DIR that
# need not be the repo, which common.sh would otherwise take for it.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "$SLURM_SUBMIT_DIR/scripts/moes-cluster-scripts/common.sh"
else
    unset SLURM_SUBMIT_DIR
    source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"
fi

N_TASKS=11
STAGGER="${E0_STAGGER:-75}"   # seconds; the variable is for tests

STATUS="exploratory"
TASKS="0"
LIMIT=11
TIME=""
PAIR_DIR="runs/e0_labeled_dev"
OUT_ROOT="runs/e0_arm_a"
DRY_RUN=false
TASK_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --status)    STATUS="$2"; shift 2 ;;
        --tasks)     TASKS="$2"; shift 2 ;;
        --limit)     LIMIT="$2"; shift 2 ;;
        --time)      TIME="$2"; shift 2 ;;
        --pair-dir)  PAIR_DIR="$2"; shift 2 ;;
        --out-root)  OUT_ROOT="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        --task)      TASK_ARG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^#SBATCH/p' "$0" | grep '^#' | grep -v '^#SBATCH' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: unrecognized argument: $1" >&2; exit 1 ;;
    esac
done

case "$STATUS" in exploratory|confirmatory) ;;
    *) echo "ERROR: --status must be exploratory or confirmatory" >&2; exit 1 ;;
esac
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --limit must be a positive integer" >&2; exit 1; }

# The container binds $REPO and $DATA_DIR only (common.sh), so anything the
# tasks read or write must be under one of them.
absolute() { case "$1" in /*) echo "$1" ;; *) echo "$REPO/$1" ;; esac; }
PAIR_DIR="$(absolute "$PAIR_DIR")"
OUT_ROOT="$(absolute "$OUT_ROOT")"
for d in "$PAIR_DIR" "$OUT_ROOT"; do
    case "$d" in "$REPO"/*|"$DATA_DIR"/*) ;;
        *) echo "ERROR: $d is outside $REPO and $DATA_DIR, the only paths the container binds" >&2
           exit 1 ;;
    esac
done

# task_spec IDX -> T_ARM, T_MODE, T_HASH, T_TAG, T_SAME_AS (the earlier task whose
# configuration, hence run-id digest, this one repeats), T_FIXED_STATUS
task_spec() {
    T_SAME_AS=""; T_FIXED_STATUS=""
    case "$1" in
        0)  T_ARM=a; T_MODE=integrated; T_HASH=0; T_TAG=a_integrated_h0 ;;
        1)  T_ARM=a; T_MODE=integrated; T_HASH=1; T_TAG=a_integrated_h1 ;;
        2)  T_ARM=a; T_MODE=integrated; T_HASH=2; T_TAG=a_integrated_h2 ;;
        3)  T_ARM=a; T_MODE=integrated; T_HASH=3; T_TAG=a_integrated_h3 ;;
        4)  T_ARM=a; T_MODE=integrated; T_HASH=4; T_TAG=a_integrated_h4 ;;
        5)  T_ARM=a; T_MODE=integrated; T_HASH=0; T_TAG=a_integrated_h0_repeat; T_SAME_AS=0 ;;
        6)  T_ARM=a; T_MODE=pure;       T_HASH=0; T_TAG=a_pure_h0 ;;
        7)  T_ARM=a; T_MODE=pure;       T_HASH=1; T_TAG=a_pure_h1 ;;
        8)  T_ARM=b; T_MODE=all;        T_HASH=0; T_TAG=b_eccen_all_h0 ;;
        9)  T_ARM=b; T_MODE=all;        T_HASH=1; T_TAG=b_eccen_all_h1; T_SAME_AS=8 ;;
        10) T_ARM=b; T_MODE=nonzero;    T_HASH=0; T_TAG=b_eccen_nonzero_h0; T_FIXED_STATUS=exploratory ;;
        *)  echo "ERROR: task id must be 0-10, got '$1'" >&2; return 1 ;;
    esac
}

# expand_tasks SPEC -> the task ids, one per line. SPEC: all, 3, 1-10, 0,2,5-7.
expand_tasks() {
    local spec="$1" part a b i
    local -a parts
    [ "$spec" = all ] && spec="0-$((N_TASKS - 1))"
    [[ "$spec" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || {
        echo "ERROR: --tasks must look like 0, 1-10, 0,2,5-7 or all; got '$1'" >&2; return 1; }
    IFS=',' read -ra parts <<<"$spec"
    for part in "${parts[@]}"; do
        a="${part%-*}"; b="${part#*-}"
        (( 10#$a <= 10#$b && 10#$b < N_TASKS )) || {
            echo "ERROR: '$part' is not a range inside 0-$((N_TASKS - 1))" >&2; return 1; }
        for ((i = 10#$a; i <= 10#$b; i++)); do echo "$i"; done
    done
}

describe_task() {   # describe_task IDX -> one table line
    task_spec "$1"
    printf '  %2s  arm %s  %-11s hash %s  %-12s  %s\n' "$1" "$T_ARM" "$T_MODE" "$T_HASH" \
        "${T_FIXED_STATUS:-$STATUS}" \
        "$([ "$T_ARM" = a ] && echo "$OUT_ROOT/<job>_$T_TAG" || echo "$PAIR_DIR")"
}

# ── Task mode: run one task ──────────────────────────────────────────────────
run_task() {
    local id="$1" status h note out ec
    task_spec "$id"
    status="${T_FIXED_STATUS:-$STATUS}"
    h="$T_HASH"
    banner "E0 — task $id (arm $T_ARM, $T_MODE, hash seed $h, $status)"
    echo " Pair dir : $PAIR_DIR"
    echo " Out root : $OUT_ROOT"
    echo "-------------------------------------------"

    run() {   # echo (shell-quoted), then run unless --dry-run
        printf '+'; printf ' %q' "$@"; printf '\n'
        if [ "$DRY_RUN" = false ]; then "$@"; fi
    }

    if [ -n "$T_SAME_AS" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo " Task $id repeats task $T_SAME_AS's configuration, hence its run-id digest;"
        echo " waiting ${STAGGER}s so the two records start in different UTC minutes."
        sleep "$STAGGER"
    fi

    # The inherited baseline is hash-order sensitive and the hash seed is this
    # experiment's replicate: set, not inherited. in_container passes it on.
    export PYTHONHASHSEED="$h"
    echo "+ export PYTHONHASHSEED=$h"

    if [ "$T_ARM" = a ]; then
        # One directory per run. The job id keeps a resubmitted task from
        # finding the previous attempt's fixed-name files (the recorder refuses
        # a non-empty --out).
        out="$OUT_ROOT/${SLURM_ARRAY_JOB_ID:-manual$(date -u +%Y%m%dT%H%M%S)}_$T_TAG"
        run in_container "$SCRIPTS/e0_arm_a_record.py" \
            --mode "$T_MODE" --out "$out" --status "$status" \
            --run-index "$id" --pair-label "$(basename "$PAIR_DIR")"
    else
        ec=$([ "$T_MODE" = nonzero ] && echo nonzero || echo all)
        note="E0 arm (b), run $id: centrality_rank_match.py pagerank, n_anch 50, phases 123, theta 0.5, eccen_support $ec, PYTHONHASHSEED=$h"
        [ "$T_MODE" = nonzero ] && note="$note; diagnostic (sigma over the nonzero scores, Report 1's support), not covered by the approved predictions"
        note="$note (Log 3, E0${SLURM_ARRAY_JOB_ID:+; SLURM job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}})"
        # Idempotent, and safe against the other arm-(b) tasks doing it too.
        run in_container "$SCRIPTS/e0_stage_pair.py" --pair-dir "$PAIR_DIR"
        run in_container "$REPO/src/baselines/centrality_rank_match.py" \
            --pair "$PAIR_DIR" --dataset labeled_dev --n-anch 50 \
            --measures pagerank --phases 123 --theta 0.5 --eccen-support "$ec" \
            --status "$status" --note "$note"
    fi

    echo "==========================================="
    echo " Task $id done$([ "$DRY_RUN" = true ] && echo ' (dry run: nothing was executed)'): $(date)"
    echo "==========================================="
}

if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    run_task "$SLURM_ARRAY_TASK_ID"
    exit 0
fi
if [ -n "$TASK_ARG" ]; then
    run_task "$TASK_ARG"
    exit 0
fi

# ── Submit mode ──────────────────────────────────────────────────────────────
mapfile -t IDS < <(expand_tasks "$TASKS")
[ "${#IDS[@]}" -gt 0 ] || exit 1
ARRAY="$TASKS"; [ "$ARRAY" = all ] && ARRAY="0-$((N_TASKS - 1))"

if [ "$DRY_RUN" = false ] && [ ! -f "$SIF" ]; then
    echo "ERROR: container not found: $SIF" >&2
    echo "       Build it: bash $SCRIPTS/build_container.sh" >&2
    exit 1
fi

echo "==========================================="
echo " E0 — submission plan"
echo " Repo     : $REPO"
echo " Status   : $STATUS  (task 10 is always exploratory)"
echo " Tasks    : $ARRAY  (${#IDS[@]} of $N_TASKS)"
echo " Pair dir : $PAIR_DIR"
echo " Out root : $OUT_ROOT"
echo "-------------------------------------------"
for id in "${IDS[@]}"; do describe_task "$id"; done
if [ "${#IDS[@]}" -gt 1 ] && [[ " ${IDS[*]} " == *" 0 "* ]]; then
    echo " NOTE: Log 3 asks that task 0 go first and alone (cost and memory are"
    echo "       guesses until it has run); submit with --tasks 0, then --tasks 1-10."
fi
echo "==========================================="

# The %N throttle only when it can bind: "0%11" is a legal array spec but not
# one this repo has used, and a throttle above the task count does nothing.
[ "${#IDS[@]}" -gt "$LIMIT" ] && ARRAY_SPEC="$ARRAY%$LIMIT" || ARRAY_SPEC="$ARRAY"
CMD=(sbatch --parsable -p pleiades -J gda_e0 "--chdir=$REPO"
     "--array=$ARRAY_SPEC"
     ${TIME:+"--time=$TIME"}
     "--output=$LOGS_DIR/gda_e0_%A_%a.out"
     "--error=$LOGS_DIR/gda_e0_%A_%a.err"
     "$SCRIPTS/e0_job.sh" --status "$STATUS" --pair-dir "$PAIR_DIR" --out-root "$OUT_ROOT")

if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] ${CMD[*]}"
    exit 0
fi

mkdir -p "$LOGS_DIR"
JOBID="$("${CMD[@]}")"
echo "  submitted: job $JOBID  (tasks $ARRAY)"
echo
echo " Monitor:  squeue -u $USER"
echo "           tail -f $LOGS_DIR/gda_e0_${JOBID}_${IDS[0]}.out"
if [ "$ARRAY" = 0 ]; then
    echo " Next:     when task 0 is done, read its seconds_main in the record and"
    echo "           sacct -j ${JOBID}_0 --format=JobID,Elapsed,MaxRSS,State"
    echo "           then: bash $SCRIPTS/e0_job.sh --status $STATUS --tasks 1-10"
fi
echo " Records:  results/runs/*_$(basename "$PAIR_DIR")_*/record*.json  (tracked in git; bytes are not)"
