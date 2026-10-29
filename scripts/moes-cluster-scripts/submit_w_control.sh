#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# E3 non-adversarial alignment control — submit one group of the Log's runs.
#
# Experiment Log 3, section E3: does a W_0 fitted by Procrustes to a seed-free
# finder's anchors (arm B) or to the ground-truth pairs in the same pool (arm C,
# oracle), followed by Adv-GD's own refinement and propagation, reach the same
# chi as Adv-GD's adversarial W on the same cached embeddings? Arm A (the
# adv_gd records) and arm D (adv_gd_no_adversary) already exist or come from E1;
# this script submits B and C, and for the optional GAE replication (D4) the
# Adv-GD cells that train the new embeddings.
#
# The status label goes on every record of a call (docs/experiment_logging_
# policy.md), and the Log's "Covered records" item labels different groups
# differently, so ONE CALL = ONE GROUP. The groups, with the status the Log gives
# each and the number of array tasks:
#
#   group        what                                         status        tasks
#   main_ge49    B and C, overlap >= 0.49 (b00 b10 b20 b30,     confirmatory   24
#                pair seeds 0-2)                       P3.1-P3.3
#   main_le36_b  B, overlap <= 0.36 (b40 b50, seeds 0-2)       confirmatory    6
#                                                      P3.5
#   main_le36_c  C, overlap <= 0.36 (no prediction)            exploratory     6
#   d4_a         D4: pair generation + the 16 Adv-GD cells     exploratory    16
#                (align seeds 0-3, b10_s0 and b30_s0) that
#                train GAE seeds 1 and 2 into
#                runs/sweep_nominal_gae1 and _gae2
#   d4_bc        D4: B and C on those 4 new embeddings         confirmatory    8
#                                                      P3.4
#   main_* = 36 tasks (24 without the six E1 pairs: --e1 exclude); d4_* = 24;
#   60 in all. `--plan` prints this table and the commands.
#
# Run from the repo root on the head node:
#
#   bash scripts/moes-cluster-scripts/submit_w_control.sh --plan
#   bash scripts/moes-cluster-scripts/submit_w_control.sh --group main_ge49 \
#        --status confirmatory --dry-run
#
# Flags:
#   --group NAME      One of the groups above. Fills in arms, betas, pair seeds,
#                     pair root and GAE seed. Without it, give --arms/--betas/
#                     --pair-seeds (and --pair-root/--gae-seed) yourself.
#   --status S        exploratory | confirmatory (default exploratory). With
#                     --group, confirmatory is REFUSED for a group the Log does
#                     not mark confirmatory, and exploratory on a covered group
#                     prints a note (the records would then not count as covered).
#   --e1 M            include | exclude | only (default include): the six pairs
#                     E1 produces (b10, b20, b50 at pair seeds 1-2). `exclude`
#                     is the 24 runs that can start before E1 finishes; the
#                     other 12 need E1's embeddings and its adv_gd records
#                     (the driver aborts without them).
#   --after IDS       Comma-separated job ids every array of this call waits on
#                     (afterok). d4_bc needs the d4_a jobs: the d4_a call prints
#                     them on its `jobs:` line.
#   --limit N         Max concurrent array tasks per array (default 12).
#   --arms "B C"  --betas "L"  --pair-seeds "L"  --pair-root DIR  --gae-seed N
#   --note TEXT       Prefix for the records' note.
#   --dry-run         Print the sbatch commands only, submit nothing, create
#                     nothing.
#   --plan            Print the groups, task counts and commands, and exit.
#
# Everything written outside the repo goes under /data/$USER/Graph-De-Anonymization/
# (logs/). Override the root with GDA_DATA_DIR.
#
# Prerequisite: bash scripts/moes-cluster-scripts/build_container.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"

DRY_RUN=false
PLAN=false
GROUP=""
STATUS="exploratory"
LIMIT=12
AFTER=""
E1="include"
ARMS=""
BETAS=""
PAIR_SEEDS=""
PAIR_ROOT=""
GAE_SEED=""
NOTE=""

# The pairs E1 produces, as <round(100*beta)>:<pair seed>. Keep in step with
# E1_PAIRS in src/adv_gd/w_control.py (the python plan is the authority; this
# list only sizes the arrays, and a mismatch makes tasks fail on a bad task id).
E1_PAIRS=" 10:1 10:2 20:1 20:2 50:1 50:2 "
ALL_GROUPS="main_ge49 main_le36_b main_le36_c d4_a d4_bc"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --plan)       PLAN=true; shift ;;
        --group)      GROUP="$2"; shift 2 ;;
        --status)     STATUS="$2"; shift 2 ;;
        --e1)         E1="$2"; shift 2 ;;
        --after)      AFTER="$2"; shift 2 ;;
        --limit)      LIMIT="$2"; shift 2 ;;
        --arms)       ARMS="$2"; shift 2 ;;
        --betas)      BETAS="$2"; shift 2 ;;
        --pair-seeds) PAIR_SEEDS="$2"; shift 2 ;;
        --pair-root)  PAIR_ROOT="$2"; shift 2 ;;
        --gae-seed)   GAE_SEED="$2"; shift 2 ;;
        --note)       NOTE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: unrecognized argument: $1" >&2; exit 1 ;;
    esac
done

case "$STATUS" in exploratory|confirmatory) ;;
    *) echo "ERROR: --status must be exploratory or confirmatory" >&2; exit 1 ;;
esac
case "$E1" in include|exclude|only) ;;
    *) echo "ERROR: --e1 must be include, exclude or only" >&2; exit 1 ;;
esac
if [ -n "$GAE_SEED" ] && ! [[ "$GAE_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --gae-seed must be a single non-negative integer" >&2; exit 1
fi

# group_spec NAME: sets G_STAGE (bc | a), G_ARMS, G_BETAS, G_SEEDS, G_SPEC (the
# status the Log gives the group) and G_GAES (GAE seeds; seed 0 = the original
# embeddings in runs/sweep_nominal, seeds 1-2 = the D4 roots).
group_spec() {
    case "$1" in
        main_ge49)   G_STAGE=bc; G_ARMS="B C"; G_BETAS="0.0 0.1 0.2 0.3"; G_SEEDS="0 1 2"; G_SPEC=confirmatory; G_GAES="0" ;;
        main_le36_b) G_STAGE=bc; G_ARMS="B";   G_BETAS="0.4 0.5";         G_SEEDS="0 1 2"; G_SPEC=confirmatory; G_GAES="0" ;;
        main_le36_c) G_STAGE=bc; G_ARMS="C";   G_BETAS="0.4 0.5";         G_SEEDS="0 1 2"; G_SPEC=exploratory;  G_GAES="0" ;;
        d4_a)        G_STAGE=a;  G_ARMS="";    G_BETAS="0.1 0.3";         G_SEEDS="0";     G_SPEC=exploratory;  G_GAES="1 2" ;;
        d4_bc)       G_STAGE=bc; G_ARMS="B C"; G_BETAS="0.1 0.3";         G_SEEDS="0";     G_SPEC=confirmatory; G_GAES="1 2" ;;
        *) return 1 ;;
    esac
}

root_for() {   # root_for GAE_SEED: where that GAE seed's pairs and caches live
    if [ -n "$PAIR_ROOT" ]; then echo "$PAIR_ROOT"
    elif [ "$1" = 0 ]; then echo "runs/sweep_nominal"
    else echo "runs/sweep_nominal_gae$1"; fi
}

# count_cells ARMS BETAS SEEDS E1: the python plan's size, without importing it
# (this script must run where there is no torch, e.g. for --dry-run).
count_cells() {
    local arms="$1" betas="$2" seeds="$3" e1="$4" n=0 b s pct isE1 na
    na=$(wc -w <<<"$arms")
    for b in $betas; do
        pct=$(awk -v b="$b" 'BEGIN{printf "%d", b * 100 + 0.5}')
        for s in $seeds; do
            isE1=false
            case "$E1_PAIRS" in *" $pct:$s "*) isE1=true ;; esac
            if [ "$e1" = exclude ] && [ "$isE1" = true ]; then continue; fi
            if [ "$e1" = only ] && [ "$isE1" = false ]; then continue; fi
            n=$((n + na))
        done
    done
    echo "$n"
}

# a_cells BETAS SEEDS: the Adv-GD cells of one GAE seed (betas x pair seeds x 4 align seeds).
a_cells() { echo $(( $(wc -w <<<"$1") * $(wc -w <<<"$2") * 4 )); }

if [ "$PLAN" = true ]; then
    printf '%-12s %-6s %-32s %-13s %s\n' group stage "arms / pairs" "Log status" tasks
    tot_main=0; tot_d4=0; tot_main_x=0
    for g in $ALL_GROUPS; do
        group_spec "$g"
        n=0
        for ge in $G_GAES; do
            if [ "$G_STAGE" = a ]; then n=$((n + $(a_cells "$G_BETAS" "$G_SEEDS")))
            else n=$((n + $(count_cells "$G_ARMS" "$G_BETAS" "$G_SEEDS" include))); fi
        done
        nx=0
        if [ "$G_STAGE" = bc ]; then
            for ge in $G_GAES; do nx=$((nx + $(count_cells "$G_ARMS" "$G_BETAS" "$G_SEEDS" exclude))); done
        fi
        what="${G_ARMS:-adv_gd} b[${G_BETAS// /,}] s[${G_SEEDS// /,}]"
        printf '%-12s %-6s %-32s %-13s %s\n' "$g" "$G_STAGE" "$what" "$G_SPEC" "$n"
        case "$g" in main_*) tot_main=$((tot_main + n)); tot_main_x=$((tot_main_x + nx)) ;; d4_*) tot_d4=$((tot_d4 + n)) ;; esac
    done
    echo
    echo "main_*: $tot_main tasks ($tot_main_x without the six E1 pairs: --e1 exclude); d4_*: $tot_d4; all: $((tot_main + tot_d4))"
    echo
    echo "One call per group; --status is yours to set, the Log's value is in the table:"
    for g in $ALL_GROUPS; do
        group_spec "$g"
        echo "  bash scripts/moes-cluster-scripts/submit_w_control.sh --group $g --status $G_SPEC"
    done
    echo "d4_bc waits on d4_a: pass the job ids from the d4_a call's 'jobs:' line as --after."
    exit 0
fi

# ── resolve the call ────────────────────────────────────────────────────────
if [ -n "$GROUP" ]; then
    group_spec "$GROUP" || { echo "ERROR: --group must be one of: $ALL_GROUPS" >&2; exit 1; }
    if [ "$STATUS" = confirmatory ] && [ "$G_SPEC" != confirmatory ]; then
        echo "ERROR: the Log does not mark group $GROUP confirmatory ('Covered records'); submit it exploratory." >&2
        exit 1
    fi
    if [ "$STATUS" = exploratory ] && [ "$G_SPEC" = confirmatory ]; then
        echo "NOTE: the Log marks group $GROUP confirmatory; with --status exploratory these records will not count as covered." >&2
    fi
    ARMS="${ARMS:-$G_ARMS}"; BETAS="${BETAS:-$G_BETAS}"; PAIR_SEEDS="${PAIR_SEEDS:-$G_SEEDS}"
    STAGE="$G_STAGE"; GAES="${GAE_SEED:-$G_GAES}"
else
    if [ -z "$ARMS" ] || [ -z "$BETAS" ] || [ -z "$PAIR_SEEDS" ]; then
        echo "ERROR: give --group NAME, or --arms, --betas and --pair-seeds (see --help)" >&2; exit 1
    fi
    STAGE=bc; GAES="${GAE_SEED:-0}"
    echo "NOTE: free-form call, status $STATUS goes on every record of it; the Log's groups are --group main_ge49 ... d4_bc." >&2
fi
for a in $ARMS; do case "$a" in B|C) ;; *) echo "ERROR: arm must be B or C (A and D are existing records), got $a" >&2; exit 1 ;; esac; done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [ "$DRY_RUN" = false ]; then
    if [ ! -f "$SIF" ]; then
        echo "ERROR: container not found: $SIF" >&2
        echo "       Build it: bash $SCRIPTS/build_container.sh" >&2
        exit 1
    fi
    mkdir -p "$LOGS_DIR"
fi

run() {
    # The preview goes to stderr: call sites capture this function's stdout via
    # $(...) to get sbatch's --parsable job id. %q, so that an argument with
    # spaces (--note) previews as the one argument it is passed as.
    { printf '+'; printf ' %q' "$@"; echo; } >&2
    if [ "$DRY_RUN" = false ]; then "$@"; fi
}

JOBS=""
TOTAL=0
DEP_AFTER=""
[ -n "$AFTER" ] && DEP_AFTER="${AFTER//,/:}"
tag() { echo "${GROUP:-custom}"; }

for ge in $GAES; do
    root="$(root_for "$ge")"
    if [ "$STAGE" = a ]; then
        # D4, comparators: generate the pair (idempotent), then the four align
        # seeds of each, exactly the sbatch lines submit_adv_gd_suite.sh uses
        # for its pair-generation and full-method stages, with this GAE seed
        # and its own root (the cache is stamped with the seed, so a GAE seed
        # needs its own root or it overwrites the other's).
        n=$(a_cells "$BETAS" "$PAIR_SEEDS")
        gen_out="$LOGS_DIR/w_control_gen_gae${ge}_${STAMP}.out"
        GEN=$(run sbatch --parsable -o "$gen_out" -e "$gen_out" \
              ${DEP_AFTER:+--dependency=afterok:$DEP_AFTER} \
              "$SCRIPTS/00_generate_adv_gd_pairs_job.sh" \
              --betas $BETAS --pair-seeds $PAIR_SEEDS --beta-mode nominal --out "$root")
        [ "$DRY_RUN" = true ] && GEN="<gen_gae${ge}_id>"
        out="$LOGS_DIR/w_control_d4a_gae${ge}_${STAMP}_%A_%a.out"
        JOBID=$(run sbatch --parsable -J gda_adv --array="0-$((n - 1))%$LIMIT" \
                --dependency="afterok:$GEN" -o "$out" -e "$out" \
                "$SCRIPTS/30_adv_gd_job.sh" --betas $BETAS --pair-seeds $PAIR_SEEDS \
                --beta-mode nominal --out "$root" --align-seeds 0 1 2 3 \
                --status "$STATUS" --gae-seed "$ge" \
                --note "${NOTE:+$NOTE | }E3 D4: Adv-GD comparator on GAE seed $ge")
        [ "$DRY_RUN" = true ] && JOBID="<adv_gd_gae${ge}_id>"
        echo "d4_a gae seed $ge ($root): pair generation job $GEN, $n Adv-GD cells, job $JOBID"
        JOBS="$JOBS gen_gae${ge}=$GEN adv_gd_gae${ge}=$JOBID"
        TOTAL=$((TOTAL + n))
    else
        n=$(count_cells "$ARMS" "$BETAS" "$PAIR_SEEDS" "$E1")
        if [ "$n" -eq 0 ]; then echo "$(tag) gae seed $ge ($root): 0 tasks, nothing to submit"; continue; fi
        ref=""
        [ "$ge" != 0 ] && ref="--reference-root runs/sweep_nominal"
        out="$LOGS_DIR/w_control_$(tag)_gae${ge}_${STAMP}_%A_%a.out"
        JOBID=$(run sbatch --parsable -J gda_wc --array="0-$((n - 1))%$LIMIT" \
                ${DEP_AFTER:+--dependency=afterok:$DEP_AFTER} -o "$out" -e "$out" \
                "$SCRIPTS/33_adv_gd_w_control_job.sh" \
                --arms $ARMS --betas $BETAS --pair-seeds $PAIR_SEEDS \
                --beta-mode nominal --pair-root "$root" --gae-seed "$ge" \
                --e1 "$E1" --status "$STATUS" $ref \
                --note "${NOTE:+$NOTE | }E3 W-control group $(tag)")
        [ "$DRY_RUN" = true ] && JOBID="<w_control_gae${ge}_id>"
        echo "$(tag) gae seed $ge ($root): arms $ARMS, $n tasks, status $STATUS, job $JOBID"
        JOBS="$JOBS w_control_gae${ge}=$JOBID"
        TOTAL=$((TOTAL + n))
    fi
done
echo "total: $TOTAL array tasks in this call ($(tag), status $STATUS)"
echo "jobs:$JOBS"
