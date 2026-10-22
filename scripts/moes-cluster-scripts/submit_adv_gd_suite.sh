#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Adv-GD comparison suite on cit-HepTh — submit the whole thing.
#
# Six arrays, all depending on one pair-generation job (--dependency=afterok),
# all over the SAME beta x pair_seed grid, into the SAME pair root
# (runs/sweep_divergence/) so every method's numbers come from the identical
# pairs:
#
#   0  generate pairs (adv_gd/sweep.py --generate)                    (1 job)
#   1  full Adv-GD: beta x pair_seed x align_seed                     (array)
#   2  Adv-GD ablation (skip_adversarial): beta x pair_seed           (array,
#      align_seed fixed at 0 -- an identity start is deterministic, so
#      sweeping align_seed over it would just be the same run repeated)
#   3  baseline_structural_fingerprint: beta x pair_seed              (array)
#   4  baseline_isorank: beta x pair_seed                             (array)
#   5  baseline_centrality_rank (docs/some-context.txt's own index-rank-
#      matching method, 5 measures scored per cell): beta x pair_seed (array)
#   6  ours_composite_centrality (this repo's own baseline: 6-feature
#      fingerprint + CSLS matching): beta x pair_seed                 (array)
#
# Stages 1-6 do not depend on each other, only on stage 0, so a failure in one
# does not block or corrupt the others (same reasoning as Report 1's
# seed_based/seed_free arrays -- see submit_report1.sh).
#
# Run from the repo root on the head node:
#
#   bash scripts/moes-cluster-scripts/submit_adv_gd_suite.sh --dry-run   # see the plan
#   bash scripts/moes-cluster-scripts/submit_adv_gd_suite.sh
#
# Submit-specific flags:
#   --dry-run           Print the sbatch commands only, submit nothing.
#   --no-generate       Assume the pairs exist; skip stage 0 and the dependency.
#   --only KIND         One of: pairs | adv_gd | ablation | structural |
#                       isorank | centrality_rank | composite_centrality.
#   --limit N           Max concurrent array tasks per stage (default 12).
#   --betas "L"         Nominal betas, space-separated (default matches
#                        Report 1's own grid: 0.0 0.1 0.2 0.3 0.4 0.5).
#   --pair-seeds "L"    Pair noise draws (default "0").
#   --align-seeds "L"   Adv-GD alignment seeds, full method only (default
#                        matches adv_gd/sweep.py's own default "0 1 2 3").
#   --gae-seed N        Phase 1 (GAE) seed for the adv_gd and ablation stages
#                        only; pairs and baselines train no GAE and never
#                        receive it. Default: not passed, so sweep.py's own
#                        default (0) applies. The embedding cache in each pair
#                        dir (Z1.npz/Z2.npz) is stamped with the seed, so a run
#                        with a different seed retrains and OVERWRITES it --
#                        give each GAE seed its own --pair-root.
#   --beta-mode M       nominal | divergence (default divergence, matching
#                        this script's original behavior). nominal deletes
#                        each copy at the stated beta (edges diverge by
#                        ~2*beta) -- the paper's own definition. The two
#                        modes write to disjoint pair roots
#                        (runs/sweep_<mode>/) so they can never collide or
#                        silently mix; --pair-root overrides that default.
#   --pair-root DIR     Override the pair root (default runs/sweep_<mode>/).
#   --status S          exploratory | confirmatory (default confirmatory,
#                        matching this script's original behavior). Per
#                        docs/experiment_logging_policy.md, confirmatory means
#                        re-running against a stated prior numeric expectation
#                        -- pass --status exploratory for first-look runs
#                        (e.g. an initial pair-seed variance check) that have
#                        no such prior to confirm or refute.
#
# Everything written outside the repo goes under
# /data/$USER/Graph-De-Anonymization/ -- logs under logs/, same as Report 1.
# Override the root with GDA_DATA_DIR.
#
# Prerequisite: bash scripts/moes-cluster-scripts/build_container.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/common.sh"

DRY_RUN=false
GENERATE=true
ONLY=""
LIMIT=12
BETAS="0.0 0.1 0.2 0.3 0.4 0.5"
PAIR_SEEDS="0"
ALIGN_SEEDS="0 1 2 3"
BETA_MODE="divergence"
PAIR_ROOT=""
STATUS="confirmatory"
GAE_SEED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --no-generate)  GENERATE=false; shift ;;
        --only)         ONLY="$2"; shift 2 ;;
        --limit)        LIMIT="$2"; shift 2 ;;
        --betas)        BETAS="$2"; shift 2 ;;
        --pair-seeds)   PAIR_SEEDS="$2"; shift 2 ;;
        --align-seeds)  ALIGN_SEEDS="$2"; shift 2 ;;
        --beta-mode)    BETA_MODE="$2"; shift 2 ;;
        --pair-root)    PAIR_ROOT="$2"; shift 2 ;;
        --status)       STATUS="$2"; shift 2 ;;
        --gae-seed)     GAE_SEED="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: unrecognized argument: $1" >&2; exit 1 ;;
    esac
done

case "$ONLY" in ""|pairs|adv_gd|ablation|structural|isorank|centrality_rank|composite_centrality) ;;
    *) echo "ERROR: --only must be pairs, adv_gd, ablation, structural, isorank, centrality_rank or composite_centrality" >&2
       exit 1 ;;
esac
case "$BETA_MODE" in nominal|divergence) ;;
    *) echo "ERROR: --beta-mode must be nominal or divergence" >&2; exit 1 ;;
esac
case "$STATUS" in exploratory|confirmatory) ;;
    *) echo "ERROR: --status must be exploratory or confirmatory" >&2; exit 1 ;;
esac
if [ -n "$GAE_SEED" ] && ! [[ "$GAE_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --gae-seed must be a single non-negative integer" >&2
    exit 1
fi
[ -z "$PAIR_ROOT" ] && PAIR_ROOT="runs/sweep_${BETA_MODE}"

mkdir -p "$LOGS_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [ ! -f "$SIF" ] && [ "$DRY_RUN" = false ]; then
    echo "ERROR: container not found: $SIF" >&2
    echo "       Build it: bash $SCRIPTS/build_container.sh" >&2
    exit 1
fi

# Count cells the same way the python planners do, without importing them
# into this shell script: betas * pair_seeds for the baselines/ablation axis,
# times align_seeds again for the full method.
n_betas=$(wc -w <<<"$BETAS")
n_seeds=$(wc -w <<<"$PAIR_SEEDS")
n_align=$(wc -w <<<"$ALIGN_SEEDS")
N_SOLO=$((n_betas * n_seeds))
N_FULL=$((n_betas * n_seeds * n_align))

run() {
    # The preview line goes to stderr: every call site captures this
    # function's stdout via $(...) to get sbatch's --parsable job id, and a
    # preview line on stdout would be captured as if it were that id.
    echo "+ $*" >&2
    if [ "$DRY_RUN" = false ]; then "$@"; fi
}

PAIR_DEP=""
if [ "$GENERATE" = true ] && { [ -z "$ONLY" ] || [ "$ONLY" = pairs ]; }; then
    GEN_OUT="$LOGS_DIR/adv_gd_gen_${STAMP}.out"
    JOBID=$(run sbatch --parsable -o "$GEN_OUT" -e "$GEN_OUT" \
            "$SCRIPTS/00_generate_adv_gd_pairs_job.sh" \
            --betas $BETAS --pair-seeds $PAIR_SEEDS --beta-mode "$BETA_MODE" \
            --out "$PAIR_ROOT")
    echo "pair generation: job ${JOBID:-<dry-run>}"
    PAIR_DEP="${JOBID:+--dependency=afterok:$JOBID}"
    [ "$ONLY" = pairs ] && exit 0
fi

submit_array() {
    local name="$1" jname="$2" script="$3" n="$4"; shift 4
    local out="$LOGS_DIR/${name}_${STAMP}_%A_%a.out"
    JOBID=$(run sbatch --parsable -J "$jname" \
            --array="0-$((n - 1))%$LIMIT" ${PAIR_DEP:+$PAIR_DEP} \
            -o "$out" -e "$out" \
            "$script" --betas $BETAS --pair-seeds $PAIR_SEEDS \
            --beta-mode "$BETA_MODE" --out "$PAIR_ROOT" "$@")
    echo "$name: $n cells, job ${JOBID:-<dry-run>}"
}

if [ -z "$ONLY" ] || [ "$ONLY" = adv_gd ]; then
    submit_array adv_gd_full gda_adv "$SCRIPTS/30_adv_gd_job.sh" "$N_FULL" \
        --align-seeds $ALIGN_SEEDS --status "$STATUS" \
        ${GAE_SEED:+--gae-seed "$GAE_SEED"} \
        --note "Adv-GD suite: full method"
fi

if [ -z "$ONLY" ] || [ "$ONLY" = ablation ]; then
    submit_array adv_gd_ablation gda_abl "$SCRIPTS/30_adv_gd_job.sh" "$N_SOLO" \
        --align-seeds 0 --skip-adversarial --status "$STATUS" \
        ${GAE_SEED:+--gae-seed "$GAE_SEED"} \
        --note "Adv-GD suite: skip_adversarial ablation"
fi

if [ -z "$ONLY" ] || [ "$ONLY" = structural ]; then
    submit_array baseline_structural gda_bl_sf "$SCRIPTS/31_baselines_job.sh" "$N_SOLO" \
        --method structural_fingerprint --status "$STATUS" \
        --note "Adv-GD suite: structural-fingerprint baseline"
fi

if [ -z "$ONLY" ] || [ "$ONLY" = isorank ]; then
    submit_array baseline_isorank gda_bl_ir "$SCRIPTS/31_baselines_job.sh" "$N_SOLO" \
        --method isorank --status "$STATUS" \
        --note "Adv-GD suite: IsoRank baseline"
fi

if [ -z "$ONLY" ] || [ "$ONLY" = centrality_rank ]; then
    submit_array baseline_centrality_rank gda_bl_cr "$SCRIPTS/31_baselines_job.sh" "$N_SOLO" \
        --method centrality_rank --status "$STATUS" \
        --note "Adv-GD suite: index-rank-matching baseline (docs/some-context.txt section 2)"
fi

if [ -z "$ONLY" ] || [ "$ONLY" = composite_centrality ]; then
    submit_array ours_composite_centrality gda_bl_cc "$SCRIPTS/31_baselines_job.sh" "$N_SOLO" \
        --method composite_centrality --status "$STATUS" \
        --note "Adv-GD suite: this repo's own 6-feature + CSLS baseline"
fi
