"""E3, the non-adversarial alignment control: Adv-GD's own refinement and
propagation, started from a W_0 that did not come from adversarial training.

Spec: Experiment Log 3, section E3 (overleaf_writeups/init_experiments/
experiment_log_03_mechanism_and_generalization.tex). Per cached embedding
(identified by hash) the log defines four arms:

    A  adversarial W: the existing `adv_gd` records, align seeds 0-3.
       Produced by sweep.py, not by this module.
    B  W_0 = Procrustes on every anchor `composite_centrality` returns (mutual
       CSLS nearest neighbours of its 6-feature fingerprint among the top-N_anch
       degree nodes). No ground truth is used.
    C  oracle: W_0 = Procrustes on the ground-truth pairs inside that same
       top-N_anch pool of each graph (an upper bound for an anchor-supervised W).
    D  identity: the existing `adv_gd_no_adversary` records.

B and C replace the adversarial step by W_0 and keep every later step of A:
`alignment.align_from_seeds` runs the same CSLS / mutual-anchor / Procrustes
refinement rounds and the same top-10 candidate sets as `align`, and `propagate`
and the scoring are Adv-GD's own. Nothing else differs. This module produces
arms B and C only; A and D are records that already exist or that
`sweep.py` writes (for the D4 embeddings, see `submit_w_control.sh`).

One array task is one (arm, pair) cell and writes one record of kind
`adv_gd_w_control`. The record carries the arm, the W_0 source, the seed
anchors' count and precision before refinement, the final anchors, chi, gamma,
the z1/z2 hashes and the hash check. B and C are deterministic, so `seed` in the
record is a placeholder (0) and there is no align seed, restart or selection.

Embeddings are read, never trained. The cached Z1.npz / Z2.npz is the artefact
(Log 3, Shared): it is loaded through adv_gd._load_cache, which refuses a cache
built for another graph, epoch count, dimension or GAE seed, and a missing cache
aborts the run. The z1/z2 hashes are those of the arrays, as in the `adv_gd`
records (adv_gd.z_sha256).

Hash rule (Log 3, E3 Design). Before a pair is used, the pair's `adv_gd`
records (same pair content, same pair root, same GAE seed and epochs, status_exit
ok) must carry one (z1, z2) pair across align seeds 0-3, and the cache on disk must
hash to it; a mismatch aborts the run. If those records disagree with each other,
the run uses the hash pair of the cache on disk at launch, records that pair, flags
the disagreement, and lists as arm A only the records carrying it. Fewer than the
four align-seed records is an abort too, unless --allow-incomplete-records is
given, because the check cannot be made.

Library use is through `run`; the CLI is

    python src/adv_gd/w_control.py --plan   [plan flags]
    python src/adv_gd/w_control.py --check-hash [plan flags | --pair DIR]
    PYTHONHASHSEED=0 python src/adv_gd/w_control.py --task-id N [plan flags] --status S
    PYTHONHASHSEED=0 python src/adv_gd/w_control.py --pair DIR --arm B
    python src/adv_gd/w_control.py --self-test

The plan is beta x pair seed x arm, rebuilt identically by every array task from
the same flags (sweep.py's design). Statuses follow docs/experiment_logging_
policy.md: exploratory unless --status confirmatory is passed, and the label goes
on every record of a call, so the Log's covered groups are separate calls (see
submit_w_control.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from adv_gd import GAE_EPOCHS, DATASETS, _load_cache, load_pair, z_sha256  # noqa: E402
from alignment import (DEFAULTS, align_from_seeds, evaluate_anchors,  # noqa: E402
                       top_degree)
from embedding import reconstruction_auc  # noqa: E402
from propagation import evaluate, propagate  # noqa: E402
from provenance import ARCHIVE, RunRecord, capture_stderr, sha256_file  # noqa: E402
from sweep import PARENTS, divergence_beta, pair_name  # noqa: E402

KIND = "adv_gd_w_control"
ARMS = {"B": "composite_centrality_anchors", "C": "ground_truth_pool_pairs"}
IGNITION_CHI = 0.10                     # Log 3, Shared: a run ignites if chi >= 0.10
ALIGN_SEEDS = (0, 1, 2, 3)
PAIR_FILES = ("G1.edgelist", "G2.edgelist", "mapping.txt")
# The nominal pairs E1 produces (b10, b20, b50 at pair seeds 1 and 2), as
# (round(100 * beta), pair seed). Static, so that a plan never depends on what
# is on disk when a task happens to start.
E1_PAIRS = frozenset({(10, 1), (10, 2), (20, 1), (20, 2), (50, 1), (50, 2)})


class ControlError(RuntimeError):
    """A run that must not continue. `exit_code` is what the process returns."""
    exit_code = 1


class EmbeddingCacheError(ControlError):
    exit_code = 5


class HashMismatch(ControlError):
    exit_code = 3


class RecordsIncomplete(ControlError):
    exit_code = 4


class PairMismatch(ControlError):
    exit_code = 6


# --------------------------------------------------------------------------------------
# The cached embeddings, and the hash rule
# --------------------------------------------------------------------------------------

def load_cached_embeddings(pair: Path, Ga, Gu, dim: int, gae_seed: int,
                           gae_epochs: int, verbose: bool = True) -> tuple:
    """(Za, nodes_a, Zu, nodes_u) from the pair's cache, which is only ever read.

    `adv_gd.embed_cached` would train a GAE when the cache is missing or was
    built as something else. A control run that did so would silently use an
    embedding no other arm saw, so it aborts instead.
    """
    want = {"epochs": gae_epochs, "dim": dim, "seed": gae_seed}
    loaded = []
    for G, name in ((Ga, "Z1.npz"), (Gu, "Z2.npz")):
        hit = _load_cache(G, pair / name, want, verbose=verbose)
        if hit is None:
            raise EmbeddingCacheError(
                f"{pair / name} is missing, or was not built for this graph as "
                f"{want}; this driver never trains a GAE (run the Adv-GD cells "
                f"that cache it first)")
        loaded.append(hit)
    (Za, nodes_a), (Zu, nodes_u) = loaded
    return Za, nodes_a, Zu, nodes_u


def pair_content_hashes(pair: Path) -> dict:
    """SHA-256 of the pair's graph and ground-truth files, as the records hash them."""
    return {f: sha256_file(pair / f) for f in PAIR_FILES}


def _as_list(x) -> list:
    if isinstance(x, str):
        return [x] if x else []
    if isinstance(x, (list, tuple)):
        return [s for s in x if isinstance(s, str) and s]
    return []


def scan_align_records(records_dir: Path, pair: Path, content: dict,
                       gae_seed: int, gae_epochs: int, dim: int) -> tuple:
    """The pair's `adv_gd` (arm A) and `adv_gd_no_adversary` (arm D) records.

    A record belongs to the pair when it names the same pair root and name (the
    last two parts of `pair_dir`: the directory name alone does not identify a
    pair, and the GAE-replication roots reuse the names), hashed the same
    G1/G2/mapping bytes, was a status_exit ok run at this GAE seed, epoch count
    and dimension, and is not named by another record's `supersedes`.
    Returns ({kind: [summary]}, records with this root and name but other pair
    content, record files scanned).
    """
    root, name = pair.parent.name, pair.name
    docs = []
    files = sorted(Path(records_dir).glob("*/record*.json"))
    for f in files:
        try:
            docs.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    superseded = {s for d in docs for s in _as_list(d.get("supersedes"))}
    found = {"adv_gd": [], "adv_gd_no_adversary": []}
    stale = 0
    for d in docs:
        kind = d.get("kind")
        if kind not in found or d.get("run_id") in superseded:
            continue
        pd = Path(str(d.get("pair_dir") or ""))
        if (pd.parent.name, pd.name) != (root, name):
            continue
        ins = {Path(i["path"]).name: i.get("sha256")
               for i in d.get("inputs", []) if "path" in i}
        if any(ins.get(f) != content[f] for f in PAIR_FILES):
            stale += 1
            continue
        m = d.get("metrics") or {}
        if (d.get("status_exit") != "ok" or d.get("gae_seed") != gae_seed
                or d.get("gae_epochs") != gae_epochs
                or (d.get("params") or {}).get("dim") != dim
                or not (m.get("z1_sha256") and m.get("z2_sha256"))):
            continue
        found[kind].append({"run_id": d["run_id"], "align_seed": d.get("seed"),
                            "z1": m["z1_sha256"], "z2": m["z2_sha256"],
                            "status": d.get("status"), "chi": m.get("chi")})
    return found, stale, len(files)


def hash_rule(z1: str, z2: str, found: dict, stale: int = 0,
              expect: tuple = (None, None), allow_incomplete: bool = False
              ) -> tuple:
    """Apply the Log 3 hash rule. Returns (check, problem).

    `check` is the full outcome, for the record. `problem` is the exception to
    raise when the run must abort, else None. The pair used is always the cache
    on disk (z1, z2); `check["recorded"]` is the single pair the four align-seed
    records agree on, or None when they disagree or are missing.
    """
    a, d = found["adv_gd"], found["adv_gd_no_adversary"]
    groups: dict = {}
    for r in a:
        groups.setdefault((r["z1"], r["z2"]), []).append(r)
    seeds = sorted({r["align_seed"] for r in a if r["align_seed"] is not None})
    disk = (z1, z2)
    recorded = next(iter(groups)) if len(groups) == 1 else None

    flags = []
    if not a:
        flags.append("no_align_records")
    elif set(seeds) != set(ALIGN_SEEDS):
        flags.append("align_seeds_incomplete")
    if len(groups) > 1:
        flags.append("align_records_disagree")
    arm_a = [r for r in a if (r["z1"], r["z2"]) == disk]
    if len(groups) > 1 and {r["align_seed"] for r in arm_a} != set(ALIGN_SEEDS):
        flags.append("arm_a_incomplete_for_used_hash")
    arm_d = [r for r in d if (r["z1"], r["z2"]) == disk]
    if len(arm_d) != len(d):
        flags.append("no_adversary_records_with_other_hash")

    check = {
        "rule": "Log 3 E3 hash rule",
        "disk": {"z1": z1, "z2": z2},
        "recorded": ({"z1": recorded[0], "z2": recorded[1]} if recorded else None),
        "used": {"z1": z1, "z2": z2},
        "n_align_records": len(a),
        "align_seeds_found": seeds,
        "distinct_align_hash_pairs": [
            {"z1": k[0], "z2": k[1], "align_seeds": sorted(
                r["align_seed"] for r in v if r["align_seed"] is not None),
             "run_ids": [r["run_id"] for r in v]}
            for k, v in groups.items()],
        "n_no_adversary_records": len(d),
        "arm_a_run_ids": [r["run_id"] for r in arm_a],
        "arm_d_run_ids": [r["run_id"] for r in arm_d],
        "records_of_other_pair_content": stale,
        "expect": {"z1": expect[0], "z2": expect[1]},
        "flags": flags,
    }

    problem = None
    if recorded is not None and recorded != disk:
        problem = HashMismatch(
            f"cache on disk hashes to {z1[:8]}/{z2[:8]} but the pair's align-seed "
            f"records carry {recorded[0][:8]}/{recorded[1][:8]}")
    for want, got, label in ((expect[0], z1, "z1"), (expect[1], z2, "z2")):
        if problem is None and want and not got.startswith(want.lower()):
            problem = HashMismatch(
                f"cache {label} {got[:8]} does not start with the expected {want}")
    if problem is None and not allow_incomplete and (
            "no_align_records" in flags or "align_seeds_incomplete" in flags):
        problem = RecordsIncomplete(
            f"found {len(a)} usable adv_gd record(s) for this pair at align seeds "
            f"{seeds}, need {list(ALIGN_SEEDS)} to apply the hash rule "
            f"({stale} record(s) of this name carry other pair content). Run the "
            f"Adv-GD cells first, or pass --allow-incomplete-records")
    check["verdict"] = ("abort" if problem else
                        "pass_flagged" if flags else "pass")
    return check, problem


# --------------------------------------------------------------------------------------
# Seed sets for W_0
# --------------------------------------------------------------------------------------

def composite_seed_pairs(Ga, Gu, n_anch: int, csls_k: int, verbose: bool = True):
    """Arm B: the anchors composite_centrality returns. No ground truth."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "baselines"))
    import composite_centrality
    return composite_centrality.find_anchors(Ga, Gu, n_anch, csls_k=csls_k,
                                             verbose=verbose)


def oracle_seed_pairs(Ga, Gu, nodes_a: list, nodes_u: list, truth: dict,
                      n_anch: int) -> list:
    """Arm C: the ground-truth pairs with both ends in the top-N_anch degree pool
    of their graph, i.e. every pair a mutual-NN anchor finder over that pool
    could possibly get right."""
    pool_a = [nodes_a[i] for i in top_degree(Ga, nodes_a, n_anch)]
    pool_u = {nodes_u[j] for j in top_degree(Gu, nodes_u, n_anch)}
    return [(a, truth[a]) for a in pool_a if truth.get(a) in pool_u]


# --------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------

def run(pair: Path, dataset: str, arm: str, gae_seed: int = 0,
        gae_epochs: int = GAE_EPOCHS, theta: float = 0.5,
        eccen_support: str = "all", status: str = "exploratory",
        note: str = "", record: bool = True, archive: Path = ARCHIVE,
        records_dir: Path = ARCHIVE, out_dir: Path | None = None,
        reference_root: Path | None = None, expect: tuple = (None, None),
        allow_incomplete: bool = False, device: str | None = None,
        verbose: bool = True) -> dict:
    """Run arm `arm` on one pair and score it; write one record.

    `archive` is where the record goes and `records_dir` where the pair's arm-A
    records are read from (both default to results/runs). `out_dir` takes the
    output mapping and the seed list (default: the pair directory).
    `reference_root`, for the GAE-replication roots, names the root that holds
    the original of this pair: its G1/G2/mapping must hash the same.
    """
    params = DATASETS[dataset]
    Ga, Gu, truth, stats = load_pair(pair)
    cfg = {**DEFAULTS, "skip_adversarial": True, "n_restarts": 1}

    rec = None
    if record:
        rec = RunRecord(KIND, pair.name, status, note, archive=Path(archive))
        rec.set(seed=0, arm=arm, w0_source=ARMS[arm],
                w0_uses_ground_truth=(arm == "C"), deterministic=True,
                gae_seed=gae_seed, beta=stats["beta"],
                beta_nominal=stats.get("beta_nominal"),
                beta_mode=stats.get("beta_mode"), pair_seed=stats.get("seed"),
                dataset=dataset, phases="123", params=params, config=cfg,
                gae_epochs=gae_epochs, theta=theta, eccen_support=eccen_support,
                matchable_nodes=stats.get("matchable_nodes"),
                pair_stats=stats, pair_dir=str(pair))
        if arm == "B":
            rec.set(composite_csls_k=cfg["csls_k"])
        if any(expect):
            rec.set(expect_z1=expect[0], expect_z2=expect[1])
        rec.add_inputs(pair / "G1.edgelist", pair / "G2.edgelist",
                       pair / "mapping.txt", pair / "stats.json",
                       pair / "Z1.npz", pair / "Z2.npz")
        rec.seal_id()

    print(f"pair {pair.name}  beta={stats['beta']}  arm {arm} ({ARMS[arm]})  "
          f"gae seed {gae_seed}  G1 {Ga.number_of_nodes():,}n/"
          f"{Ga.number_of_edges():,}e  G2 {Gu.number_of_nodes():,}n/"
          f"{Gu.number_of_edges():,}e", file=sys.stderr)

    t0 = time.time()
    with capture_stderr() as console:
        try:
            out = _execute(rec, pair, dataset, params, Ga, Gu, truth, stats, arm,
                           gae_seed, gae_epochs, theta, eccen_support, cfg,
                           Path(records_dir), out_dir, reference_root, expect,
                           allow_incomplete, device, verbose,
                           rec.run_id if rec else "unrecorded", t0)
            exit_status, error = "ok", None
        except BaseException as exc:      # a killed run is recorded, as in adv_gd.run
            import traceback
            exit_status, error = "error", traceback.format_exc()
            out = {"failed_in": f"{type(exc).__name__}: {exc}"}
            raise
        finally:
            if rec:
                for key in ("mapping_path", "seed_path"):
                    if key in out:
                        rec.add_outputs(Path(out.pop(key)))
                rec.set(refine_history=out.get("refine_history"),
                        refine_scored=out.get("refine_scored"),
                        config=out.get("config", cfg))
                rec.metric(**{k: out.get(k) for k in (
                    "auc_g1", "auc_g2", "z1_sha256", "z2_sha256",
                    "seed_anchors_found", "seed_anchors_correct",
                    "seed_anchor_precision", "anchors_found", "anchors_correct",
                    "anchor_precision", "mapped", "correct", "chi", "gamma",
                    "ignited")})
                rec.finish(exit_status, error, console.getvalue(), verbose)
                out["run_id"] = rec.run_id

    result_dir = Path(out_dir) if out_dir else pair
    (result_dir / f"{KIND}_result_arm{arm}.json").write_text(
        json.dumps(out, indent=2))
    return out


def _execute(rec, pair, dataset, params, Ga, Gu, truth, stats, arm, gae_seed,
             gae_epochs, theta, eccen_support, cfg, records_dir, out_dir,
             reference_root, expect, allow_incomplete, device, verbose, run_id,
             t0) -> dict:
    """Embeddings, hash rule, seed set, align_from_seeds, propagate, score."""
    print("phase 1: cached embeddings (read only)", file=sys.stderr)
    Za, nodes_a, Zu, nodes_u = load_cached_embeddings(
        pair, Ga, Gu, params["dim"], gae_seed, gae_epochs, verbose)
    z1, z2 = z_sha256(Za), z_sha256(Zu)
    print(f"  z1 {z1[:8]}  z2 {z2[:8]}", file=sys.stderr)
    if rec:
        rec.metric(z1_sha256=z1, z2_sha256=z2)

    content = pair_content_hashes(pair)
    found, stale, n_files = scan_align_records(records_dir, pair, content,
                                               gae_seed, gae_epochs, params["dim"])
    check, problem = hash_rule(z1, z2, found, stale, expect, allow_incomplete)
    check["records_dir"] = str(records_dir)
    check["record_files_scanned"] = n_files
    print(f"  hash rule: {check['verdict']}  ({check['n_align_records']} align-seed "
          f"records, seeds {check['align_seeds_found']}, flags {check['flags']})",
          file=sys.stderr)
    if rec:
        rec.set(hash_check=check)
    if problem:
        raise problem

    if reference_root is not None:
        ref_root = Path(reference_root)
        ref = (ref_root if ref_root.is_absolute() else REPO_ROOT / ref_root) / pair.name
        ref_hashes = pair_content_hashes(ref)
        same = ref_hashes == content
        if rec:
            rec.set(reference_pair={"dir": str(ref), "inputs_match": same})
        print(f"  pair content vs {ref}: {'identical' if same else 'DIFFERENT'}",
              file=sys.stderr)
        if not same:
            raise PairMismatch(
                f"{pair} is not the same pair as {ref} (G1/G2/mapping hashes "
                f"differ), so its embeddings are not replicates of that pair's")

    auc_a = reconstruction_auc(Ga, Za, nodes_a)
    auc_u = reconstruction_auc(Gu, Zu, nodes_u)
    print(f"  reconstruction AUC  G1 {auc_a:.4f}  G2 {auc_u:.4f}", file=sys.stderr)

    print(f"W_0 source: {ARMS[arm]}", file=sys.stderr)
    if arm == "B":
        seeds = composite_seed_pairs(Ga, Gu, params["n_anch"], cfg["csls_k"],
                                     verbose)
    else:
        seeds = oracle_seed_pairs(Ga, Gu, nodes_a, nodes_u, truth,
                                  params["n_anch"])
    seed_score = evaluate_anchors(seeds, truth)
    print(f"  seed pairs {seed_score['correct']}/{seed_score['found']}  "
          f"precision {seed_score['precision']:.4f}", file=sys.stderr)

    # Observation only: how many of each refinement round's anchors are right.
    scored = []

    def on_round(rnd, pairs):
        sc = evaluate_anchors(pairs, truth)
        scored.append({"round": rnd, "anchors": sc["found"],
                       "correct": sc["correct"], "precision": sc["precision"]})

    print("phase 2: refinement from W_0", file=sys.stderr)
    res = align_from_seeds(Za, nodes_a, Zu, nodes_u, Ga, Gu,
                           n_anch=params["n_anch"], m=params["m"],
                           seed_pairs=seeds, cfg=cfg, device=device,
                           verbose=verbose, on_round=on_round)
    score = evaluate_anchors(res["anchors"], truth)

    print("phase 3: propagation", file=sys.stderr)
    mapping = propagate(Ga, Gu, res["anchors"], res["candidates"], theta=theta,
                        support=eccen_support, verbose=verbose)
    final = evaluate(mapping, truth, stats.get("matchable_nodes"))

    result_dir = Path(out_dir) if out_dir else pair
    result_dir.mkdir(parents=True, exist_ok=True)
    mpath = result_dir / f"out_mapping_{KIND}_{run_id}.txt"
    mpath.write_text("".join(f"{u} {v}\n" for u, v in mapping.items()))
    spath = result_dir / f"seed_anchors_{KIND}_{run_id}.txt"
    spath.write_text("".join(f"{a} {u}\n" for a, u in seeds))

    out = {
        "pair": pair.name, "dataset": dataset, "arm": arm,
        "w0_source": ARMS[arm], "beta": stats["beta"],
        "gae_seed": gae_seed, "gae_epochs": gae_epochs,
        "matchable_nodes": stats.get("matchable_nodes"),
        "auc_g1": auc_a, "auc_g2": auc_u, "z1_sha256": z1, "z2_sha256": z2,
        "hash_verdict": check["verdict"], "hash_flags": check["flags"],
        "seed_anchors_found": seed_score["found"],
        "seed_anchors_correct": seed_score["correct"],
        "seed_anchor_precision": seed_score["precision"],
        "anchors_found": score["found"], "anchors_correct": score["correct"],
        "anchor_precision": score["precision"],
        "mapped": final["mapped"], "correct": final["correct"],
        "chi": final["chi"], "gamma": final["gamma"],
        "ignited": bool(final["chi"] >= IGNITION_CHI),
        "theta": theta, "eccen_support": eccen_support,
        "refine_history": res["history"], "refine_scored": scored,
        "config": res["config"], "seconds": round(time.time() - t0, 1),
        "mapping_path": str(mpath), "seed_path": str(spath),
    }
    print("", file=sys.stderr)
    print(f"  seed pairs       {seed_score['correct']}/{seed_score['found']}  "
          f"precision {seed_score['precision']:.4f}", file=sys.stderr)
    print(f"  anchors (final)  {score['correct']}/{score['found']}  "
          f"precision {score['precision']:.4f}", file=sys.stderr)
    print(f"  de-anonymized    {final['correct']:,}/{final['mapped']:,} mapped  "
          f"chi {final['chi']:.4f}  gamma {final['gamma']:.4f}  "
          f"ignited {out['ignited']}", file=sys.stderr)
    print(f"  {time.time() - t0:.0f}s total", file=sys.stderr)
    return out


# --------------------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------------------

def build_plan(args) -> list[dict]:
    """The cells, in a fixed order that --task-id indexes into: beta, pair seed, arm.

    Rebuilt identically by every array task from the same flags, so it must not
    depend on anything on disk: which pairs belong to E1 is a constant.
    """
    stem = PARENTS[args.dataset].stem.replace(".", "_")
    root = Path(args.pair_root)
    root = root if root.is_absolute() else REPO_ROOT / root
    cells = []
    for beta in args.betas:
        gen_beta = divergence_beta(beta) if args.beta_mode == "divergence" else beta
        for pair_seed in args.pair_seeds:
            e1 = (args.beta_mode == "nominal"
                  and (int(round(beta * 100)), pair_seed) in E1_PAIRS)
            if (args.e1 == "exclude" and e1) or (args.e1 == "only" and not e1):
                continue
            path = root / pair_name(stem, gen_beta, pair_seed)
            for arm in args.arms:
                cells.append({
                    "task_id": len(cells), "dataset": args.dataset, "arm": arm,
                    "beta_nominal": beta, "beta_generated": round(gen_beta, 6),
                    "beta_mode": args.beta_mode, "pair_seed": pair_seed,
                    "pair": str(path.relative_to(REPO_ROOT))
                            if path.is_relative_to(REPO_ROOT) else str(path),
                    "e1_pair": e1, "gae_seed": args.gae_seed,
                    "gae_epochs": args.gae_epochs,
                })
    return cells


def _abs(pair: str) -> Path:
    p = Path(pair)
    return p if p.is_absolute() else REPO_ROOT / p


def run_cell(cell: dict, args) -> int:
    pair = _abs(cell["pair"])
    if not (pair / "stats.json").exists():
        print(f"pair {cell['pair']} does not exist", file=sys.stderr)
        return 2
    note = (args.note + " | " if args.note else "") + (
        f"E3 W-control task {cell['task_id']}: arm {cell['arm']} "
        f"({ARMS[cell['arm']]}), nominal beta={cell['beta_nominal']}, pair seed "
        f"{cell['pair_seed']}, gae seed {cell['gae_seed']}")
    return _run_one(pair, cell["dataset"], cell["arm"], cell["gae_seed"],
                    cell["gae_epochs"], note, args)


def _run_one(pair: Path, dataset: str, arm: str, gae_seed: int,
             gae_epochs: int, note: str, args) -> int:
    t0 = time.time()
    try:
        out = run(pair, dataset, arm, gae_seed=gae_seed, gae_epochs=gae_epochs,
                  theta=args.theta, eccen_support=args.eccen_support,
                  status=args.status, note=note, record=not args.no_record,
                  archive=Path(args.archive), records_dir=Path(args.records_dir),
                  out_dir=Path(args.out_dir) if args.out_dir else None,
                  reference_root=args.reference_root,
                  expect=(args.expect_z1, args.expect_z2),
                  allow_incomplete=args.allow_incomplete_records,
                  device=args.device)
    except ControlError as exc:
        print(f"ABORT ({type(exc).__name__}): {exc}", file=sys.stderr)
        return exc.exit_code
    print(f"done in {time.time() - t0:.0f}s  arm {arm}  chi={out.get('chi')}  "
          f"anchors={out.get('anchors_correct')}/{out.get('anchors_found')}  "
          f"seeds={out.get('seed_anchors_correct')}/{out.get('seed_anchors_found')}")
    return 0


def check_hash_pairs(paths: list, args) -> int:
    """The hash rule on each pair, with no compute: the pre-flight for a call."""
    params = DATASETS[args.dataset]
    bad = 0
    widths = (31, 3, 17, 27, 8)

    def row(cells):
        padded = [str(c).ljust(w) for c, w in zip(cells[:-1], widths)]
        return "  ".join(padded + [str(cells[-1])])

    print(row(("pair", "gae", "cache z1/z2", "align-seed records", "seeds",
               "result")), flush=True)
    for pair in paths:
        label = f"{pair.parent.name}/{pair.name}"
        try:
            Ga, Gu, _, _ = load_pair(pair)
            Za, _, Zu, _ = load_cached_embeddings(pair, Ga, Gu, params["dim"],
                                                  args.gae_seed, args.gae_epochs,
                                                  verbose=False)
        except (ControlError, OSError) as exc:
            print(row((label, args.gae_seed, "-", "-", "-",
                       f"NO CACHE ({type(exc).__name__})")), flush=True)
            bad += 1
            continue
        z1, z2 = z_sha256(Za), z_sha256(Zu)
        found, stale, _ = scan_align_records(
            Path(args.records_dir), pair, pair_content_hashes(pair),
            args.gae_seed, args.gae_epochs, params["dim"])
        check, problem = hash_rule(z1, z2, found, stale,
                                   (args.expect_z1, args.expect_z2),
                                   args.allow_incomplete_records)
        rec = check["recorded"]
        n = check["n_align_records"]
        agree = (f"{n}, one pair {rec['z1'][:8]}/{rec['z2'][:8]}" if rec else
                 f"{n}, {len(check['distinct_align_hash_pairs'])} hash pairs")
        verdict = check["verdict"].upper()
        if check["flags"]:
            verdict += " [" + ",".join(check["flags"]) + "]"
        if problem:
            verdict += f"  <- {problem}"
        print(row((label, args.gae_seed, f"{z1[:8]}/{z2[:8]}", agree,
                   ",".join(map(str, check["align_seeds_found"])) or "-",
                   verdict)), flush=True)
        bad += problem is not None
    return 1 if bad else 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="E3 W-control: Adv-GD refinement and propagation from a "
                    "non-adversarial W_0 (arms B and C).")
    g = ap.add_argument_group("plan (rebuilt identically by every array task)")
    g.add_argument("--dataset", default="hepth", choices=sorted(PARENTS))
    g.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=["B", "C"],
                   help="B: composite_centrality anchors. C: ground-truth pairs "
                        "inside the anchor pool (oracle).")
    g.add_argument("--betas", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                   help="Nominal noise levels.")
    g.add_argument("--beta-mode", choices=("nominal", "divergence"),
                   default="nominal")
    g.add_argument("--pair-seeds", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--pair-root", "--out", dest="pair_root", default=None,
                   help="Root of the pair directories (default runs/sweep_<beta-"
                        "mode>). The GAE-replication embeddings live in their "
                        "own roots, e.g. runs/sweep_nominal_gae1.")
    g.add_argument("--gae-seed", type=int, default=0,
                   help="GAE seed the cached embeddings were built with; the "
                        "cache is stamped with it and a mismatch aborts.")
    g.add_argument("--gae-epochs", type=int, default=GAE_EPOCHS)
    g.add_argument("--e1", choices=("include", "exclude", "only"),
                   default="include",
                   help="The six pairs E1 produces (b10, b20, b50 at pair seeds "
                        "1-2): include them, leave them out (the 24 runs that "
                        "can start before E1 ends), or run only them.")
    r = ap.add_argument_group("run")
    r.add_argument("--theta", type=float, default=0.5)
    r.add_argument("--eccen-support", default="all", choices=("all", "nonzero"))
    r.add_argument("--status", choices=("exploratory", "confirmatory"),
                   default="exploratory",
                   help="Declare before the run. Confirmatory only for the "
                        "records the Log's 'Covered records' item names.")
    r.add_argument("--note", default="")
    r.add_argument("--no-record", action="store_true")
    r.add_argument("--archive", default=str(ARCHIVE),
                   help="Where the run record is written (default results/runs).")
    r.add_argument("--records-dir", default=str(ARCHIVE),
                   help="Where the pair's adv_gd records are read from for the "
                        "hash rule (default results/runs).")
    r.add_argument("--out-dir", default=None,
                   help="Where the output mapping and seed list go (default: "
                        "the pair directory).")
    r.add_argument("--reference-root", default=None,
                   help="Root holding the original of each pair; the pair's "
                        "G1/G2/mapping must hash the same (GAE-replication roots).")
    r.add_argument("--expect-z1", default=None,
                   help="Hex prefix (>= 8) the cached z1 must start with.")
    r.add_argument("--expect-z2", default=None)
    r.add_argument("--allow-incomplete-records", action="store_true",
                   help="Proceed when fewer than the four align-seed records "
                        "exist; the check is then flagged, not made.")
    r.add_argument("--device", default=None, help="cuda or cpu (default: auto).")
    r.add_argument("--allow-unset-hashseed", action="store_true")
    m = ap.add_argument_group("modes")
    m.add_argument("--plan", action="store_true", help="Print the cells and exit.")
    m.add_argument("--check-hash", action="store_true",
                   help="Hash rule on the plan's pairs (or --pair), no compute.")
    m.add_argument("--task-id", type=int, default=None,
                   help="Run exactly this cell of the plan.")
    m.add_argument("--pair", default=None,
                   help="Run (or --check-hash) this one pair directory.")
    m.add_argument("--arm", choices=sorted(ARMS), default=None,
                   help="With --pair: which arm to run.")
    m.add_argument("--self-test", action="store_true")
    return ap


def main(argv=None) -> int:
    ap = make_parser()
    args = ap.parse_args(argv)
    if args.pair_root is None:
        args.pair_root = f"runs/sweep_{args.beta_mode}"
    if args.self_test:
        return _self_test()

    if args.pair and (args.check_hash or args.arm):
        paths = [_abs(args.pair)]
    else:
        paths = None
    if args.check_hash:
        if paths is None:
            cells = build_plan(args)
            paths = [_abs(p) for p in dict.fromkeys(c["pair"] for c in cells)]
        return check_hash_pairs(paths, args)

    if args.pair and not args.arm and not args.plan:
        ap.error("--pair needs --arm (or --check-hash)")
    cells = build_plan(args) if (args.plan or not args.pair) else None

    if args.plan or (args.task_id is None and not args.pair):
        print(f"{len(cells)} cells  arms {' '.join(args.arms)}  e1={args.e1}  "
              f"root={args.pair_root}  gae seed {args.gae_seed}")
        for c in cells:
            print(f"  {c['task_id']:3d}  arm {c['arm']}  beta {c['beta_nominal']:.2f}  "
                  f"{Path(c['pair']).name}  gae {c['gae_seed']}"
                  f"{'  [E1 pair]' if c['e1_pair'] else ''}")
        if not args.plan:
            print("nothing to do: pass --plan, --check-hash, --task-id or --pair",
                  file=sys.stderr)
        return 0

    if not os.environ.get("PYTHONHASHSEED") and not args.allow_unset_hashseed:
        print("PYTHONHASHSEED is not set. Set it (0 is conventional here) or "
              "pass --allow-unset-hashseed.", file=sys.stderr)
        return 2
    if args.pair:
        return _run_one(_abs(args.pair), args.dataset, args.arm, args.gae_seed,
                        args.gae_epochs, args.note, args)
    if not 0 <= args.task_id < len(cells):
        print(f"--task-id must be in [0, {len(cells)}); the plan has "
              f"{len(cells)} cells", file=sys.stderr)
        return 2
    return run_cell(cells[args.task_id], args)


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------

def _self_test() -> int:
    """Plan algebra, the hash rule, the record filter and the oracle seed set.
    Needs no dataset. (align_from_seeds has its own check in alignment.py.)"""
    import tempfile
    ap = make_parser()

    def count(argv):
        a = ap.parse_args(argv)
        a.pair_root = a.pair_root or "runs/sweep_nominal"
        return len(build_plan(a))

    # Plan algebra: the Log's counts. B and C on the 18 nominal pairs = 36, split
    # as the 'Covered records' item splits them; 24 of them need no E1 output.
    hi = ["--arms", "B", "C", "--betas", "0.0", "0.1", "0.2", "0.3"]
    lo_b = ["--arms", "B", "--betas", "0.4", "0.5"]
    lo_c = ["--arms", "C", "--betas", "0.4", "0.5"]
    n_hi, n_b, n_c = count(hi), count(lo_b), count(lo_c)
    assert (n_hi, n_b, n_c) == (24, 6, 6), (n_hi, n_b, n_c)
    ex = [count(g + ["--e1", "exclude"]) for g in (hi, lo_b, lo_c)]
    on = [count(g + ["--e1", "only"]) for g in (hi, lo_b, lo_c)]
    assert ex == [16, 4, 4] and on == [8, 2, 2], (ex, on)
    d4 = count(["--arms", "B", "C", "--betas", "0.1", "0.3", "--pair-seeds", "0",
                "--pair-root", "runs/sweep_nominal_gae1", "--gae-seed", "1"])
    assert d4 == 4, d4
    cells = build_plan(ap.parse_args(hi + ["--pair-root", "runs/sweep_nominal"]))
    assert [c["task_id"] for c in cells] == list(range(24))
    assert cells[0]["pair"].endswith("runs/sweep_nominal/cit-HepTh_b00_s0")
    assert {c["pair"].split("/")[-1] for c in cells if c["e1_pair"]} == {
        f"cit-HepTh_b{b}_s{s}" for b in ("10", "20") for s in (1, 2)}
    print(f"  plan: B,C >=0.49 = {n_hi}, B <=0.36 = {n_b}, C <=0.36 = {n_c} "
          f"(36); without E1 pairs {sum(ex)}; D4 B,C = {d4} per GAE root")

    # Hash rule.
    def rec(rid, seed, z1, z2, kind="adv_gd"):
        return {"run_id": rid, "align_seed": seed, "z1": z1, "z2": z2,
                "status": "exploratory", "chi": 0.9}

    def fnd(spec, d=()):
        return {"adv_gd": [rec(f"r{i}", s, a, b) for i, (s, a, b) in enumerate(spec)],
                "adv_gd_no_adversary": [rec(f"d{i}", 0, a, b) for i, (a, b) in enumerate(d)]}

    X, Y = "x" * 64, "y" * 64
    P, Q = "p" * 64, "q" * 64
    good = fnd([(s, X, Y) for s in range(4)], [(X, Y)])
    check, problem = hash_rule(X, Y, good)
    assert problem is None and check["verdict"] == "pass" and not check["flags"]
    assert len(check["arm_a_run_ids"]) == 4 and len(check["arm_d_run_ids"]) == 1
    check, problem = hash_rule(P, Y, good)
    assert isinstance(problem, HashMismatch) and check["verdict"] == "abort"
    dis = fnd([(0, X, Y), (1, X, Y), (2, P, Q), (3, P, Q)])
    check, problem = hash_rule(P, Q, dis)
    assert problem is None and check["verdict"] == "pass_flagged"
    assert "align_records_disagree" in check["flags"]
    assert "arm_a_incomplete_for_used_hash" in check["flags"]
    assert check["used"] == {"z1": P, "z2": Q} and len(check["arm_a_run_ids"]) == 2
    _, problem = hash_rule(X, Y, fnd([(s, X, Y) for s in (0, 1, 2)]))
    assert isinstance(problem, RecordsIncomplete)
    check, problem = hash_rule(X, Y, fnd([(s, X, Y) for s in (0, 1, 2)]),
                               allow_incomplete=True)
    assert problem is None and "align_seeds_incomplete" in check["flags"]
    _, problem = hash_rule(X, Y, good, expect=("xxxxxxxx", "zzzzzzzz"))
    assert isinstance(problem, HashMismatch)
    _, problem = hash_rule(X, Y, good, expect=("xxxxxxxx", "yyyyyyyy"))
    assert problem is None
    print("  hash rule: agree/pass, mismatch/abort, disagree/flag+disk, "
          "incomplete/abort or flag, --expect-z*")

    # Record filter: root, pair content, GAE seed, status, supersedes.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pair = td / "runs" / "sweep_nominal" / "cit-HepTh_b10_s0"
        pair.mkdir(parents=True)
        for f in PAIR_FILES:
            (pair / f).write_text(f + "\n")
        content = pair_content_hashes(pair)

        def write(rid, **kw):
            d = {"kind": "adv_gd", "run_id": rid, "status": "exploratory",
                 "status_exit": "ok", "seed": 0, "gae_seed": 0, "gae_epochs": 600,
                 "params": {"dim": 32}, "pair_dir": str(pair),
                 "inputs": [{"path": f"x/{f}", "sha256": content[f]}
                            for f in PAIR_FILES],
                 "metrics": {"z1_sha256": X, "z2_sha256": Y}, "supersedes": None}
            d.update(kw)
            (td / "arch" / rid).mkdir(parents=True, exist_ok=True)
            (td / "arch" / rid / "record.json").write_text(json.dumps(d))

        write("ok0")
        write("ok1", seed=1)
        write("badroot", pair_dir=str(td / "runs" / "sweep_nominal_gae1" / pair.name))
        write("othercontent", inputs=[{"path": "G1.edgelist", "sha256": "0" * 64}] +
              [{"path": f"x/{f}", "sha256": content[f]} for f in PAIR_FILES[1:]])
        write("gae1", gae_seed=1)
        write("failed", status_exit="error")
        write("old", seed=2)
        write("new", seed=2, supersedes="old")
        write("noadv", kind="adv_gd_no_adversary")
        found, stale, n = scan_align_records(td / "arch", pair, content, 0, 600, 32)
        got = sorted(r["run_id"] for r in found["adv_gd"])
        assert got == ["new", "ok0", "ok1"], got
        assert [r["run_id"] for r in found["adv_gd_no_adversary"]] == ["noadv"]
        assert stale == 1 and n == 9, (stale, n)
    print("  record filter: pair root, pair content, GAE seed, status_exit, "
          "supersedes")

    # Oracle seed set: both ends in the degree pool, truth only.
    import networkx as nx
    Ga = nx.star_graph(5)                                  # node 0 has degree 5
    Ga = nx.relabel_nodes(Ga, {i: f"a{i}" for i in range(6)})
    Gu = nx.relabel_nodes(Ga, {f"a{i}": f"u{i}" for i in range(6)})
    na, nu = sorted(Ga.nodes()), sorted(Gu.nodes())
    truth = {f"a{i}": f"u{i}" for i in range(6)}
    seeds = oracle_seed_pairs(Ga, Gu, na, nu, truth, n_anch=2)
    assert seeds == [("a0", "u0"), ("a1", "u1")], seeds     # hub, then lowest id
    assert oracle_seed_pairs(Ga, Gu, na, nu, {"a0": "u5", "a1": "u1"}, 2) == [
        ("a1", "u1")]
    print("  oracle seed set: pairs with both ends in the pool")
    print("w_control self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
