"""Index-rank-matching baseline: docs/some-context.txt section 2's own
"Index Matching" method, reproduced measure-for-measure.

The method is about as simple as seed-free de-anonymization gets: pick one
centrality measure, sort each graph's nodes by it (descending, ties broken
by node id for determinism), and pair same-rank nodes across the two sorted
lists -- node 0 in G^a's sorted order with node 0 in G^u's, node 1 with node
1, and so on. No embedding, no similarity search, just two sorted lists
zipped together. The report measured this "pure" (the rank-bijection scored
directly against ground truth) for five measures: PageRank, Eigenvector,
Degree, K-core, K-truss (Node2Vec was measured too but is excluded here --
see `centrality.py`'s module docstring for why), then separately measured
an "integration approach": feed the method's own top-ranked pairs into
Phase 1's seed-based propagation algorithm as anchors, and score the
resulting mapping instead. Both numbers are reproduced here per measure, in
the same two-column shape as the report's own table, so this file's output
can be compared to it directly.

Unlike `structural_fingerprint.py` and `isorank.py`, this is not a
similarity-search baseline -- there is no anchor/candidate search over
nearest neighbours, no cosine, no CSLS. The full-graph rank-bijection IS the
Phase 1/2 output; "anchors" for the integration step are just its top-N_anch
ranked pairs, taken as singleton candidate sets (each top-ranked node's only
candidate is the partner its own rank position already names), which is a
much tighter restriction than the top-10 candidate sets the similarity
baselines use.

CLI:
    python src/baselines/centrality_rank_match.py --pair runs/sweep_nominal/cit-HepTh_b10_s0 --dataset hepth

`--n-anch N` replaces Table 1's anchor count and lets `--dataset` be any label,
for a pair that is not one of the generated HepTh/FB/Gplus ones (E0 runs it on
Report 1's own labeled_dev pair at N=50). Without it nothing changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATASETS, RunRecord, capture_stderr, load_pair,  # noqa: E402
                    propagate, evaluate, evaluate_anchors, REPO_ROOT)
from centrality import MEASURES  # noqa: E402

KIND_PREFIX = "baseline_centrality_"


def rank_mapping(Ga: nx.Graph, Gu: nx.Graph, measure: str) -> dict:
    """Zip the two graphs' nodes, both sorted descending by `measure` (ties
    broken by node id so the order is deterministic), position for position.

    Rank position is the only information used -- the actual measure values
    are discarded once the sort is done, exactly as "index matching" implies.
    """
    score_fn = MEASURES[measure]
    sa = score_fn(Ga)
    su = score_fn(Gu)
    order_a = sorted(Ga.nodes(), key=lambda n: (-sa[n], n))
    order_u = sorted(Gu.nodes(), key=lambda n: (-su[n], n))
    return dict(zip(order_a, order_u))


def run(pair: Path, dataset: str, measures: list[str] | None = None,
       phases: str = "123", theta: float = 0.5, eccen_support: str = "all",
       status: str = "exploratory", note: str = "", record: bool = True,
       verbose: bool = True, n_anch: int | None = None) -> dict:
    """Run every requested measure on one pair, returning one sub-result dict
    per measure (keyed by measure name) rather than a single flat dict --
    this is the one baseline here that sweeps a method-internal axis
    (which centrality measure) in addition to the beta x pair_seed grid
    `sweep_baselines.py` already covers, and multiplying the grid by measure
    would make every other method's run count inconsistent with this one's.

    `n_anch` is the one addition for E0. Left as None, the anchor count is the
    Table 1 row of `dataset`, exactly as before, and the records are exactly as
    before. Given, it replaces that lookup, `dataset` becomes a free label that
    only names the data in the record (so a pair outside DATASETS can be run),
    and each record also carries `anchors_found`, `anchors_correct` and
    `anchor_precision`, scored by the same `evaluate_anchors` as Adv-GD's.
    """
    measures = measures or sorted(MEASURES)
    pinned = n_anch is not None
    if pinned and n_anch < 1:
        raise ValueError(f"n_anch must be a positive integer, got {n_anch!r}")
    if not pinned:
        n_anch = DATASETS[dataset]["n_anch"]
    Ga, Gu, truth, stats = load_pair(pair)

    results = {}
    for measure in measures:
        kind = KIND_PREFIX + measure
        rec = RunRecord(kind, pair.name, status, note) if record else None
        if rec:
            rec.set(seed=0, beta=stats["beta"], dataset=dataset,
                   measure=measure, phases=phases, theta=theta,
                   eccen_support=eccen_support, n_anch=n_anch,
                   matchable_nodes=stats.get("matchable_nodes"),
                   pair_stats=stats, pair_dir=str(pair))
            rec.add_inputs(pair / "G1.edgelist", pair / "G2.edgelist",
                           pair / "mapping.txt", pair / "stats.json")
            rec.seal_id()

        t0 = time.time()
        with capture_stderr() as console:
            try:
                if verbose:
                    print(f"pair {pair.name}  beta={stats['beta']}  "
                          f"measure={measure}", file=sys.stderr)
                mapping_full = rank_mapping(Ga, Gu, measure)
                pure = evaluate(mapping_full, truth, stats.get("matchable_nodes"))

                out = {
                    "pair": pair.name, "dataset": dataset, "measure": measure,
                    "beta": stats["beta"],
                    "matchable_nodes": stats.get("matchable_nodes"),
                    "pure_mapped": pure["mapped"], "pure_correct": pure["correct"],
                    "pure_chi": pure["chi"], "pure_gamma": pure["gamma"],
                    "seconds": round(time.time() - t0, 1),
                }
                if "3" in phases:
                    # Top-ranked pairs, in RANK order -- mapping_full's keys
                    # are already in descending-rank order (rank_mapping zips
                    # the two rank-sorted lists together), and must be read
                    # that way rather than re-sorted by node id: G^a and G^u
                    # can have different node counts (noise adds/drops
                    # nodes), so the zip silently drops G^a's lowest-ranked
                    # nodes from mapping_full -- indexing by id order can
                    # land on exactly one of those dropped nodes and KeyError.
                    anchor_nodes = list(mapping_full.keys())[:min(n_anch, len(mapping_full))]
                    anchors = [(a, mapping_full[a]) for a in anchor_nodes]
                    candidates = {a: [mapping_full[a]] for a in anchor_nodes}
                    mapping = propagate(Ga, Gu, anchors, candidates,
                                        theta=theta, support=eccen_support,
                                        verbose=verbose)
                    final = evaluate(mapping, truth, stats.get("matchable_nodes"))
                    mpath = pair / f"out_mapping_{kind}_{rec.run_id if rec else 'unrecorded'}.txt"
                    mpath.write_text("".join(f"{u} {v}\n" for u, v in mapping.items()))
                    out.update({"mapping_path": str(mpath), "mapped": final["mapped"],
                               "correct": final["correct"], "chi": final["chi"],
                               "gamma": final["gamma"], "theta": theta,
                               "eccen_support": eccen_support})
                    if pinned:
                        scored = evaluate_anchors(anchors, truth)
                        out.update({"anchors_found": scored["found"],
                                   "anchors_correct": scored["correct"],
                                   "anchor_precision": scored["precision"]})
                exit_status, error = "ok", None
            except BaseException as exc:
                import traceback
                exit_status, error = "error", traceback.format_exc()
                out = {"failed_in": f"{type(exc).__name__}: {exc}"}
                raise
            finally:
                if rec:
                    if "3" in phases and "mapping_path" in out:
                        rec.add_outputs(Path(out.pop("mapping_path")))
                    rec.metric(**{k: out.get(k) for k in
                                  ("pure_chi", "pure_gamma", "chi", "gamma",
                                   "anchors_found", "anchors_correct",
                                   "anchor_precision")})
                    rec.finish(exit_status, error, console.getvalue(), verbose)
                    out["run_id"] = rec.run_id

        if verbose:
            print(f"  [{measure}] pure chi {out['pure_chi']:.4f}  "
                  f"gamma {out['pure_gamma']:.4f}", file=sys.stderr)
            if "chi" in out:
                print(f"  [{measure}] integrated chi {out['chi']:.4f}  "
                      f"gamma {out['gamma']:.4f}", file=sys.stderr)
            if "anchors_correct" in out:
                print(f"  [{measure}] anchors {out['anchors_correct']}/"
                      f"{out['anchors_found']} correct", file=sys.stderr)
            print(f"  [{measure}] {time.time() - t0:.0f}s total", file=sys.stderr)
        results[measure] = out

    (pair / "centrality_rank_result.json").write_text(json.dumps(results, indent=2))
    return results


def self_test(verbose: bool = True) -> bool:
    """Synthetic check: on a relabelled copy of the same graph, every measure
    must produce a deterministic, well-formed full bijection (every G^a node
    mapped to exactly one G^u node), and `evaluate` on it must return a chi
    in [0, 1]. Not a check that rank matching recovers the permutation --
    the report's own numbers (0.10-0.83%) establish that it mostly does not,
    which is the whole reason the "integration" half of this file exists.
    """
    import numpy as np
    ok = True
    rng = np.random.default_rng(0)
    G = nx.gnm_random_graph(200, 700, seed=0)
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
    nodes = sorted(G.nodes())
    perm = rng.permutation(nodes)
    relabel = dict(zip(nodes, perm))
    Gp = nx.relabel_nodes(G, relabel)
    truth = {n: relabel[n] for n in nodes}

    for measure in sorted(MEASURES):
        mapping = rank_mapping(G, Gp, measure)
        if set(mapping) != set(nodes) or len(set(mapping.values())) != len(mapping):
            print(f"  [FAIL] {measure}: rank mapping is not a full bijection "
                  f"({len(mapping)} entries, {len(set(mapping.values()))} unique "
                  f"targets, expected {len(nodes)} of each)", file=sys.stderr)
            ok = False
            continue
        score = evaluate(mapping, truth, len(nodes))
        if not (0.0 <= score["chi"] <= 1.0):
            print(f"  [FAIL] {measure}: chi {score['chi']} out of [0,1]",
                  file=sys.stderr)
            ok = False
            continue
        print(f"  [PASS] {measure}: full bijection, pure chi "
              f"{score['chi']:.4f} ({score['correct']}/{len(nodes)})",
              file=sys.stderr)

    # Regression check for the mismatched-node-count KeyError: a real pair
    # has G^a and G^u with DIFFERENT node counts (noise adds/drops nodes), so
    # rank_mapping's zip silently drops G^a's lowest-ranked nodes from
    # mapping_full's keys. Anchor selection must read mapping_full's own
    # keys (rank order) rather than re-deriving anchor nodes from a
    # node-id-sorted list of G^a, or it can pick one of the dropped nodes
    # and KeyError -- exactly what happened on a real beta>0 sweep cell
    # before this was caught.
    Gb = G.copy()
    Gb.remove_nodes_from(list(Gb.nodes())[:5])  # G^a now has 5 more nodes than G^b
    for measure in sorted(MEASURES):
        mapping_full = rank_mapping(G, Gb, measure)
        assert len(mapping_full) < G.number_of_nodes(), (
            "test setup error: mapping_full should be truncated to G^b's "
            "smaller node count, or this check is not exercising the bug")
        n_anch = min(50, len(mapping_full))
        anchor_nodes_by_rank = list(mapping_full.keys())[:n_anch]      # correct
        anchor_nodes_by_id = sorted(G.nodes())[:n_anch]                # the old, buggy approach
        try:
            _ = [(a, mapping_full[a]) for a in anchor_nodes_by_rank]
        except KeyError as exc:
            print(f"  [FAIL] {measure}: rank-order anchor selection "
                  f"KeyError: {exc}", file=sys.stderr)
            ok = False
            continue
        if not any(a not in mapping_full for a in anchor_nodes_by_id):
            print(f"  [WARN] {measure}: this run of the regression check "
                  f"did not actually land on a dropped node by chance -- "
                  f"not a failure, but weaker evidence than usual",
                  file=sys.stderr)
    if ok:
        print(f"  [PASS] anchor selection survives mismatched node counts "
              f"({G.number_of_nodes()} vs {Gb.number_of_nodes()} nodes)",
              file=sys.stderr)

    print("all checks passed" if ok else "CHECKS FAILED", file=sys.stderr)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Index-rank-matching baseline (one or more centrality "
                    "measures, report section 2's own method).")
    ap.add_argument("--pair", type=str,
                    help="Pair directory from the beta noise generator "
                         "(flat layout, not --compat-layout).")
    ap.add_argument("--dataset", type=str,
                    help="Which Table 1 N_anch/M row to use (" +
                         ", ".join(sorted(DATASETS)) + "). With --n-anch it "
                         "may be any label and only names the data.")
    ap.add_argument("--n-anch", type=int, default=None,
                    help="Number of top-ranked pairs handed to propagation. "
                         "Replaces the Table 1 value, and makes --dataset a "
                         "free label (E0: labeled_dev, 50).")
    ap.add_argument("--measures", type=str, nargs="+",
                    choices=sorted(MEASURES), default=sorted(MEASURES),
                    help="Which centrality measures to run (default: all).")
    ap.add_argument("--phases", type=str, default="123", choices=("12", "123"))
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--eccen-support", type=str, default="all",
                    choices=("all", "nonzero"))
    ap.add_argument("--status", choices=("exploratory", "confirmatory"),
                    default="exploratory")
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1
    if not args.pair or not args.dataset:
        ap.error("--pair and --dataset are required (or pass --self-test)")
    if args.n_anch is not None and args.n_anch < 1:
        ap.error("argument --n-anch: must be a positive integer")
    if args.n_anch is None and args.dataset not in DATASETS:
        ap.error(f"argument --dataset: invalid choice: {args.dataset!r} "
                 f"(choose from {', '.join(repr(d) for d in sorted(DATASETS))};"
                 f" any other label needs --n-anch)")

    run(Path(args.pair), args.dataset, measures=args.measures,
        phases=args.phases, theta=args.theta, eccen_support=args.eccen_support,
        status=args.status, note=args.note, record=not args.no_record,
        n_anch=args.n_anch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
