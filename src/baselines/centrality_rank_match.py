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
    python src/baselines/centrality_rank_match.py --pair runs/sweep_divergence/cit-HepTh_b10_s0 --dataset hepth
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
                    propagate, evaluate, REPO_ROOT)
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
       verbose: bool = True) -> dict:
    """Run every requested measure on one pair, returning one sub-result dict
    per measure (keyed by measure name) rather than a single flat dict --
    this is the one baseline here that sweeps a method-internal axis
    (which centrality measure) in addition to the beta x pair_seed grid
    `sweep_baselines.py` already covers, and multiplying the grid by measure
    would make every other method's run count inconsistent with this one's.
    """
    measures = measures or sorted(MEASURES)
    params = DATASETS[dataset]
    Ga, Gu, truth, stats = load_pair(pair)
    nodes_a = sorted(Ga.nodes())

    results = {}
    for measure in measures:
        kind = KIND_PREFIX + measure
        rec = RunRecord(kind, pair.name, status, note) if record else None
        if rec:
            rec.set(seed=0, beta=stats["beta"], dataset=dataset,
                   measure=measure, phases=phases, theta=theta,
                   eccen_support=eccen_support, n_anch=params["n_anch"],
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
                    n_anch = params["n_anch"]
                    anchor_nodes = nodes_a[:min(n_anch, len(nodes_a))]
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
                                  ("pure_chi", "pure_gamma", "chi", "gamma")})
                    rec.finish(exit_status, error, console.getvalue(), verbose)
                    out["run_id"] = rec.run_id

        if verbose:
            print(f"  [{measure}] pure chi {out['pure_chi']:.4f}  "
                  f"gamma {out['pure_gamma']:.4f}", file=sys.stderr)
            if "chi" in out:
                print(f"  [{measure}] integrated chi {out['chi']:.4f}  "
                      f"gamma {out['gamma']:.4f}", file=sys.stderr)
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

    print("all checks passed" if ok else "CHECKS FAILED", file=sys.stderr)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Index-rank-matching baseline (one or more centrality "
                    "measures, report section 2's own method).")
    ap.add_argument("--pair", type=str,
                    help="Pair directory from the beta noise generator "
                         "(flat layout, not --compat-layout).")
    ap.add_argument("--dataset", type=str, choices=sorted(DATASETS),
                    help="Which Table 1 N_anch/M row to use.")
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

    run(Path(args.pair), args.dataset, measures=args.measures,
        phases=args.phases, theta=args.theta, eccen_support=args.eccen_support,
        status=args.status, note=args.note, record=not args.no_record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
