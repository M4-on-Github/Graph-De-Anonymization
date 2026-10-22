"""Classic structural-fingerprint baseline: degree + clustering + egonet size,
matched by plain cosine similarity.

This is the pre-embedding de-anonymization approach (Narayanan & Shmatikov
2009-era): describe each node by a handful of local structural statistics
and match across graphs by similarity of that description. No GAE, no
training, no CSLS hubness correction -- it exists as the floor Adv-GD and its
own ablation (`adv_gd.py --skip-adversarial`) should both clear. If a learned
embedding does no better than three hand-picked numbers and cosine, the
embedding is not earning its complexity.

Feature vector per node (all computed from topology alone, same as the GAE's
input):
    degree             nx.degree
    clustering         nx.clustering (triangle density around the node)
    egonet size        number of nodes in the radius-1 ego graph

The three are on wildly different scales (degree can run into the thousands
on a hub, clustering is in [0, 1]), so each is z-scored within its own graph
before the vectors are compared -- without that, cosine similarity is
dominated by whichever feature happens to have the largest raw magnitude, and
the other two contribute almost nothing. z-scoring per graph rather than
jointly across both graphs is deliberate: the baseline should not get to see
both graphs' statistics at once to decide how to normalize, which is exactly
the kind of leakage the perturbation is supposed to test robustness against.

Anchors and candidates are produced exactly as `adv_gd.alignment.align` does
for Phase 2 (mutual nearest neighbour among the top N_anch degree nodes for
anchors, top-10 matches for the top M degree nodes as candidates), substituting
plain cosine similarity for CSLS, so that a difference between this file's
numbers and Adv-GD's is a difference in representation and matching, not in
how generously each one was allowed to search.

CLI:
    python src/baselines/structural_fingerprint.py --pair runs/sweep_nominal/cit-HepTh_b10_s0 --dataset hepth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATASETS, RunRecord, capture_stderr, candidate_sets,  # noqa: E402
                    evaluate_anchors, load_pair, mutual_anchors, propagate,
                    evaluate, top_degree, TOP10, REPO_ROOT)

KIND = "baseline_structural_fingerprint"


def fingerprint(G: nx.Graph, nodes: list) -> np.ndarray:
    """The three-feature vector per node, z-scored within this graph.

    `nodes` fixes row order so the matrix lines up with every other array
    this module builds from the same graph.
    """
    deg = dict(G.degree())
    clu = nx.clustering(G)
    ego = {n: nx.ego_graph(G, n, radius=1).number_of_nodes() for n in G.nodes()}

    X = np.array([[deg[n], clu[n], ego[n]] for n in nodes], dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # a constant column (possible on a tiny graph) must not divide by zero
    return ((X - mean) / std).astype(np.float32)


def _unit_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def cosine_matrix(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    return _unit_rows(src) @ _unit_rows(tgt).T


def match(Fa: np.ndarray, nodes_a: list, Fu: np.ndarray, nodes_u: list,
         Ga: nx.Graph, Gu: nx.Graph, n_anch: int, m: int,
         verbose: bool = True) -> dict:
    """Anchors (mutual cosine NN among top-N_anch degree nodes) and candidates
    (top-10 cosine matches for the top-M degree nodes), in the same shape
    `adv_gd.alignment.align` returns them in.
    """
    import torch
    anch_a = top_degree(Ga, nodes_a, n_anch)
    anch_u = top_degree(Gu, nodes_u, n_anch)
    s_anch = torch.as_tensor(cosine_matrix(Fa[anch_a], Fu[anch_u]))
    pairs = mutual_anchors(s_anch)
    anchors = [(nodes_a[anch_a[i]], nodes_u[anch_u[j]]) for i, j in pairs]
    if verbose:
        print(f"  anchors: {len(anchors):,} mutual cosine nearest neighbours "
              f"among top {len(anch_a):,} degree nodes", file=sys.stderr)

    cand_a = top_degree(Ga, nodes_a, m)
    s_cand = torch.as_tensor(cosine_matrix(Fa[cand_a], Fu))
    top = candidate_sets(s_cand, TOP10).numpy()
    candidates = {nodes_a[cand_a[i]]: [nodes_u[j] for j in top[i]]
                  for i in range(len(cand_a))}
    return {"anchors": anchors, "candidates": candidates}


def run(pair: Path, dataset: str, phases: str = "123", theta: float = 0.5,
       eccen_support: str = "all", status: str = "exploratory", note: str = "",
       record: bool = True, verbose: bool = True) -> dict:
    """Run the baseline on one pair. Mirrors `adv_gd.adv_gd.run`'s shape so a
    reader moving between the two files finds the same structure.
    """
    params = DATASETS[dataset]
    Ga, Gu, truth, stats = load_pair(pair)

    rec = RunRecord(KIND, pair.name, status, note) if record else None
    if rec:
        # seed is fixed, not swept: nothing here is stochastic -- z-scoring,
        # cosine and top-degree selection are all deterministic given the
        # graph, so there is no sense in which a different seed would be a
        # different run. Recorded as 0 by convention, per the logging
        # policy's requirement that every record state a seed.
        rec.set(seed=0, beta=stats["beta"], dataset=dataset, phases=phases,
                theta=theta, eccen_support=eccen_support,
                n_anch=params["n_anch"], m=params["m"],
                matchable_nodes=stats.get("matchable_nodes"),
                pair_stats=stats, pair_dir=str(pair))
        rec.add_inputs(pair / "G1.edgelist", pair / "G2.edgelist",
                       pair / "mapping.txt", pair / "stats.json")
        rec.seal_id()

    t0 = time.time()
    with capture_stderr() as console:
        try:
            nodes_a = sorted(Ga.nodes())
            nodes_u = sorted(Gu.nodes())
            if verbose:
                print(f"pair {pair.name}  beta={stats['beta']}  "
                      f"G1 {Ga.number_of_nodes():,}n  G2 {Gu.number_of_nodes():,}n",
                      file=sys.stderr)
            Fa = fingerprint(Ga, nodes_a)
            Fu = fingerprint(Gu, nodes_u)
            m = match(Fa, nodes_a, Fu, nodes_u, Ga, Gu,
                     n_anch=params["n_anch"], m=params["m"], verbose=verbose)
            score = evaluate_anchors(m["anchors"], truth)

            out = {
                "pair": pair.name, "dataset": dataset, "beta": stats["beta"],
                "matchable_nodes": stats.get("matchable_nodes"),
                "anchors_found": score["found"],
                "anchors_correct": score["correct"],
                "anchor_precision": score["precision"],
                "seconds": round(time.time() - t0, 1),
            }
            if "3" in phases:
                mapping = propagate(Ga, Gu, m["anchors"], m["candidates"],
                                    theta=theta, support=eccen_support,
                                    verbose=verbose)
                final = evaluate(mapping, truth, stats.get("matchable_nodes"))
                mpath = pair / f"out_mapping_{KIND}_{rec.run_id if rec else 'unrecorded'}.txt"
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
                              ("anchors_found", "anchors_correct",
                               "anchor_precision", "mapped", "correct",
                               "chi", "gamma")})
                rec.finish(exit_status, error, console.getvalue(), verbose)
                out["run_id"] = rec.run_id

    (pair / "structural_fingerprint_result.json").write_text(
        json.dumps(out, indent=2))
    if verbose:
        print(f"  anchors {out['anchors_correct']}/{out['anchors_found']}  "
              f"precision {out['anchor_precision']:.4f}", file=sys.stderr)
        if "chi" in out:
            print(f"  de-anonymized {out['correct']:,}/{out['mapped']:,} mapped  "
                  f"chi {out['chi']:.4f}  gamma {out['gamma']:.4f}", file=sys.stderr)
        print(f"  {time.time() - t0:.0f}s total", file=sys.stderr)
    return out


def self_test(verbose: bool = True) -> bool:
    """Synthetic check: fingerprint computation must be exactly invariant
    under relabelling -- node n's vector in G and node relabel[n]'s vector in
    the relabelled copy must be identical, since degree/clustering/egonet
    size are all isomorphism invariants.

    This is deliberately NOT a check that matching recovers the permutation.
    A sparse random graph has very few distinct (degree, clustering, egonet)
    combinations, so most nodes share a fingerprint with several others and
    are genuinely indistinguishable on these three features alone -- cosine
    matching is exactly tied among them, and which one wins the mutual-NN
    tie-break is arbitrary. That ambiguity is a true property of this
    baseline's weakness (the reason it is the floor, not Adv-GD's row), not a
    pipeline bug, so asserting high recovery here would be testing the wrong
    thing. The anchor-recovery number is still printed, as context.
    """
    ok = True
    rng = np.random.default_rng(0)
    G = nx.gnm_random_graph(300, 900, seed=0)
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
    nodes = sorted(G.nodes())

    perm = rng.permutation(nodes)
    relabel = dict(zip(nodes, perm))
    Gp = nx.relabel_nodes(G, relabel)
    truth = {n: relabel[n] for n in nodes}

    nodes_a = sorted(G.nodes())
    nodes_u = sorted(Gp.nodes())
    Fa = fingerprint(G, nodes_a)
    Fu = fingerprint(Gp, nodes_u)

    idx_u = {n: i for i, n in enumerate(nodes_u)}
    mismatches = sum(
        0 if np.allclose(Fa[i], Fu[idx_u[relabel[n]]], atol=1e-5) else 1
        for i, n in enumerate(nodes_a))
    if mismatches:
        print(f"  [FAIL] {mismatches}/{len(nodes_a)} nodes' fingerprints "
              f"changed under relabelling; feature computation is not "
              f"isomorphism-invariant", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] fingerprints are exactly relabelling-invariant "
              f"({len(nodes_a)} nodes)", file=sys.stderr)

    n_unique = len(set(map(tuple, np.round(Fa, 4))))
    m = match(Fa, nodes_a, Fu, nodes_u, G, Gp, n_anch=300, m=300, verbose=False)
    correct = sum(1 for a, u in m["anchors"] if truth[a] == u)
    print(f"  [INFO] only {n_unique}/{len(nodes_a)} nodes have a unique "
          f"fingerprint on this synthetic graph, so mutual-NN recovered just "
          f"{correct}/{len(m['anchors'])} anchors despite perfect features -- "
          f"expected: this is exactly the ambiguity that makes this a weak "
          f"baseline", file=sys.stderr)

    print("all checks passed" if ok else "CHECKS FAILED", file=sys.stderr)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Structural-fingerprint + cosine baseline.")
    ap.add_argument("--pair", type=str,
                    help="Pair directory from the beta noise generator "
                         "(flat layout, not --compat-layout).")
    ap.add_argument("--dataset", type=str, choices=sorted(DATASETS),
                    help="Which Table 1 N_anch/M row to use.")
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

    run(Path(args.pair), args.dataset, phases=args.phases, theta=args.theta,
        eccen_support=args.eccen_support, status=args.status, note=args.note,
        record=not args.no_record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
