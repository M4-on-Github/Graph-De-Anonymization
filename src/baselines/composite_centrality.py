"""This repo's own baseline: a richer per-node feature vector than
`structural_fingerprint.py`'s three (degree, clustering, egonet size), plus
three of the global centrality measures `centrality_rank_match.py` scores on
their own (k-core, pagerank, eigenvector), matched by CSLS rather than plain
cosine.

The question this baseline is built to answer: `structural_fingerprint.py`'s
cosine match is weak mostly because three features leave most nodes with
duplicate fingerprints (see that file's self-test), and separately, index
rank-matching on any single global measure is weak because sorting throws
away everything except rank position. Neither weakness is about learning --
no part of either baseline is trained. This file tests whether just adding
more of the same kind of information (six features instead of three) and
switching to CSLS (which corrects for a few feature vectors acting as
"hubs" that every query would otherwise falsely claim as its nearest
neighbour) closes any of the gap to Adv-GD's own learned embeddings,
entirely without training -- i.e. whether the gap is about model capacity,
or just about how many hand-picked numbers you bother to compute.

Feature vector per node, z-scored per graph exactly as in
`structural_fingerprint.py` (six columns now instead of three):
    degree, clustering, egonet size         (from structural_fingerprint's
                                              own definitions, reused here)
    k_core, pagerank, eigenvector            (from centrality.py)

CSLS (Cross-domain Similarity Local Scaling, Conneau et al. 2018 /
`adv_gd.alignment.csls_matrix`) replaces cosine for both anchor and
candidate matching: 2*cos(x,y) - r_src(x) - r_tgt(y), where r is the mean
similarity to the k nearest neighbours on the OTHER side. A plain hub (a
fingerprint many nodes are each other's nearest cosine neighbour to) gets a
large r and so a penalised CSLS score, which is exactly the failure mode
`structural_fingerprint.py`'s self-test calls out (many nodes sharing a
fingerprint, all equally "nearest" to each other).

CLI:
    python src/baselines/composite_centrality.py --pair runs/sweep_divergence/cit-HepTh_b10_s0 --dataset hepth
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
                    evaluate, top_degree, TOP10, csls_matrix, DEFAULTS, REPO_ROOT)
from centrality import k_core_scores, pagerank_scores, eigenvector_scores  # noqa: E402

KIND = "ours_composite_centrality"


def fingerprint(G: nx.Graph, nodes: list) -> np.ndarray:
    """The six-feature vector per node, z-scored within this graph. Same
    z-scoring rationale as `structural_fingerprint.fingerprint`: the six
    columns run on very different scales, and normalising within one graph
    at a time (never jointly across both) keeps the baseline from secretly
    peeking at both graphs' statistics to decide how to scale.
    """
    deg = dict(G.degree())
    clu = nx.clustering(G)
    ego = {n: nx.ego_graph(G, n, radius=1).number_of_nodes() for n in G.nodes()}
    core = k_core_scores(G)
    pr = pagerank_scores(G)
    eig = eigenvector_scores(G)

    X = np.array([[deg[n], clu[n], ego[n], core[n], pr[n], eig[n]]
                 for n in nodes], dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32)


def match(Fa: np.ndarray, nodes_a: list, Fu: np.ndarray, nodes_u: list,
         Ga: nx.Graph, Gu: nx.Graph, n_anch: int, m: int, csls_k: int,
         verbose: bool = True) -> dict:
    """Anchors (mutual CSLS nearest neighbour among top-N_anch degree nodes)
    and candidates (top-10 CSLS matches for the top-M degree nodes), same
    shape `structural_fingerprint.match` returns, substituting CSLS for
    cosine.
    """
    import torch
    anch_a = top_degree(Ga, nodes_a, n_anch)
    anch_u = top_degree(Gu, nodes_u, n_anch)
    s_anch = csls_matrix(torch.as_tensor(Fa[anch_a]), torch.as_tensor(Fu[anch_u]), csls_k)
    pairs = mutual_anchors(s_anch)
    anchors = [(nodes_a[anch_a[i]], nodes_u[anch_u[j]]) for i, j in pairs]
    if verbose:
        print(f"  anchors: {len(anchors):,} mutual CSLS nearest neighbours "
              f"(k={csls_k}) among top {len(anch_a):,} degree nodes", file=sys.stderr)

    cand_a = top_degree(Ga, nodes_a, m)
    s_cand = csls_matrix(torch.as_tensor(Fa[cand_a]), torch.as_tensor(Fu), csls_k)
    top = candidate_sets(s_cand, TOP10).numpy()
    candidates = {nodes_a[cand_a[i]]: [nodes_u[j] for j in top[i]]
                 for i in range(len(cand_a))}
    return {"anchors": anchors, "candidates": candidates}


def run(pair: Path, dataset: str, csls_k: int = DEFAULTS["csls_k"],
       phases: str = "123", theta: float = 0.5, eccen_support: str = "all",
       status: str = "exploratory", note: str = "", record: bool = True,
       verbose: bool = True) -> dict:
    """Mirrors `structural_fingerprint.run`'s shape -- same pipeline, six
    features and CSLS instead of three and cosine.
    """
    params = DATASETS[dataset]
    Ga, Gu, truth, stats = load_pair(pair)

    rec = RunRecord(KIND, pair.name, status, note) if record else None
    if rec:
        rec.set(seed=0, beta=stats["beta"], dataset=dataset, phases=phases,
                theta=theta, eccen_support=eccen_support, csls_k=csls_k,
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
                     n_anch=params["n_anch"], m=params["m"], csls_k=csls_k,
                     verbose=verbose)
            score = evaluate_anchors(m["anchors"], truth)

            out = {
                "pair": pair.name, "dataset": dataset, "beta": stats["beta"],
                "matchable_nodes": stats.get("matchable_nodes"),
                "csls_k": csls_k,
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

    (pair / "composite_centrality_result.json").write_text(json.dumps(out, indent=2))
    if verbose:
        print(f"  anchors {out['anchors_correct']}/{out['anchors_found']}  "
              f"precision {out['anchor_precision']:.4f}", file=sys.stderr)
        if "chi" in out:
            print(f"  de-anonymized {out['correct']:,}/{out['mapped']:,} mapped  "
                  f"chi {out['chi']:.4f}  gamma {out['gamma']:.4f}", file=sys.stderr)
        print(f"  {time.time() - t0:.0f}s total", file=sys.stderr)
    return out


def self_test(verbose: bool = True) -> bool:
    """Synthetic check: the six-feature fingerprint must be exactly
    relabelling-invariant, same style as `structural_fingerprint.self_test`.

    Unlike that file's self-test, this one does NOT assume the same
    "most nodes are indistinguishable" narrative going in -- six features
    plus CSLS hubness correction may resolve far more nodes uniquely than
    three features plus cosine. The anchor-recovery count is reported as
    plain [INFO], not pre-judged either way.
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
        0 if np.allclose(Fa[i], Fu[idx_u[relabel[n]]], atol=1e-4) else 1
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
    m = match(Fa, nodes_a, Fu, nodes_u, G, Gp, n_anch=300, m=300,
             csls_k=DEFAULTS["csls_k"], verbose=False)
    correct = sum(1 for a, u in m["anchors"] if truth[a] == u)
    print(f"  [INFO] {n_unique}/{len(nodes_a)} nodes have a unique "
          f"6-feature fingerprint on this synthetic graph; mutual-CSLS-NN "
          f"recovered {correct}/{len(m['anchors'])} anchors", file=sys.stderr)

    print("all checks passed" if ok else "CHECKS FAILED", file=sys.stderr)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="This repo's own baseline: 6-feature structural "
                    "fingerprint + CSLS matching.")
    ap.add_argument("--pair", type=str,
                    help="Pair directory from the beta noise generator "
                         "(flat layout, not --compat-layout).")
    ap.add_argument("--dataset", type=str, choices=sorted(DATASETS),
                    help="Which Table 1 N_anch/M row to use.")
    ap.add_argument("--csls-k", type=int, default=DEFAULTS["csls_k"])
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

    run(Path(args.pair), args.dataset, csls_k=args.csls_k, phases=args.phases,
        theta=args.theta, eccen_support=args.eccen_support,
        status=args.status, note=args.note, record=not args.no_record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
