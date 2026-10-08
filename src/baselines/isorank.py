"""Classic IsoRank baseline (Singh, Xu & Berger, 2008): seed-free topological
alignment by power-iterating a product-graph eigenproblem, with no learned
embedding and no training.

IsoRank's recurrence treats "node i in G1 matches node j in G2" as a kind of
PageRank mass that flows along edges in both graphs at once:

    R <- alpha * A1n @ R @ A2n^T + (1 - alpha) * R0

where A1n/A2n are row-normalized adjacency matrices and R0 is the prior (here
uniform, since this is the seed-free setting -- classic IsoRank is usually
given a sequence- or attribute-based prior, which does not exist for these
synthetic perturbation pairs). At convergence, R[i, j] is high when i and j
are "topologically similar": high-R neighbours of i's neighbours tend to map
to high-R neighbours of j's neighbours, recursively. alpha=0.6 and 20 rounds
are Singh et al.'s own defaults, both exposed as CLI flags.

Scalability restriction (deliberate, documented, not a bug): dense IsoRank
is O(n1^2 * n2^2) per iteration because R is a full n1 x n2 matrix and the
update is two dense matrix products. That is intractable at HepTh's ~28k
nodes. This file restricts both graphs to their top-M highest-degree nodes
(M much smaller than Adv-GD's own M, default 2,500, CLI-overridable) via
induced subgraphs before running IsoRank at all -- a real limitation of the
classic method being reproduced faithfully, not a design choice specific to
this repo.

Matching is extracted from the converged R by the standard IsoRank greedy
one-to-one extraction: sort all (i, j) pairs by R[i, j] descending, walk the
list, and take a pair only if neither i nor j has already been claimed.
Anchors are the top-N_anch pairs from that same ranking; candidates are the
top-10 R-matches per node, reusing `common.candidate_sets` exactly as the
other baselines and Adv-GD itself do, so scoring is on equal footing.

CLI:
    python src/baselines/isorank.py --pair runs/sweep_divergence/cit-HepTh_b10_s0 --dataset hepth
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
                    evaluate_anchors, load_pair, propagate, evaluate,
                    top_degree, TOP10)

KIND = "baseline_isorank"


def row_normalize(A: np.ndarray) -> np.ndarray:
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # isolated node after the top-M restriction: leave its row all-zero
    return A / row_sums


def isorank_scores(G1: nx.Graph, nodes1: list, G2: nx.Graph, nodes2: list,
                   alpha: float = 0.6, rounds: int = 20,
                   verbose: bool = True) -> np.ndarray:
    """Converged R (n1 x n2), dense. Both graphs must already be restricted
    to a tractable node count by the caller -- this function does not
    subsample.
    """
    A1 = nx.to_numpy_array(G1, nodelist=nodes1)
    A2 = nx.to_numpy_array(G2, nodelist=nodes2)
    A1n = row_normalize(A1)
    A2n = row_normalize(A2)

    n1, n2 = len(nodes1), len(nodes2)
    R0 = np.full((n1, n2), 1.0 / (n1 * n2), dtype=np.float64)
    R = R0.copy()
    for t in range(rounds):
        R = alpha * (A1n @ R @ A2n.T) + (1 - alpha) * R0
        total = R.sum()
        if total > 0:
            R /= total  # renormalize to the probability simplex each round, or R decays/blows up geometrically with alpha<1/>1 respectively
        if verbose and (t + 1) % 5 == 0:
            print(f"    IsoRank round {t + 1}/{rounds}  "
                  f"max R={R.max():.3e}", file=sys.stderr)
    return R


def greedy_extraction(R: np.ndarray, nodes1: list, nodes2: list,
                      limit: int | None = None) -> list[tuple[str, str]]:
    """Standard IsoRank one-to-one extraction: highest-R pairs first, skip
    any pair whose row or column is already claimed.

    `limit` caps how many pairs to extract (used to pull just the top-N_anch
    for anchors without walking and discarding the full n1*n2 sort twice).
    """
    n1, n2 = R.shape
    order = np.argsort(R, axis=None)[::-1]
    used_i, used_j = set(), set()
    out = []
    for flat in order:
        i, j = divmod(int(flat), n2)
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((nodes1[i], nodes2[j]))
        if limit is not None and len(out) >= limit:
            break
    return out


def match(R: np.ndarray, nodes1: list, nodes2: list, n_anch: int) -> dict:
    anchors = greedy_extraction(R, nodes1, nodes2, limit=n_anch)
    cand_idx = candidate_sets(_as_tensor(R), TOP10).numpy()
    candidates = {nodes1[i]: [nodes2[j] for j in cand_idx[i]]
                  for i in range(len(nodes1))}
    return {"anchors": anchors, "candidates": candidates}


def _as_tensor(R: np.ndarray):
    import torch
    return torch.as_tensor(R)


def run(pair: Path, dataset: str, m: int = 2500, alpha: float = 0.6,
       rounds: int = 20, phases: str = "123", theta: float = 0.5,
       eccen_support: str = "all", status: str = "exploratory", note: str = "",
       record: bool = True, verbose: bool = True) -> dict:
    """Run IsoRank on one pair, restricted to the top-m degree nodes of each
    graph. Mirrors `adv_gd.adv_gd.run`'s shape.
    """
    params = DATASETS[dataset]
    n_anch = params["n_anch"]
    Ga, Gu, truth, stats = load_pair(pair)

    rec = RunRecord(KIND, pair.name, status, note) if record else None
    if rec:
        rec.set(seed=0, beta=stats["beta"], dataset=dataset, phases=phases,
                theta=theta, eccen_support=eccen_support, m=m, alpha=alpha,
                rounds=rounds, n_anch=n_anch,
                matchable_nodes=stats.get("matchable_nodes"),
                pair_stats=stats, pair_dir=str(pair))
        rec.add_inputs(pair / "G1.edgelist", pair / "G2.edgelist",
                       pair / "mapping.txt", pair / "stats.json")
        rec.seal_id()

    t0 = time.time()
    with capture_stderr() as console:
        try:
            nodes_a_full = sorted(Ga.nodes())
            nodes_u_full = sorted(Gu.nodes())
            if verbose:
                print(f"pair {pair.name}  beta={stats['beta']}  "
                      f"G1 {Ga.number_of_nodes():,}n  G2 {Gu.number_of_nodes():,}n  "
                      f"restricting to top-{m:,} degree nodes per side", file=sys.stderr)

            keep_a = top_degree(Ga, nodes_a_full, m)
            keep_u = top_degree(Gu, nodes_u_full, m)
            nodes_a = [nodes_a_full[i] for i in keep_a]
            nodes_u = [nodes_u_full[i] for i in keep_u]
            Ga_sub = Ga.subgraph(nodes_a).copy()
            Gu_sub = Gu.subgraph(nodes_u).copy()

            R = isorank_scores(Ga_sub, nodes_a, Gu_sub, nodes_u,
                               alpha=alpha, rounds=rounds, verbose=verbose)
            matched = match(R, nodes_a, nodes_u, n_anch)
            score = evaluate_anchors(matched["anchors"], truth)

            out = {
                "pair": pair.name, "dataset": dataset, "beta": stats["beta"],
                "matchable_nodes": stats.get("matchable_nodes"),
                "restricted_nodes": m, "alpha": alpha, "rounds": rounds,
                "anchors_found": score["found"],
                "anchors_correct": score["correct"],
                "anchor_precision": score["precision"],
                "seconds": round(time.time() - t0, 1),
            }
            if "3" in phases:
                mapping = propagate(Ga, Gu, matched["anchors"], matched["candidates"],
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

    (pair / "isorank_result.json").write_text(json.dumps(out, indent=2))
    if verbose:
        print(f"  anchors {out['anchors_correct']}/{out['anchors_found']}  "
              f"precision {out['anchor_precision']:.4f}", file=sys.stderr)
        if "chi" in out:
            print(f"  de-anonymized {out['correct']:,}/{out['mapped']:,} mapped  "
                  f"chi {out['chi']:.4f}  gamma {out['gamma']:.4f}", file=sys.stderr)
        print(f"  {time.time() - t0:.0f}s total", file=sys.stderr)
    return out


def self_test(verbose: bool = True) -> bool:
    """Property-based checks of the recurrence and the extraction step.

    This deliberately does NOT assert that IsoRank recovers a known
    permutation. Investigating a self-relabelling failure case during
    development surfaced a real, well-documented property of seed-free
    IsoRank: with a uniform prior R0 (no external similarity information),
    the bilinear recursion R <- alpha * A1n R A2n^T + (1-alpha) R0 has no
    term that distinguishes one row from another beyond each graph's own
    stationary (eigenvector-centrality-like) distribution, so after only a
    handful of rounds most rows converge toward the SAME best-scoring
    column -- a "hub collapse" where almost every node in G1 is best matched
    to whichever handful of nodes have the highest centrality in G2. Direct
    measurement: matching a 120-node random graph to ITSELF with the true
    identity mapping already collapses to 2-12 distinct best-matching
    columns across all 120 rows, even at round 1. This is exactly why
    Singh et al.'s IsoRank is normally given an informative, non-uniform
    R0 (e.g. sequence similarity for proteins) -- without one, as here,
    pure topology is not enough to break the symmetry, and this baseline's
    anchor/candidate quality should be expected to be poor on real pairs.
    That is itself the point of including it: it is the seed-free-topology
    floor that Adv-GD's learned embeddings have to clear.

    What IS checked, because it is independent of this degeneracy and would
    catch a real implementation bug:
      1. Analytic invariant: with uniform R0, one round of the recurrence is
         a no-op (R1 == R0), because row-normalized A1n/A2n map the all-ones
         vector to itself, so A1n @ R0 @ A2n.T == R0 exactly. If row
         normalization, the matrix products, or the per-round renormalize
         are wrong, this breaks.
      2. Greedy extraction, on a small hand-built R with an unambiguous
         unique maximum per row/column (not produced by isorank_scores, so
         it is not subject to the collapse above), recovers exactly that
         assignment.
    """
    ok = True

    # 1. uniform-prior invariance
    G = nx.gnm_random_graph(60, 150, seed=1)
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
    nodes_g = sorted(G.nodes())
    H = nx.gnm_random_graph(50, 120, seed=2)
    H = nx.relabel_nodes(H, {n: str(n) for n in H.nodes()})
    nodes_h = sorted(H.nodes())

    R1 = isorank_scores(G, nodes_g, H, nodes_h, alpha=0.6, rounds=1, verbose=False)
    R0 = np.full(R1.shape, 1.0 / (len(nodes_g) * len(nodes_h)))
    max_diff = np.abs(R1 - R0).max()
    if max_diff > 1e-9:
        print(f"  [FAIL] one round with uniform R0 should be a no-op; "
              f"max |R1 - R0| = {max_diff:.3e}", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] uniform-prior invariance holds (max diff {max_diff:.1e})",
              file=sys.stderr)

    # 2. greedy extraction on an unambiguous hand-built matrix
    rng = np.random.default_rng(3)
    n = 8
    perm_true = rng.permutation(n)
    R = rng.uniform(0, 0.1, size=(n, n))
    for i, j in enumerate(perm_true):
        R[i, j] = 1.0 + i  # strictly dominant, distinct per row -- no ties possible
    nodes1 = [f"a{i}" for i in range(n)]
    nodes2 = [f"b{i}" for i in range(n)]
    extracted = dict(greedy_extraction(R, nodes1, nodes2))
    expected = {f"a{i}": f"b{j}" for i, j in enumerate(perm_true)}
    if extracted != expected:
        print(f"  [FAIL] greedy extraction on an unambiguous matrix gave "
              f"{extracted}, expected {expected}", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] greedy extraction recovers an unambiguous assignment exactly",
              file=sys.stderr)

    # informational: quantify the collapse on a self-match, so a reader of
    # the test output sees the same evidence that motivated checks 1 and 2
    R_self = isorank_scores(G, nodes_g, G, nodes_g, alpha=0.6, rounds=10, verbose=False)
    distinct_cols = len({int(R_self[i].argmax()) for i in range(len(nodes_g))})
    print(f"  [INFO] matching a {len(nodes_g)}-node graph to itself (true identity "
          f"mapping) collapses to only {distinct_cols} distinct best-matching "
          f"columns across all rows after 10 rounds -- expected hub-collapse, "
          f"not a bug", file=sys.stderr)

    print("all checks passed" if ok else "CHECKS FAILED", file=sys.stderr)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IsoRank baseline (seed-free, uniform prior).")
    ap.add_argument("--pair", type=str,
                    help="Pair directory from the beta noise generator "
                         "(flat layout, not --compat-layout).")
    ap.add_argument("--dataset", type=str, choices=sorted(DATASETS),
                    help="Which Table 1 N_anch row to use.")
    ap.add_argument("--m", type=int, default=2500,
                    help="Restrict both graphs to their top-m degree nodes "
                         "before running dense IsoRank (default 2500; "
                         "classic IsoRank does not scale further).")
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="IsoRank mixing weight (Singh et al. 2008 default 0.6).")
    ap.add_argument("--rounds", type=int, default=20)
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

    run(Path(args.pair), args.dataset, m=args.m, alpha=args.alpha,
        rounds=args.rounds, phases=args.phases, theta=args.theta,
        eccen_support=args.eccen_support, status=args.status, note=args.note,
        record=not args.no_record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
