"""Classic IsoRank baseline (Singh, Xu & Berger, 2008): seed-free topological
alignment by power-iterating a product-graph eigenproblem, with no learned
embedding and no training.

IsoRank's recurrence treats "node i in G1 matches node j in G2" as a kind of
PageRank mass that flows along edges in both graphs at once. Singh et al.'s
element-wise form is

    R[i, j] = sum_{u in N(i)} sum_{v in N(j)} R[u, v] / (|N(u)| |N(v)|)

-- each neighbour pair (u, v) *distributes* its score over its own
neighbourhoods, dividing by the NEIGHBOUR's degree. In matrix form, with the
prior mixed in:

    R <- alpha * A1n @ R @ A2n^T + (1 - alpha) * R0,   Akn = Ak @ Dk^-1

i.e. A1n/A2n are COLUMN-normalized (column-stochastic) adjacency matrices,
and R0 is the prior (here uniform, since this is the seed-free setting --
classic IsoRank is usually given a sequence- or attribute-based prior, which
does not exist for these synthetic perturbation pairs). alpha=0.6 and 20
rounds are Singh et al.'s own defaults, both exposed as CLI flags.

Bug fixed 2026-10-18: this file previously ROW-normalized (D^-1 A), which
divides by the degree of i and j themselves and so *averages* the neighbour
scores instead of distributing them. Averaging a constant returns the same
constant, so a uniform R0 was an exact fixed point: R never moved off
1/(n1*n2), every "argmax" was a floating-point tie-break, and the old
"hub collapse" reading of the resulting 2404/2500-shared-argmax rows on the
beta=0 HepTh pair was a misdiagnosis of a signal-free matrix. Every
baseline_isorank record before this date was produced by that bug.

What a uniform prior can and cannot give (derivation, not tuning): starting
from R0 = c 1 1^T, the recurrence keeps R inside
span{ (A1n^k 1)(A2n^k 1)^T : k = 0..rounds } -- a short sum of rank-1
outer products of k-step walk-count vectors. R is therefore a Gram-like
kernel between nodes' walk-centrality profiles, NOT a pairwise "my
neighbours match your neighbours" signal. Two consequences:
  * per-row argmax collapses onto the one or two G2 nodes with the largest
    profile (genuine hub collapse, by construction), so argmax diversity is
    not a quality measure here and should not be read as one;
  * the greedy one-to-one extraction below is what carries the signal: on a
    near-rank-1 positive matrix it reduces to rank-matching both graphs on
    their walk-centrality profiles, and on an exactly isomorphic, randomly
    relabelled copy it recovers the true permutation (checked in
    `self_test`), up to nodes whose profiles tie exactly.
So with no prior, this baseline is best understood as a walk-centrality
rank-matching method -- a legitimate seed-free topology floor, but expect it
to degrade quickly under edge noise, like the other rank-matching baselines.

Scalability restriction (deliberate, documented, not a bug): dense IsoRank
costs O(n1^2 * n2 + n1 * n2^2) time and O(n1 * n2) memory per iteration,
because R is a full n1 x n2 matrix and the update is two dense matrix
products (n1^2 * n2^2 is the Kronecker-product formulation, not this one).
That is intractable at HepTh's ~28k nodes. This file restricts both graphs to
their top-M highest-degree nodes (M much smaller than Adv-GD's own M, default
2,500, CLI-overridable) via induced subgraphs before running IsoRank at all --
a real limitation of the classic method being reproduced faithfully, not a
design choice specific to this repo.

Matching is extracted from the converged R by the standard IsoRank greedy
one-to-one extraction: sort all (i, j) pairs by R[i, j] descending, walk the
list, and take a pair only if neither i nor j has already been claimed.
Anchors are the top-N_anch pairs from that same ranking; candidates are the
top-10 R-matches per node, reusing `common.candidate_sets` exactly as the
other baselines and Adv-GD itself do, so scoring is on equal footing.

CLI:
    python src/baselines/isorank.py --pair runs/sweep_nominal/cit-HepTh_b10_s0 --dataset hepth
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


def column_normalize(A: np.ndarray) -> np.ndarray:
    """A @ D^-1: divide column u by deg(u), so every non-isolated column sums
    to 1 (column-stochastic). This is Singh et al.'s 1/|N(u)| factor; see the
    module docstring for why the row-normalized D^-1 A is wrong here.
    """
    col_sums = A.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0  # isolated node after the top-M restriction: leave its column all-zero
    return A / col_sums


def isorank_scores(G1: nx.Graph, nodes1: list, G2: nx.Graph, nodes2: list,
                   alpha: float = 0.6, rounds: int = 20,
                   verbose: bool = True) -> np.ndarray:
    """Converged R (n1 x n2), dense. Both graphs must already be restricted
    to a tractable node count by the caller -- this function does not
    subsample.
    """
    A1 = nx.to_numpy_array(G1, nodelist=nodes1)
    A2 = nx.to_numpy_array(G2, nodelist=nodes2)
    A1n = column_normalize(A1)
    A2n = column_normalize(A2)

    n1, n2 = len(nodes1), len(nodes2)
    R0 = np.full((n1, n2), 1.0 / (n1 * n2), dtype=np.float64)
    R = R0.copy()
    for t in range(rounds):
        R = alpha * (A1n @ R @ A2n.T) + (1 - alpha) * R0
        total = R.sum()
        if total > 0:
            # Column-stochastic A1n/A2n conserve R's total mass except what
            # isolated nodes (all-zero columns) drop; renormalize so R stays
            # on the probability simplex either way.
            R /= total
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
    """Checks of the recurrence against Singh et al.'s own equation, and of
    the whole pipeline against a case with a known answer.

    The previous version of this test asserted that one round from a uniform
    R0 is a no-op. That is exactly the bug the module docstring describes --
    under the correct normalization a uniform prior is a fixed point only on
    a regular graph -- so the test was enshrining it. Checks now:
      1. Element-wise equation: one application of the matrix recurrence
         equals Singh et al.'s sum_{u~i} sum_{v~j} R[u,v]/(deg u * deg v),
         computed by explicit loops, on non-regular graphs and a random R.
         A row/column normalization mix-up fails this by O(1).
      2. Uniform prior is not a fixed point on a non-regular graph, and one
         round equals its closed form
         normalize(alpha*c*(A1n 1)(A2n 1)^T + (1-alpha)*c*1 1^T).
      3. Greedy extraction on a hand-built R with an unambiguous maximum per
         row/column recovers that assignment exactly.
      4. Gold standard: IsoRank on a graph and a randomly relabelled
         isomorphic copy, followed by greedy extraction, recovers the true
         permutation on >= 99% of nodes (the module docstring explains why
         this holds under a uniform prior). The relabelling is random so
         that sorted-id order carries no information; under the old row
         normalization this check scored ~0%.
    Per-row argmax diversity is printed but deliberately not asserted: with a
    uniform prior it collapses by construction (module docstring).
    """
    ok = True

    def _graph(G):
        G = nx.relabel_nodes(G, {n: f"n{n:04d}" for n in G.nodes()})
        G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
        return G, sorted(G.nodes())

    G, nodes_g = _graph(nx.gnm_random_graph(60, 150, seed=1))
    H, nodes_h = _graph(nx.gnm_random_graph(50, 120, seed=2))
    alpha = 0.6

    # 1. element-wise Singh et al. equation vs. the matrix form
    A1n = column_normalize(nx.to_numpy_array(G, nodelist=nodes_g))
    A2n = column_normalize(nx.to_numpy_array(H, nodelist=nodes_h))
    rng = np.random.default_rng(0)
    Rr = rng.uniform(size=(len(nodes_g), len(nodes_h)))
    ig = {u: k for k, u in enumerate(nodes_g)}
    ih = {v: k for k, v in enumerate(nodes_h)}
    ref = np.array([[sum(Rr[ig[u], ih[v]] / (G.degree(u) * H.degree(v))
                         for u in G[i] for v in H[j])
                     for j in nodes_h] for i in nodes_g])
    diff = np.abs(A1n @ Rr @ A2n.T - ref).max()
    if diff > 1e-12:
        print(f"  [FAIL] matrix recurrence != Singh et al. element-wise "
              f"equation; max diff {diff:.3e}", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] matrix recurrence matches Singh et al.'s element-wise "
              f"equation (max diff {diff:.1e})", file=sys.stderr)

    # 2. uniform prior is not a fixed point; round 1 equals its closed form
    R1 = isorank_scores(G, nodes_g, H, nodes_h, alpha=alpha, rounds=1, verbose=False)
    c = 1.0 / (len(nodes_g) * len(nodes_h))
    closed = (alpha * c * np.outer(A1n.sum(axis=1), A2n.sum(axis=1))
              + (1 - alpha) * c)
    closed /= closed.sum()
    moved = np.abs(R1 - c).max()
    off = np.abs(R1 - closed).max()
    if moved < 1e-3 * c or off > 1e-15:
        print(f"  [FAIL] round 1 from uniform R0: max |R1 - R0| = {moved:.3e} "
              f"(must be clearly nonzero), max |R1 - closed form| = {off:.3e}",
              file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] uniform R0 is not a fixed point (max |R1 - R0| = "
              f"{moved:.3e} = {moved / c:.2f} x R0) and round 1 matches its "
              f"closed form (max diff {off:.1e})", file=sys.stderr)

    # 3. greedy extraction on an unambiguous hand-built matrix
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

    # 4. gold standard: randomly relabelled isomorphic copy
    for name, base in (("gnm(300, 900)", nx.gnm_random_graph(300, 900, seed=1)),
                       ("barabasi_albert(300, 3)",
                        nx.barabasi_albert_graph(300, 3, seed=1))):
        Gi, nodes_i = _graph(base)
        perm = np.random.default_rng(7).permutation(len(nodes_i))
        truth = {u: f"m{perm[k]:04d}" for k, u in enumerate(nodes_i)}
        Hi = nx.relabel_nodes(Gi, truth)
        nodes_j = sorted(Hi.nodes())
        R = isorank_scores(Gi, nodes_i, Hi, nodes_j, alpha=alpha, rounds=20,
                           verbose=False)
        got = dict(greedy_extraction(R, nodes_i, nodes_j))
        acc = sum(got.get(u) == truth[u] for u in nodes_i) / len(nodes_i)
        distinct = len(set(R.argmax(axis=1).tolist()))
        tag = "PASS" if acc >= 0.99 else "FAIL"
        ok &= acc >= 0.99
        print(f"  [{tag}] {name} vs. a randomly relabelled isomorphic copy: "
              f"greedy extraction recovers {acc:.4f} of the true permutation "
              f"(need >= 0.99)", file=sys.stderr)
        print(f"  [INFO]   per-row argmax hits only {distinct}/{len(nodes_i)} "
              f"distinct columns -- the expected uniform-prior collapse, "
              f"not a failure", file=sys.stderr)

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
