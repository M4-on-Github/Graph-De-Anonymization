"""Phase 3 of Adv-GD: propagating de-anonymization.

Takes the anchor pairs from Phase 2 as a starting mapping and spreads it across
the graph: an unmapped node is matched to whichever node in the other graph
shares the most already-matched neighbours, provided the best candidate stands
out clearly enough from the runner-up.

Paper provenance (docs/SeedFree_Li.etal.pdf, section 3.2, p.748):
    Algorithm 1 GetScores                  -- implemented in `get_scores`
    Algorithm 2 Propagating De-anonymization -- implemented in `propagate`
    Eccen(S) = (S_p - S_q) / sigma(S)      -- `eccentricity`
    theta = 0.5 for all datasets            (section 4.2, p.750)

This is the most faithfully reproducible part of Adv-GD: unlike Phases 1 and 2,
both algorithms are given as line-by-line pseudocode, and the one free
parameter has a stated value.

One ambiguity remains, and it is not minor. S is defined as "a zero vector of
size |G2|" -- roughly 27,000 entries for HepTh, almost all of which stay zero --
and sigma is "the standard of the vector". Taken literally, sigma is computed
over all those zeros, so it is tiny, Eccen is correspondingly huge, and theta
= 0.5 rejects nearly nothing. Computing sigma over the non-zero scores only
makes the threshold actually bind. The literal reading is the default here,
because it is what the paper says; `--eccen-support nonzero` selects the other.

Library use:
    from propagation import propagate
    mapping = propagate(Ga, Gu, anchors, candidates, theta=0.5)
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import networkx as nx


def eccentricity(scores: dict, universe: int, support: str = "all") -> float:
    """(largest - second largest) / standard deviation of the score vector.

    `scores` holds only the non-zero entries; `universe` is the full length the
    paper's zero vector would have. With support="all" the zeros are included
    in the standard deviation, which is the literal reading of the paper.
    """
    if not scores:
        return 0.0
    vals = sorted(scores.values(), reverse=True)
    top = vals[0]
    second = vals[1] if len(vals) > 1 else 0.0

    total = sum(vals)
    total_sq = sum(v * v for v in vals)
    n = universe if support == "all" else len(vals)
    if n <= 1:
        return 0.0

    mean = total / n
    var = total_sq / n - mean * mean
    if var <= 0:
        return 0.0
    return (top - second) / math.sqrt(var)


def get_scores(G1: nx.Graph, G2: nx.Graph, v1: str, phi: dict,
               mapped_targets: set) -> dict:
    """Algorithm 1: score every candidate in G2 against node v1 of G1.

    For each already-matched neighbour of v1, walk the neighbours of that
    neighbour's image in G2 and credit each unmatched one. The credit is
    1/sqrt(degree) so that high-degree nodes, which neighbour everything, do
    not win by default (paper section 3.2).
    """
    scores: dict[str, float] = {}
    for nb in G1[v1]:
        img = phi.get(nb)
        if img is None or img not in G2:
            continue
        for cand in G2[img]:
            if cand in mapped_targets:
                continue
            deg = G2.degree(cand)
            if deg:
                scores[cand] = scores.get(cand, 0.0) + 1.0 / math.sqrt(deg)
    return scores


def propagate(Ga: nx.Graph, Gu: nx.Graph, anchors: list, candidates: dict,
              theta: float = 0.5, support: str = "all",
              max_sweeps: int = 100, verbose: bool = True) -> dict:
    """Algorithm 2: two-stage propagation from the Phase 2 anchors.

    Stage one (flag = 0) only accepts a match that the Phase 2 candidate set
    already proposed, which keeps early mistakes from compounding. Once that
    stalls, stage two (flag = 1) drops the restriction and lets propagation run
    on topology alone.

    Returns the mapping from G^a ids to G^u ids, anchors included.
    """
    # Drop any anchor naming a node that is not in both graphs, then build the
    # two directions from the surviving pairs. Deriving psi from the filtered
    # pair list rather than by inverting phi keeps the key spaces straight:
    # phi is keyed by G^a ids, psi by G^u ids, and the reverse-direction
    # get_scores below is silently useless if they are ever the same.
    kept = [(a, u) for a, u in anchors if a in Ga and u in Gu]
    phi = {a: u for a, u in kept}                  # G^a -> G^u
    psi = {u: a for a, u in kept}                  # G^u -> G^a

    n_a, n_u = Gu.number_of_nodes(), Ga.number_of_nodes()
    flag = 0
    t0 = time.time()

    for sweep in range(1, max_sweeps + 1):
        added = 0
        for v_a in list(Ga.nodes()):
            if v_a in phi:
                continue

            s1 = get_scores(Ga, Gu, v_a, phi, set(psi))
            if eccentricity(s1, n_a, support) < theta:
                continue
            v_u = max(s1.items(), key=lambda kv: kv[1])[0]

            # Reverse check: v_u must pick v_a back, which is what stops two
            # G^a nodes from both claiming the same G^u node.
            s2 = get_scores(Gu, Ga, v_u, psi, set(phi))
            if eccentricity(s2, n_u, support) < theta:
                continue
            v_back = max(s2.items(), key=lambda kv: kv[1])[0]
            if v_back != v_a:
                continue

            # Stage one additionally requires the match to be in the candidate
            # set built by Phase 2 (Algorithm 2, line 11).
            if flag == 0 and v_u not in candidates.get(v_a, ()):
                continue

            phi[v_a] = v_u
            psi[v_u] = v_a
            added += 1

        if verbose:
            print(f"  sweep {sweep:3d}  stage {flag}  +{added:,} "
                  f"-> {len(phi):,} mapped  {time.time() - t0:.0f}s",
                  file=sys.stderr)

        if added == 0:
            if flag == 0:
                flag = 1            # candidate restriction lifted
                continue
            break

    return phi


def evaluate(mapping: dict, truth: dict, matchable: int | None = None) -> dict:
    """The paper's two metrics (section 4.3, p.750-751).

    chi   -- successfully de-anonymized nodes over |V^a intersect V^u|, the
             number that could in principle be recovered.
    gamma -- successfully de-anonymized nodes over the number actually mapped,
             i.e. the precision of the output.

    These differ sharply when a method maps only a confident subset, so a
    figure quoted without saying which one it is cannot be compared.
    """
    correct = sum(1 for a, u in mapping.items() if truth.get(a) == u)
    denom = matchable if matchable else len(truth)
    return {
        "mapped": len(mapping),
        "correct": correct,
        "chi": correct / denom if denom else float("nan"),
        "gamma": correct / len(mapping) if mapping else float("nan"),
        "matchable": denom,
    }


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------

def self_test(verbose: bool = True) -> bool:
    """Propagate on a graph paired with itself, seeded with perfect anchors.

    With two identical graphs and correct seeds, propagation should recover
    almost everything. If it cannot solve this, it cannot solve anything, and
    a failure here points at the algorithm rather than at noise.
    """
    ok = True
    G = nx.gnm_random_graph(800, 6000, seed=11)
    Ga = nx.relabel_nodes(G, {n: f"a{n}" for n in G.nodes()})

    # G^u carries *different* ids for the same nodes. This matters: with one
    # graph paired against itself the G^a and G^u id spaces coincide, and a
    # mapping keyed by the wrong one still resolves, so a phi/psi mix-up passes
    # unnoticed. Disjoint ids make the two directions distinguishable, which is
    # what caught exactly that bug in `propagate`.
    Gu = nx.relabel_nodes(G, {n: f"u{n}" for n in G.nodes()})
    truth = {f"a{n}": f"u{n}" for n in G.nodes()}

    nodes = sorted(G.nodes(), key=lambda n: -G.degree(n))
    anchors = [(f"a{n}", f"u{n}") for n in nodes[:50]]
    candidates = {f"a{n}": [f"u{n}"] for n in G.nodes()}

    mapping = propagate(Ga, Gu, anchors, candidates, theta=0.5, verbose=verbose)
    res = evaluate(mapping, truth)

    if res["correct"] < 0.90 * Ga.number_of_nodes():
        print(f"  [FAIL] identical graphs: only {res['correct']}/"
              f"{Ga.number_of_nodes()} recovered", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] identical graphs: {res['correct']}/"
              f"{Ga.number_of_nodes()} recovered (chi {res['chi']:.4f})",
              file=sys.stderr)

    if res["gamma"] < 0.99:
        print(f"  [FAIL] precision {res['gamma']:.4f} on identical graphs",
              file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] precision {res['gamma']:.4f}", file=sys.stderr)

    # Wrong seeds must not produce a correct mapping -- this catches a scoring
    # bug that ignores the anchors and matches on node id.
    bad_anchors = [(f"a{a}", f"u{b}")
                   for a, b in zip(nodes[:50], nodes[50:100])]
    bad = propagate(Ga, Gu, bad_anchors, candidates, theta=0.5, verbose=False)
    bad_res = evaluate(bad, truth)
    if bad_res["correct"] > 0.20 * Ga.number_of_nodes():
        print(f"  [FAIL] wrong seeds still recovered {bad_res['correct']} nodes "
              f"-- ground truth is leaking", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] wrong seeds recover little ({bad_res['correct']} nodes)",
              file=sys.stderr)

    # Eccentricity arithmetic, both readings.
    s = {"a": 3.0, "b": 1.0}
    e_all = eccentricity(s, universe=100, support="all")
    e_nz = eccentricity(s, universe=100, support="nonzero")
    if not (e_all > e_nz > 0):
        print(f"  [FAIL] eccentricity readings inconsistent: all={e_all:.3f} "
              f"nonzero={e_nz:.3f}", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] eccentricity: all-entries {e_all:.2f} > non-zero "
              f"{e_nz:.2f}, as expected", file=sys.stderr)

    print("all checks passed" if ok else "CHECKS FAILED", file=sys.stderr)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Adv-GD Phase 3: propagation.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    ap.error("nothing to do; Phase 3 runs from the driver, or use --self-test")


if __name__ == "__main__":
    raise SystemExit(main())
