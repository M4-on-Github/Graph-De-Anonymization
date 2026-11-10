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


def inverse_sqrt_degrees(G: nx.Graph) -> dict:
    """node -> 1.0 / math.sqrt(degree), 0.0 for an isolated node.

    The same expression `get_scores` evaluates inline, so each value is the
    identical double; it is just computed once per node instead of once per
    visit.
    """
    return {n: (1.0 / math.sqrt(d) if d else 0.0) for n, d in G.degree()}


def get_scores(G1: nx.Graph, G2: nx.Graph, v1: str, phi: dict,
               mapped_targets: set, inv_sqrt_deg: dict | None = None) -> dict:
    """Algorithm 1: score every candidate in G2 against node v1 of G1.

    For each already-matched neighbour of v1, walk the neighbours of that
    neighbour's image in G2 and credit each unmatched one. The credit is
    1/sqrt(degree) so that high-degree nodes, which neighbour everything, do
    not win by default (paper section 3.2).

    `inv_sqrt_deg` is an optional precomputed `inverse_sqrt_degrees(G2)`; when
    omitted it is built here, so the result is the same either way. Neighbours
    are visited in the graphs' own adjacency order and credits are summed in
    that order, which fixes the floating-point result.
    """
    if inv_sqrt_deg is None:
        inv_sqrt_deg = inverse_sqrt_degrees(G2)
    adj1, adj2 = G1._adj, G2._adj          # same iteration order as G[v]
    scores: dict[str, float] = {}
    get = scores.get
    for nb in adj1[v1]:
        img = phi.get(nb)
        if img is None or img not in adj2:
            continue
        for cand in adj2[img]:
            if cand in mapped_targets:
                continue
            w = inv_sqrt_deg[cand]
            if w:
                scores[cand] = get(cand, 0.0) + w
    return scores


# Per-sweep count of forward evaluations (get_scores calls on G^a nodes) made
# by the most recent `propagate` call. Diagnostics only: nothing in the
# records reads it, and it does not affect the mapping.
LAST_EVALUATIONS: list = []


def propagate(Ga: nx.Graph, Gu: nx.Graph, anchors: list, candidates: dict,
              theta: float = 0.5, support: str = "all",
              max_sweeps: int = 100, verbose: bool = True) -> dict:
    """Algorithm 2: two-stage propagation from the Phase 2 anchors.

    Stage one (flag = 0) only accepts a match that the Phase 2 candidate set
    already proposed, which keeps early mistakes from compounding. Once that
    stalls, stage two (flag = 1) drops the restriction and lets propagation run
    on topology alone.

    Returns the mapping from G^a ids to G^u ids, anchors included.

    Skipping unchanged nodes (exact; the mapping, its insertion order, the
    number of sweeps and every sweep's "+N" are those of the full rescan).
    The algorithm is a sequential scan: every sweep visits Ga.nodes() in order
    and each evaluation sees the state left by the earlier ones. The state is
    phi, psi, mapped_a = set(phi), mapped_u = set(psi) and flag; the graphs,
    degrees, candidates, theta and support are fixed. The decision for an
    unmapped w is a function of
      (a) s1(w) = get_scores(Ga, Gu, w, phi, mapped_u): reads phi on N_a(w)
          and mapped_u on Cand1(w) = union of N_u(phi[nb]) over nb in
          N_a(w) with nb in phi; insertion order and float sums are fixed by
          those reads, so s1 (and hence ecc and argmax z) changes only if one
          of them changes;
      (b) if ecc(s1) >= theta with argmax z: s2(z) = get_scores(Gu, Ga, z,
          psi, mapped_a), which reads psi on N_u(z) and mapped_a on Cand2(z)
          = union of N_a(psi[y]) over y in N_u(z) with y in psi;
      (c) if v_back == w: the stage-one test, which reads flag only.
    Accepting (v_a, v_u) changes exactly phi[v_a], psi[v_u], v_a in mapped_a
    and v_u in mapped_u (v_u was not in mapped_u, which contains every value
    of phi, and v_a not in mapped_a, which contains every value of psi). So
    the outcome of w can change only if
      D1  v_a in N_a(w)                          (phi on N_a(w));
      D2  v_u in Cand1(w): w in N_a(a) for some a with phi[a] in N_u(v_u)
                                                 (mapped_u on Cand1(w));
      D3  w last reached (b) with argmax z and either v_u in N_u(z) (psi on
          N_u(z)) or v_a in Cand2(z): z in N_u(y) for some y with psi[y] in
          N_a(v_a)                               (mapped_a on Cand2(z)).
    D2 and D3 go through the inverse maps `pre` (x -> every a with phi[a] =
    x) and `qre` (b -> every y with psi[y] = b), not through psi / phi: with
    duplicated anchors phi and psi are not inverses of each other (pairs
    (a1,u1), (a2,u1) give phi[a1] = phi[a2] = u1 but psi[u1] = a2 only).
    N_a / N_u above are successor sets; the marks walk predecessors, which
    are the same sets for an undirected graph.
    `best_of[z]` holds the w whose last evaluation reached (b) with argmax z;
    w leaves it when re-evaluated, and the whole set is dropped once dirtied.
    v_u itself is added to the D3 set, which costs nothing and also covers
    every w whose argmax was v_u (D2 already does). The flag flip changes only
    (c), so it dirties exactly the w whose last evaluation failed at the
    stage-one test (`cand_fail`). Initially every node is dirty.
    Invariant: an unmapped node not in `dirty` would be rejected if evaluated
    on the current state, so skipping it is the same as the old rescan
    evaluating and rejecting it (a rejection changes no state). The sets are
    only tested for membership, so set iteration order never reaches phi.
    """
    # Drop any anchor naming a node that is not in both graphs, then build the
    # two directions from the surviving pairs. Deriving psi from the filtered
    # pair list rather than by inverting phi keeps the key spaces straight:
    # phi is keyed by G^a ids, psi by G^u ids, and the reverse-direction
    # get_scores below is silently useless if they are ever the same.
    kept = [(a, u) for a, u in anchors if a in Ga and u in Gu]
    phi = {a: u for a, u in kept}                  # G^a -> G^u
    psi = {u: a for a, u in kept}                  # G^u -> G^a

    # Mapped-target sets, kept in step with phi/psi below instead of copied
    # from them on every get_scores call (the copy was O(mapped) per node).
    mapped_u = set(psi)                            # G^u nodes already matched
    mapped_a = set(phi)                            # G^a nodes already matched
    inv_u = inverse_sqrt_degrees(Gu)
    inv_a = inverse_sqrt_degrees(Ga)

    # Dependency bookkeeping for skipping nodes (see the docstring). The
    # marks need "who has v among its neighbours", i.e. predecessors; for an
    # undirected graph that is the adjacency itself.
    adj_a = Ga._pred if Ga.is_directed() else Ga._adj
    adj_u = Gu._pred if Gu.is_directed() else Gu._adj
    pre: dict = {}                                 # G^u x -> [a : phi[a] = x]
    for a, u in phi.items():
        pre.setdefault(u, []).append(a)
    qre: dict = {}                                 # G^a b -> [y : psi[y] = b]
    for u, a in psi.items():
        qre.setdefault(a, []).append(u)
    dirty = set(Ga.nodes())                        # G^a nodes to evaluate
    best_of: dict = {}                             # G^u z -> {w reaching (b)}
    best_cur: dict = {}                            # G^a w -> its registered z
    cand_fail = set()                              # failed only the stage-1 test

    n_a, n_u = Gu.number_of_nodes(), Ga.number_of_nodes()
    flag = 0
    t0 = time.time()
    LAST_EVALUATIONS.clear()

    for sweep in range(1, max_sweeps + 1):
        added = 0
        n_eval = 0
        for v_a in list(Ga.nodes()):
            if v_a in phi:
                continue
            if v_a not in dirty:
                continue            # last evaluation rejected; inputs unchanged
            dirty.discard(v_a)
            n_eval += 1
            z_old = best_cur.pop(v_a, None)
            if z_old is not None:
                ws = best_of.get(z_old)
                if ws is not None:
                    ws.discard(v_a)
                    if not ws:
                        del best_of[z_old]
            cand_fail.discard(v_a)

            s1 = get_scores(Ga, Gu, v_a, phi, mapped_u, inv_u)
            if eccentricity(s1, n_a, support) < theta:
                continue
            v_u = max(s1.items(), key=lambda kv: kv[1])[0]
            best_cur[v_a] = v_u
            best_of.setdefault(v_u, set()).add(v_a)

            # Reverse check: v_u must pick v_a back, which is what stops two
            # G^a nodes from both claiming the same G^u node.
            s2 = get_scores(Gu, Ga, v_u, psi, mapped_a, inv_a)
            if eccentricity(s2, n_u, support) < theta:
                continue
            v_back = max(s2.items(), key=lambda kv: kv[1])[0]
            if v_back != v_a:
                continue

            # Stage one additionally requires the match to be in the candidate
            # set built by Phase 2 (Algorithm 2, line 11).
            if flag == 0 and v_u not in candidates.get(v_a, ()):
                cand_fail.add(v_a)
                continue

            phi[v_a] = v_u
            psi[v_u] = v_a
            mapped_u.add(v_u)
            mapped_a.add(v_a)
            added += 1
            pre.setdefault(v_u, []).append(v_a)
            qre.setdefault(v_a, []).append(v_u)

            # Mark every node whose outcome may have changed.
            dirty.update(adj_a[v_a])                            # D1
            for x in adj_u[v_u]:                                # D2
                for a in pre.get(x, ()):
                    dirty.update(adj_a[a])
            zs = set(adj_u[v_u])                                # D3
            zs.add(v_u)
            for b in adj_a[v_a]:
                for y in qre.get(b, ()):
                    zs.update(adj_u[y])
            for z in best_of.keys() & zs:
                dirty.update(best_of.pop(z))

        LAST_EVALUATIONS.append(n_eval)
        if verbose:
            print(f"  sweep {sweep:3d}  stage {flag}  +{added:,} "
                  f"-> {len(phi):,} mapped  {time.time() - t0:.0f}s",
                  file=sys.stderr)

        if added == 0:
            if flag == 0:
                flag = 1            # candidate restriction lifted
                dirty.update(cand_fail)
                cand_fail.clear()
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
