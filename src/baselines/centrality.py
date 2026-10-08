"""Node centrality / cohesion measures, shared by two baselines:
`centrality_rank_match.py` (which ranks nodes by one measure at a time and
pairs same-rank nodes across graphs -- docs/some-context.txt section 2's own
"Index Matching" method) and `composite_centrality.py` (this repo's own
baseline, which uses several of these at once as feature dimensions).

Pure functions only: a graph in, a {node: score} dict out. No pair-loading,
no matching, no CLI -- that belongs in the files that use these.

    degree_scores        nx.degree, as a dict
    pagerank_scores       nx.pagerank
    eigenvector_scores     nx.eigenvector_centrality, falling back to the
                          numpy eigensolver if power iteration does not
                          converge (can happen on graphs with a near-
                          degenerate dominant eigenvalue)
    k_core_scores         nx.core_number -- already a per-node decomposition,
                          nothing to implement
    k_truss_scores        per-node truss number -- NOT built into networkx
                          (its own `nx.k_truss(G, k)` returns a subgraph for
                          one k, not a full decomposition), so built here by
                          calling that primitive for increasing k.

Node2Vec, which docs/some-context.txt's own table also measured, is
deliberately not included: it scored worst by a wide margin in that table
(0.01% vs. 0.10-0.83% for everything else) and pulls in a heavy extra
dependency (random-walk sampling + word2vec) for a method that is not
competitive even among this file's own weak floor measures.
"""
from __future__ import annotations

import networkx as nx

__all__ = ["degree_scores", "pagerank_scores", "eigenvector_scores",
           "k_core_scores", "k_truss_scores", "MEASURES"]


def degree_scores(G: nx.Graph) -> dict:
    return dict(G.degree())


def pagerank_scores(G: nx.Graph, alpha: float = 0.85) -> dict:
    return nx.pagerank(G, alpha=alpha)


def eigenvector_scores(G: nx.Graph, max_iter: int = 1000) -> dict:
    try:
        return nx.eigenvector_centrality(G, max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        return nx.eigenvector_centrality_numpy(G)


def k_core_scores(G: nx.Graph) -> dict:
    return nx.core_number(G)


def k_truss_scores(G: nx.Graph) -> dict:
    """Per-node truss number: the largest k such that the node has at least
    one incident edge in the k-truss subgraph.

    Delegates the actual k-truss computation to `networkx.k_truss(G, k)`
    (the correctly-handled, already-verified implementation) rather than
    hand-rolling the support-peeling decomposition: a first attempt at that
    peeling by hand got K4 wrong (gave trussness 4/4/3/2 instead of 4/4/4/4)
    because, exactly like k-core peeling, edges tied at the SAME support
    level must not decrement each other -- only a neighbor still at a
    strictly higher level may be decremented, which is subtle enough to get
    wrong (and would've gotten it wrong silently without the K4 self-test
    below) that reusing networkx's own primitive is the safer choice here.

    k-truss subgraphs are nested (the (k+1)-truss is always a subgraph of
    the k-truss), so each iteration restricts the PREVIOUS level's subgraph
    rather than recomputing from all of G -- the search naturally shrinks as
    k increases instead of repeating full-graph work every time.
    """
    node_score = {n: 0 for n in G.nodes()}
    H, k = G, 2
    while H.number_of_edges() > 0:
        H = nx.k_truss(H, k)
        if H.number_of_edges() == 0:
            break
        for n in H.nodes():
            node_score[n] = k
        k += 1
    return node_score


MEASURES = {
    "degree": degree_scores,
    "pagerank": pagerank_scores,
    "eigenvector": eigenvector_scores,
    "k_core": k_core_scores,
    "k_truss": k_truss_scores,
}


def self_test(verbose: bool = True) -> bool:
    import sys
    ok = True

    # K4: every edge sits in exactly 2 triangles (the other two edges of
    # whichever third node), and K4 is the textbook example of a maximal
    # 4-truss -- every edge, hence every node, should score trussness 4.
    K4 = nx.complete_graph(4)
    scores = k_truss_scores(K4)
    if any(v != 4 for v in scores.values()):
        print(f"  [FAIL] K4 node truss numbers should all be 4, got {scores}",
              file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] K4 truss numbers all 4: {scores}", file=sys.stderr)

    # A single triangle plus a pendant edge: the triangle's 3 edges each have
    # 1 common neighbour (support 1, trussness 3); the pendant edge has no
    # triangle at all (support 0, trussness 2). Node "pendant" should score 2
    # (its only edge), node "hub" (shared by the triangle and the pendant)
    # should score 3 (its best edge is a triangle edge).
    G = nx.Graph([(0, 1), (1, 2), (2, 0), (0, "pendant")])
    scores = k_truss_scores(G)
    expected = {0: 3, 1: 3, 2: 3, "pendant": 2}
    if scores != expected:
        print(f"  [FAIL] triangle+pendant truss numbers: got {scores}, "
              f"expected {expected}", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] triangle+pendant truss numbers correct: {scores}",
              file=sys.stderr)

    # k-core sanity: networkx's own function, just confirm the import path works.
    core = k_core_scores(nx.karate_club_graph())
    if not core or min(core.values()) < 0:
        print("  [FAIL] k_core_scores returned something degenerate",
              file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] k_core_scores on karate club: max core "
              f"{max(core.values())}", file=sys.stderr)

    print("all checks passed" if ok else "CHECKS FAILED", file=sys.stderr)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
