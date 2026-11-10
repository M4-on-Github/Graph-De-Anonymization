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
    python src/baselines/composite_centrality.py --pair runs/sweep_nominal/cit-HepTh_b10_s0 --dataset hepth
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATASETS, RunRecord, capture_stderr,  # noqa: E402
                    evaluate_anchors, load_pair, mutual_anchors, propagate,
                    evaluate, top_degree, TOP10, csls_matrix, DEFAULTS, REPO_ROOT)
from centrality import k_core_scores, pagerank_scores, eigenvector_scores  # noqa: E402

KIND = "ours_composite_centrality"

# Column order of the fingerprint. Part of the on-disk cache key, so that a
# cache written for a different feature list is never read back as this one.
FEATURES = ("degree", "clustering", "egonet_size", "k_core", "pagerank",
            "eigenvector")
# Bump when anything that decides the fingerprint's values changes (feature
# definitions, z-scoring, dtype). Also part of the cache key.
FINGERPRINT_VERSION = 1

# Row chunk of the CSLS passes. Must stay equal to `csls_matrix`'s default:
# `_csls_top10` reproduces csls_matrix(...) row by row, and keeping the same
# chunk keeps every matmul the same shape as the one csls_matrix issues.
CSLS_CHUNK = 2048


def _clustering_and_egonet(G: nx.Graph) -> tuple[dict, dict]:
    """(nx.clustering(G), {n: nx.ego_graph(G, n, radius=1).number_of_nodes()}),
    computed without nx's per-node set copies and without building a subgraph
    per node. The values are the same Python objects' values, not
    approximations:

    clustering. networkx 3.7's unweighted undirected `clustering` is
        {v: 0 if t == 0 else t / (d * (d - 1))}
    with, from `_triangles_and_degree_iter`, vs = set(G[v]) - {v}, d = len(vs)
    and t = sum over w in vs of len(vs & (set(G[w]) - {w})). It builds
    set(G[w]) afresh for every (v, w) pair, i.e. sum_w deg(w)^2 element copies,
    which is what costs ~2 h per gplus graph (hub degree 20,127). Here each
    S(x) = set(G[x]) - {x} is built once, and t is accumulated per edge: every
    w in S(v) is exactly one non-loop edge {v, w}, and that edge contributes
    len(S(v) & S(w)) to both t[v] and t[w]. So t[v] is the same integer as
    nx's, d is the same integer, and the final division is the same
    expression on the same Python ints (int / int true division, correctly
    rounded), so the floats are bit-identical. A node with t == 0 gets the
    int 0, as in nx.

    egonet size. For an undirected graph, nx.ego_graph(G, n, radius=1) is the
    subgraph induced on the nodes at distance <= 1 from n, i.e. {n} | S(n), so
    its number_of_nodes() is len(S(n)) + 1. On the pairs `load_pair` reads
    (simple graphs; `perturbation.load_parent` drops self-loops) that equals
    deg(n) + 1, but len(S(n)) + 1 is used rather than deg + 1 because it is
    also exact if a self-loop were ever present (a self-loop adds 2 to deg
    and nothing to the ego-net).

    Directed graphs and multigraphs fall back to the networkx calls verbatim.
    """
    if G.is_directed() or G.is_multigraph():
        return (nx.clustering(G),
                {n: nx.ego_graph(G, n, radius=1).number_of_nodes() for n in G.nodes()})
    adj = G._adj
    S = {v: set(nbrs) - {v} for v, nbrs in adj.items()}
    t = dict.fromkeys(adj, 0)
    for u, v in G.edges():
        if u == v:
            continue
        c = len(S[u] & S[v])
        t[u] += c
        t[v] += c
    clu, ego = {}, {}
    for v in adj:
        d = len(S[v])
        tv = t[v]
        clu[v] = 0 if tv == 0 else tv / (d * (d - 1))
        ego[v] = d + 1
    return clu, ego


def fingerprint(G: nx.Graph, nodes: list, verbose: bool = False,
                label: str = "") -> np.ndarray:
    """The six-feature vector per node, z-scored within this graph. Same
    z-scoring rationale as `structural_fingerprint.fingerprint`: the six
    columns run on very different scales, and normalising within one graph
    at a time (never jointly across both) keeps the baseline from secretly
    peeking at both graphs' statistics to decide how to scale.

    With `verbose`, prints one timing line per feature to stderr.
    """
    def tick(name, t0):
        if verbose:
            print(f"    fingerprint{' ' + label if label else ''}: {name} "
                  f"{time.time() - t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    deg = dict(G.degree())
    tick("degree", t0)
    t0 = time.time()
    clu, ego = _clustering_and_egonet(G)
    tick("clustering + egonet size", t0)
    t0 = time.time()
    core = k_core_scores(G)
    tick("k_core", t0)
    t0 = time.time()
    pr = pagerank_scores(G)
    tick("pagerank", t0)
    t0 = time.time()
    eig = eigenvector_scores(G)
    tick("eigenvector", t0)

    X = np.array([[deg[n], clu[n], ego[n], core[n], pr[n], eig[n]]
                 for n in nodes], dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _fingerprint_meta(G: nx.Graph, nodes: list, edgelist: Path) -> dict:
    """Everything the cached fingerprint's values depend on."""
    import scipy
    return {
        "kind": "composite_centrality_fingerprint",
        "version": FINGERPRINT_VERSION,
        "features": list(FEATURES),
        "dtype": "float32",
        "edgelist": edgelist.name,
        "edgelist_sha256": _sha256_file(edgelist),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        # The z-scoring runs over `nodes` in this order, so the order is part
        # of the key, not just the set.
        "nodes_sha256": hashlib.sha256(
            "\n".join(map(str, nodes)).encode()).hexdigest(),
        "networkx": nx.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def fingerprint_cache_path(edgelist: Path) -> Path:
    return edgelist.with_name(f"composite_fingerprint_{edgelist.stem}.npz")


def _load_fingerprint(path: Path, meta: dict, n: int,
                      quiet: bool = False) -> np.ndarray | None:
    """The cached F at `path` if its stored key equals `meta`, else None."""
    def say(msg):
        if not quiet:
            print(f"  {path.name} {msg}; recomputing", file=sys.stderr)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as blob:
            cached = json.loads(str(blob["meta"]))
            F = blob["F"]
    except Exception as exc:  # unreadable or truncated
        say(f"is unreadable ({type(exc).__name__})")
        return None
    if cached != meta:
        diff = sorted(k for k in set(cached) | set(meta)
                      if cached.get(k) != meta.get(k))
        say(f"does not match this graph/code ({', '.join(diff)} differ)")
        return None
    if F.shape != (n, len(FEATURES)) or F.dtype != np.float32:
        say(f"has shape {F.shape} {F.dtype}")
        return None
    return F


def fingerprint_cached(G: nx.Graph, nodes: list, edgelist: Path,
                       verbose: bool = True, label: str = "") -> np.ndarray:
    """`fingerprint(G, nodes)`, read from `<pair>/composite_fingerprint_<G>.npz`
    when that file was written for the same edge list (sha256), node order,
    graph size, feature list, FINGERPRINT_VERSION and networkx / numpy / scipy
    versions; otherwise computed and written there. The key lives inside the
    .npz (one atomically replaced file); a .json beside it repeats it for
    people.

    Tasks sharing a pair directory serialize on `<cache>.lock`, the same
    NFS-safe lock and write-then-rename `adv_gd.embed_cached` uses, so the
    W-control arms and a rerun wait for one computation instead of each
    repeating it.
    """
    from adv_gd import _cache_lock, _revalidate_dir
    path = fingerprint_cache_path(edgelist)
    t0 = time.time()
    meta = _fingerprint_meta(G, nodes, edgelist)
    F = _load_fingerprint(path, meta, len(nodes))
    if F is None:
        with _cache_lock(path):
            _revalidate_dir(path.parent)
            F = _load_fingerprint(path, meta, len(nodes), quiet=True)
            if F is None:
                F = fingerprint(G, nodes, verbose=verbose, label=label)
                tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp.npz")
                with open(tmp, "wb") as fh:
                    np.savez(fh, F=F, meta=np.array(json.dumps(meta, sort_keys=True)))
                os.replace(tmp, path)
                side = path.with_suffix(".json")
                stmp = side.with_name(f"{side.stem}.{uuid.uuid4().hex}.tmp.json")
                stmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
                os.replace(stmp, side)
                if verbose:
                    print(f"    fingerprint{' ' + label if label else ''}: "
                          f"wrote {path.name} ({time.time() - t0:.1f}s)",
                          file=sys.stderr)
                return F
    if verbose:
        print(f"    fingerprint{' ' + label if label else ''}: reusing "
              f"{path.name} ({time.time() - t0:.1f}s)", file=sys.stderr)
    return F


def _csls_top10(src, tgt, k: int, top: int = TOP10, chunk: int = CSLS_CHUNK):
    """candidate_sets(csls_matrix(src, tgt, k), top), without the
    len(src) x len(tgt) matrix (50,000 x 107,002 float32 = 21 GB on gplus).

    The r_t and r_s passes are `alignment.csls_matrix`'s, line for line. The
    last pass evaluates the same expression csls_matrix writes into `out` for
    the same row chunk, then takes `topk` of that chunk at once, the same call
    `candidate_sets` makes on the full matrix. topk works per row, and a row's
    values are the same floats here as in csls_matrix's `out`, so the indices
    (ties included) are the same.
    """
    import torch
    from alignment import _unit
    src, tgt = _unit(src), _unit(tgt)

    r_t = torch.empty(src.shape[0], device=src.device)
    for lo in range(0, src.shape[0], chunk):
        sims = src[lo:lo + chunk] @ tgt.t()
        r_t[lo:lo + chunk] = sims.topk(min(k, sims.shape[1]), dim=1).values.mean(1)

    r_s = torch.empty(tgt.shape[0], device=tgt.device)
    for lo in range(0, tgt.shape[0], chunk):
        sims = tgt[lo:lo + chunk] @ src.t()
        r_s[lo:lo + chunk] = sims.topk(min(k, sims.shape[1]), dim=1).values.mean(1)

    width = min(top, tgt.shape[0])
    idx = torch.empty(src.shape[0], width, dtype=torch.long, device=src.device)
    for lo in range(0, src.shape[0], chunk):
        rows = (2 * (src[lo:lo + chunk] @ tgt.t())
                - r_t[lo:lo + chunk, None] - r_s[None, :])
        idx[lo:lo + chunk] = rows.topk(width, dim=1).indices
    return idx


def match(Fa: np.ndarray, nodes_a: list, Fu: np.ndarray, nodes_u: list,
         Ga: nx.Graph, Gu: nx.Graph, n_anch: int, m: int, csls_k: int,
         verbose: bool = True) -> dict:
    """Anchors (mutual CSLS nearest neighbour among top-N_anch degree nodes)
    and candidates (top-10 CSLS matches for the top-M degree nodes), same
    shape `structural_fingerprint.match` returns, substituting CSLS for
    cosine. The candidate step goes through `_csls_top10`, which gives the
    same indices as candidate_sets(csls_matrix(...)) in O(chunk x n) memory.
    """
    import torch
    t0 = time.time()
    anch_a = top_degree(Ga, nodes_a, n_anch)
    anch_u = top_degree(Gu, nodes_u, n_anch)
    s_anch = csls_matrix(torch.as_tensor(Fa[anch_a]), torch.as_tensor(Fu[anch_u]), csls_k)
    pairs = mutual_anchors(s_anch)
    anchors = [(nodes_a[anch_a[i]], nodes_u[anch_u[j]]) for i, j in pairs]
    if verbose:
        print(f"  anchors: {len(anchors):,} mutual CSLS nearest neighbours "
              f"(k={csls_k}) among top {len(anch_a):,} degree nodes", file=sys.stderr)
        print(f"    anchor CSLS {len(anch_a):,} x {len(anch_u):,}: "
              f"{time.time() - t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    cand_a = top_degree(Ga, nodes_a, m)
    top = _csls_top10(torch.as_tensor(Fa[cand_a]), torch.as_tensor(Fu), csls_k,
                      TOP10).numpy()
    candidates = {nodes_a[cand_a[i]]: [nodes_u[j] for j in top[i]]
                 for i in range(len(cand_a))}
    if verbose:
        print(f"    candidate CSLS top-{TOP10} {len(cand_a):,} x {len(nodes_u):,}: "
              f"{time.time() - t0:.1f}s", file=sys.stderr)
    return {"anchors": anchors, "candidates": candidates}


def find_anchors(Ga: nx.Graph, Gu: nx.Graph, n_anch: int,
                 csls_k: int = DEFAULTS["csls_k"], nodes_a: list | None = None,
                 nodes_u: list | None = None, Fa: np.ndarray | None = None,
                 Fu: np.ndarray | None = None, verbose: bool = True,
                 pair: Path | None = None) -> list:
    """The anchor list `match` returns, without its candidate sets and without
    `propagate`.

    The first half of `match` on the same inputs: mutual CSLS nearest
    neighbours of the six-feature fingerprints among the top-`n_anch` degree
    nodes of each graph, as [(node in Ga, node in Gu)]. Nothing in the
    fingerprint or the match is random, so the list is a function of the two
    graphs alone. A caller that wants these anchors as an input to something
    else (the E3 W-control arms fit a Procrustes W to them) can have them
    without a propagate run and without this baseline's run record. This
    function writes no record and reads no ground truth.

    `nodes_a` / `nodes_u` default to the sorted node ids `run` uses. Pass `Fa`
    / `Fu` to reuse fingerprints already computed for these node lists. Pass
    `pair` (the directory Ga / Gu were loaded from as G1 / G2) to read and
    write the on-disk fingerprint cache `run` uses (see `fingerprint_cached`).
    """
    import torch
    nodes_a = sorted(Ga.nodes()) if nodes_a is None else nodes_a
    nodes_u = sorted(Gu.nodes()) if nodes_u is None else nodes_u
    if pair is not None:
        Fa = fingerprint_cached(Ga, nodes_a, Path(pair) / "G1.edgelist",
                                verbose=verbose, label="G1") if Fa is None else Fa
        Fu = fingerprint_cached(Gu, nodes_u, Path(pair) / "G2.edgelist",
                                verbose=verbose, label="G2") if Fu is None else Fu
    Fa = fingerprint(Ga, nodes_a, verbose=verbose, label="G1") if Fa is None else Fa
    Fu = fingerprint(Gu, nodes_u, verbose=verbose, label="G2") if Fu is None else Fu
    anch_a = top_degree(Ga, nodes_a, n_anch)
    anch_u = top_degree(Gu, nodes_u, n_anch)
    s_anch = csls_matrix(torch.as_tensor(Fa[anch_a]), torch.as_tensor(Fu[anch_u]), csls_k)
    pairs = mutual_anchors(s_anch)
    anchors = [(nodes_a[anch_a[i]], nodes_u[anch_u[j]]) for i, j in pairs]
    if verbose:
        print(f"  anchors: {len(anchors):,} mutual CSLS nearest neighbours "
              f"(k={csls_k}) among top {len(anch_a):,} degree nodes", file=sys.stderr)
    return anchors


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
            Fa = fingerprint_cached(Ga, nodes_a, pair / "G1.edgelist",
                                    verbose=verbose, label="G1")
            Fu = fingerprint_cached(Gu, nodes_u, pair / "G2.edgelist",
                                    verbose=verbose, label="G2")
            m = match(Fa, nodes_a, Fu, nodes_u, Ga, Gu,
                     n_anch=params["n_anch"], m=params["m"], csls_k=csls_k,
                     verbose=verbose)
            score = evaluate_anchors(m["anchors"], truth)
            if verbose:
                print(f"    fingerprints + anchors + candidates: "
                      f"{time.time() - t0:.0f}s", file=sys.stderr)

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

    # `find_anchors` duplicates the first half of `match` so that it can run
    # without the candidate sets; this is what keeps the two from drifting.
    alone = find_anchors(G, Gp, n_anch=300, csls_k=DEFAULTS["csls_k"],
                         nodes_a=nodes_a, nodes_u=nodes_u, Fa=Fa, Fu=Fu,
                         verbose=False)
    if alone != m["anchors"]:
        print(f"  [FAIL] find_anchors returned {len(alone)} anchors that differ "
              f"from match()'s {len(m['anchors'])}", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] find_anchors == match()['anchors'] "
              f"({len(alone)} anchors, same order)", file=sys.stderr)

    # The fast clustering / egonet sizes must be nx's own values, bit for bit,
    # including on a self-loop and on isolated nodes (`_clustering_and_egonet`
    # docstring says why they are).
    import torch
    from alignment import candidate_sets
    H = nx.powerlaw_cluster_graph(400, 5, 0.6, seed=1)
    H.add_edges_from([(3, 3), (7, 7)])
    H.add_nodes_from([1000, 1001])
    H = nx.relabel_nodes(H, {n: str(n) for n in H.nodes()})
    bad = []
    for name, g in (("gnm", G), ("powerlaw+selfloops+isolated", H),
                    ("K6", nx.complete_graph(6)), ("star", nx.star_graph(30))):
        clu, ego = _clustering_and_egonet(g)
        ref_ego = {n: nx.ego_graph(g, n, radius=1).number_of_nodes() for n in g}
        if clu != nx.clustering(g) or ego != ref_ego:
            bad.append(name)
    if bad:
        print(f"  [FAIL] fast clustering/egonet differ from networkx on {bad}",
              file=sys.stderr)
        ok = False
    else:
        print("  [PASS] fast clustering and egonet size == networkx exactly "
              "(4 graphs, incl. self-loops and isolated nodes)", file=sys.stderr)

    # The chunked candidate step must give candidate_sets(csls_matrix(...))'s
    # indices exactly, also with heavy ties (rounded features) and with more
    # rows than one chunk.
    gen = torch.Generator().manual_seed(0)
    bad = []
    for ns, nt, rnd in ((300, 300, None), (5000, 700, 1), (50, 7, None)):
        src = torch.randn(ns, 6, generator=gen)
        tgt = torch.randn(nt, 6, generator=gen)
        if rnd is not None:
            src, tgt = src.round(decimals=rnd), tgt.round(decimals=rnd)
        want = candidate_sets(csls_matrix(src, tgt, 10), TOP10)
        if not torch.equal(_csls_top10(src, tgt, 10, TOP10), want):
            bad.append((ns, nt))
    if bad:
        print(f"  [FAIL] _csls_top10 differs from candidate_sets(csls_matrix) "
              f"for shapes {bad}", file=sys.stderr)
        ok = False
    else:
        print("  [PASS] _csls_top10 == candidate_sets(csls_matrix(...)) "
              "(3 shapes, incl. ties and > 1 chunk)", file=sys.stderr)

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

