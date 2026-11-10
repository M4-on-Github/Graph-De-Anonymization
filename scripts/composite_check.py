#!/usr/bin/env python3
"""Old vs new `composite_centrality`: fingerprints, anchors and candidate sets must be equal.

The new module (src/baselines/composite_centrality.py) computes clustering and egonet size
without nx's per-node set copies / ego subgraphs, caches fingerprints on disk, and takes
the CSLS top-10 candidates chunk by chunk instead of building the m x n CSLS matrix. This
script runs the OLD code and the NEW code on the same inputs and checks, exactly:
  - fingerprints: np.array_equal (and same shape/dtype), per graph;
  - fast clustering / egonet size == nx.clustering / nx.ego_graph sizes (dict equality);
  - the on-disk fingerprint cache returns the same array (written, then reused);
  - match(): anchor list (same order) and candidate dict (same keys, same key order,
    same lists);
  - find_anchors() == match()['anchors'], old and new.
It prints the wall time of every stage, and exits 1 on any difference, 2 on setup error.

OLD code: by default the verbatim copy of the pre-change fingerprint/match/find_anchors
embedded below (git 3e8f37f, src/baselines/composite_centrality.py lines 58-138); with
--old PATH, that file is imported instead (e.g. a saved copy of the old module).

Usage (in the container, from the repo root, on a compute node):
    python3 scripts/composite_check.py --pair runs/sweep_nominal/cit-HepTh_b10_s0 \\
        --dataset hepth --record 20261019T0414Z_cit-HepTh_b10_s0_4654b45c
    python3 scripts/composite_check.py --pair ... --dataset hepth --old /path/to/old.py
    python3 scripts/composite_check.py --synthetic
    python3 scripts/composite_check.py --pair runs/pilot_gplus/gplus_combined_b10_s3 \\
        --dataset gplus --new-only          # time the new path only (old takes hours)
    python3 scripts/composite_check.py --records OLD_RUN_ID NEW_RUN_ID
        # compare two ours_composite_centrality records (anchors, chi, mapping sha)

--record RUN_ID also checks anchors_found / anchors_correct against that record (and that
the pair's G1/G2 sha256 match the record's inputs). --ref-anchors FILE compares the anchor
list with a "<a> <u>" per line file (an arm-B seed_anchors_adv_gd_w_control_*.txt).
--skip-cache leaves the pair directory untouched (no composite_fingerprint_G*.npz written).
"""
import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDS_DIR = REPO_ROOT / "results" / "runs"
sys.path.insert(0, str(REPO_ROOT / "src" / "baselines"))
sys.path.insert(0, str(REPO_ROOT / "src" / "adv_gd"))

import networkx as nx  # noqa: E402
import numpy as np     # noqa: E402

from common import (DATASETS, load_pair, mutual_anchors, candidate_sets,  # noqa: E402
                    TOP10, csls_matrix, DEFAULTS, top_degree, evaluate_anchors)
from centrality import k_core_scores, pagerank_scores, eigenvector_scores  # noqa: E402
import composite_centrality as new  # noqa: E402


# ---------------------------------------------------------------------------------------
# OLD code, verbatim from git 3e8f37f src/baselines/composite_centrality.py (58-138)
# ---------------------------------------------------------------------------------------

class _EmbeddedOld:
    @staticmethod
    def fingerprint(G: nx.Graph, nodes: list) -> np.ndarray:
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

    @staticmethod
    def match(Fa, nodes_a, Fu, nodes_u, Ga, Gu, n_anch, m, csls_k, verbose=True):
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

    @staticmethod
    def find_anchors(Ga, Gu, n_anch, csls_k=DEFAULTS["csls_k"], nodes_a=None,
                     nodes_u=None, Fa=None, Fu=None, verbose=True):
        import torch
        nodes_a = sorted(Ga.nodes()) if nodes_a is None else nodes_a
        nodes_u = sorted(Gu.nodes()) if nodes_u is None else nodes_u
        Fa = _EmbeddedOld.fingerprint(Ga, nodes_a) if Fa is None else Fa
        Fu = _EmbeddedOld.fingerprint(Gu, nodes_u) if Fu is None else Fu
        anch_a = top_degree(Ga, nodes_a, n_anch)
        anch_u = top_degree(Gu, nodes_u, n_anch)
        s_anch = csls_matrix(torch.as_tensor(Fa[anch_a]), torch.as_tensor(Fu[anch_u]), csls_k)
        pairs = mutual_anchors(s_anch)
        anchors = [(nodes_a[anch_a[i]], nodes_u[anch_u[j]]) for i, j in pairs]
        if verbose:
            print(f"  anchors: {len(anchors):,} mutual CSLS nearest neighbours "
                  f"(k={csls_k}) among top {len(anch_a):,} degree nodes", file=sys.stderr)
        return anchors


# ---------------------------------------------------------------------------------------

FAILS: list = []
TIMES: list = []


def say(msg):
    print(msg, file=sys.stderr, flush=True)


def check(name: str, ok: bool, detail: str = ""):
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def timed(label, fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    dt = time.time() - t0
    TIMES.append((label, dt))
    say(f"  [time] {label}: {dt:.1f}s")
    return out


def load_old(path):
    if path is None:
        say("old code: embedded verbatim copy (git 3e8f37f)")
        return _EmbeddedOld
    spec = importlib.util.spec_from_file_location("composite_centrality_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    say(f"old code: {path}")
    return mod


def compare_fingerprints(tag, Fo, Fn):
    same = (Fo.shape == Fn.shape and Fo.dtype == Fn.dtype and np.array_equal(Fo, Fn))
    detail = "" if same else (
        f"shape {Fo.shape}/{Fn.shape} dtype {Fo.dtype}/{Fn.dtype}"
        + (f" rows differing {int((Fo != Fn).any(1).sum())}, cols differing "
           f"{np.flatnonzero((Fo != Fn).any(0)).tolist()}" if Fo.shape == Fn.shape else ""))
    check(f"fingerprint {tag}: old == new (np.array_equal)", same, detail)


def compare_features(tag, G):
    clu, ego = timed(f"new clustering+egonet {tag}", new._clustering_and_egonet, G)
    ref_clu = timed(f"nx.clustering {tag}", nx.clustering, G)
    ref_ego = timed(f"nx.ego_graph sizes {tag}",
                    lambda: {n: nx.ego_graph(G, n, radius=1).number_of_nodes() for n in G})
    bad_c = [n for n in G if type(clu[n]) is not type(ref_clu[n]) or clu[n] != ref_clu[n]]
    bad_e = [n for n in G if ego[n] != ref_ego[n]]
    check(f"clustering {tag}: fast == nx.clustering (value and type, every node)",
          not bad_c, f"{len(bad_c)} nodes differ, e.g. {bad_c[:3]}" if bad_c else "")
    check(f"egonet size {tag}: fast == nx.ego_graph(...).number_of_nodes()",
          not bad_e, f"{len(bad_e)} nodes differ, e.g. {bad_e[:3]}" if bad_e else "")


def compare_match(mo, mn):
    check("match anchors: old == new (same list, same order)",
          mo["anchors"] == mn["anchors"],
          f"old {len(mo['anchors'])} new {len(mn['anchors'])}")
    co, cn = mo["candidates"], mn["candidates"]
    diff = [a for a in co if co[a] != cn.get(a)]
    check("match candidates: old == new (same keys in same order, same top-10 lists)",
          list(co) == list(cn) and not diff,
          f"{len(co)} keys; {len(diff)} rows differ" + (f", e.g. {diff[:3]}" if diff else ""))


def check_cache(tag, G, nodes, edgelist, Fref):
    path = new.fingerprint_cache_path(edgelist)
    existed = path.exists()
    F1 = timed(f"fingerprint_cached {tag} (1st call; file existed before: {existed})",
               new.fingerprint_cached, G, nodes, edgelist, True, tag)
    F2 = timed(f"fingerprint_cached {tag} (2nd call, must reuse)",
               new.fingerprint_cached, G, nodes, edgelist, True, tag)
    check(f"cache {tag}: 1st call == fresh fingerprint", np.array_equal(F1, Fref)
          and F1.dtype == Fref.dtype)
    check(f"cache {tag}: 2nd call == fresh fingerprint", np.array_equal(F2, Fref)
          and F2.dtype == Fref.dtype)
    check(f"cache {tag}: {path.name} and {path.with_suffix('.json').name} exist",
          path.exists() and path.with_suffix(".json").exists())


def run_pair_check(Ga, Gu, nodes_a, nodes_u, n_anch, m, csls_k, old, pair=None,
                   new_only=False, features=True, cache=True, truth=None):
    """Everything above on one (Ga, Gu). `pair` is a directory holding G1/G2.edgelist
    for the cache check (the real pair dir, or a temp dir for synthetic graphs)."""
    Fn_a = timed("new fingerprint G1", new.fingerprint, Ga, nodes_a, True, "G1")
    Fn_u = timed("new fingerprint G2", new.fingerprint, Gu, nodes_u, True, "G2")
    if not new_only:
        Fo_a = timed("old fingerprint G1", old.fingerprint, Ga, nodes_a)
        Fo_u = timed("old fingerprint G2", old.fingerprint, Gu, nodes_u)
        compare_fingerprints("G1", Fo_a, Fn_a)
        compare_fingerprints("G2", Fo_u, Fn_u)
        if features:
            compare_features("G1", Ga)
            compare_features("G2", Gu)
    if cache and pair is not None:
        check_cache("G1", Ga, nodes_a, Path(pair) / "G1.edgelist", Fn_a)
        check_cache("G2", Gu, nodes_u, Path(pair) / "G2.edgelist", Fn_u)

    mn = timed(f"new match (anchors top {n_anch}, candidates top {m})", new.match,
               Fn_a, nodes_a, Fn_u, nodes_u, Ga, Gu, n_anch, m, csls_k, True)
    fa_new = timed("new find_anchors" + (" (pair cache)" if cache and pair else ""),
                   new.find_anchors, Ga, Gu, n_anch, csls_k, nodes_a, nodes_u,
                   None if (cache and pair) else Fn_a,
                   None if (cache and pair) else Fn_u, True,
                   pair if cache else None)
    check("new find_anchors == new match anchors", fa_new == mn["anchors"])
    if not new_only:
        mo = timed("old match", old.match, Fo_a, nodes_a, Fo_u, nodes_u, Ga, Gu,
                   n_anch, m, csls_k, True)
        compare_match(mo, mn)
        fa_old = timed("old find_anchors (given old F)", old.find_anchors, Ga, Gu,
                       n_anch, csls_k, nodes_a, nodes_u, Fo_a, Fo_u, False)
        check("old find_anchors == new find_anchors", fa_old == fa_new)
    if truth is not None:
        s = evaluate_anchors(mn["anchors"], truth)
        say(f"  anchors {s['correct']}/{s['found']}  precision {s['precision']:.4f}")
        return mn, s
    return mn, None


# ---------------------------------------------------------------------------------------

def perturbed_copy(G, rng, drop=0.1):
    """Drop a fraction of edges and relabel by a random permutation (string ids)."""
    H = G.copy()
    edges = list(H.edges())
    k = int(drop * len(edges))
    for i in rng.choice(len(edges), size=k, replace=False):
        H.remove_edge(*edges[i])
    nodes = sorted(H.nodes())
    perm = rng.permutation(len(nodes))
    relabel = {n: f"u{nodes[p]}" for n, p in zip(nodes, perm)}
    return nx.relabel_nodes(H, relabel), {n: relabel[n] for n in nodes}


def synthetic(args, old) -> None:
    rng = np.random.default_rng(0)

    def strs(G):
        return nx.relabel_nodes(G, {n: str(n) for n in G})

    two_ba = nx.disjoint_union(nx.barabasi_albert_graph(300, 3, seed=2),
                               nx.barabasi_albert_graph(200, 2, seed=3))
    two_ba.add_nodes_from(range(500, 510))           # isolated nodes
    cases = [
        ("gnm 300/900", strs(nx.gnm_random_graph(300, 900, seed=0)), 300, 300),
        ("powerlaw_cluster 2000", strs(nx.powerlaw_cluster_graph(2000, 4, 0.5, seed=1)),
         200, 1000),
        ("random 4-regular 600 (ties)", strs(nx.random_regular_graph(4, 600, seed=4)),
         300, 600),
        ("two BA components + isolated", strs(two_ba), 200, 510),
        ("tiny (8 nodes, < TOP10)", strs(nx.cycle_graph(8)), 8, 8),
        ("star 50 + clique 12", strs(nx.disjoint_union(nx.star_graph(50),
                                                        nx.complete_graph(12))), 30, 63),
    ]
    tmp_root = Path(tempfile.mkdtemp(prefix="composite_check_"))
    try:
        for i, (name, G, n_anch, m) in enumerate(cases):
            say(f"\n=== synthetic: {name}  ({G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges) ===")
            Gu, truth = perturbed_copy(G, rng)
            d = tmp_root / f"case{i}"
            d.mkdir()
            nx.write_edgelist(G, d / "G1.edgelist", data=False)
            nx.write_edgelist(Gu, d / "G2.edgelist", data=False)
            # Read back the way load_pair reads, so node/adjacency order is the real one.
            Ga = nx.read_edgelist(d / "G1.edgelist", nodetype=str)
            Gb = nx.read_edgelist(d / "G2.edgelist", nodetype=str)
            Ga.add_nodes_from(n for n in G if n not in Ga)  # isolated nodes
            Gb.add_nodes_from(truth[n] for n in G if truth[n] not in Gb)
            run_pair_check(Ga, Gb, sorted(Ga), sorted(Gb), n_anch, m,
                           DEFAULTS["csls_k"], old, pair=d, truth=truth)

        # Cache invalidation: change G1.edgelist, the cache must be recomputed.
        say("\n=== synthetic: cache invalidation on edge-list change ===")
        d = tmp_root / "case1"
        Ga = nx.read_edgelist(d / "G1.edgelist", nodetype=str)
        stale = new.fingerprint_cached(Ga, sorted(Ga), d / "G1.edgelist", True, "G1")
        u, v = sorted(Ga)[:2]
        with open(d / "G1.edgelist", "a") as fh:
            fh.write(f"{u} {v}\n" if not Ga.has_edge(u, v) else f"{u} x_new\n")
        Gc = nx.read_edgelist(d / "G1.edgelist", nodetype=str)
        Fc = new.fingerprint_cached(Gc, sorted(Gc), d / "G1.edgelist", True, "G1")
        check("cache recomputed after edge-list change (== fresh, != stale)",
              np.array_equal(Fc, new.fingerprint(Gc, sorted(Gc)))
              and not (Fc.shape == stale.shape and np.array_equal(Fc, stale)))

        # Self-loops: the fast helper only (core_number refuses self-loops, so the
        # full fingerprint cannot run on such a graph, old or new).
        say("\n=== synthetic: self-loops (fast clustering / egonet helper only) ===")
        H = strs(nx.powerlaw_cluster_graph(500, 5, 0.6, seed=5))
        H.add_edges_from([("3", "3"), ("17", "17"), ("200", "200")])
        compare_features("self-loop graph", H)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def compare_records(old_id, new_id) -> None:
    ro = json.loads((RECORDS_DIR / old_id / "record.json").read_text())
    rn = json.loads((RECORDS_DIR / new_id / "record.json").read_text())
    for k in ("kind", "pair", "dataset", "beta", "csls_k", "n_anch", "m", "theta",
              "eccen_support", "phases"):
        check(f"record field {k}: {ro.get(k)!r} == {rn.get(k)!r}", ro.get(k) == rn.get(k))
    ins_o = {Path(i["path"]).name: i["sha256"] for i in ro["inputs"]}
    ins_n = {Path(i["path"]).name: i["sha256"] for i in rn["inputs"]}
    check("inputs sha256 equal (G1, G2, mapping, stats)", ins_o == ins_n)
    for k in ("anchors_found", "anchors_correct", "anchor_precision", "mapped",
              "correct", "chi", "gamma"):
        a, b = ro["metrics"].get(k), rn["metrics"].get(k)
        check(f"metric {k}: {a} == {b}", a == b)
    so = [o["sha256"] for o in ro.get("outputs", [])]
    sn = [o["sha256"] for o in rn.get("outputs", [])]
    check(f"mapping sha256: {so[:1]} == {sn[:1]}", so == sn)
    say(f"  seconds: old {ro.get('seconds')}  new {rn.get('seconds')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair", type=str)
    ap.add_argument("--dataset", choices=sorted(DATASETS))
    ap.add_argument("--csls-k", type=int, default=DEFAULTS["csls_k"])
    ap.add_argument("--old", type=str, default=None,
                    help="Path to the old composite_centrality.py (default: embedded copy).")
    ap.add_argument("--record", type=str, default=None,
                    help="ours_composite_centrality run_id on this pair to check "
                         "anchors_found/anchors_correct against.")
    ap.add_argument("--ref-anchors", type=str, default=None,
                    help='"<a> <u>" per line anchor file to compare with (arm-B seeds).')
    ap.add_argument("--new-only", action="store_true",
                    help="Time the new path only (no old code, no comparisons).")
    ap.add_argument("--skip-features", action="store_true",
                    help="Skip the per-feature nx.clustering / ego_graph comparison.")
    ap.add_argument("--skip-cache", action="store_true",
                    help="Do not read or write composite_fingerprint_G*.npz.")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--records", nargs=2, metavar=("OLD_RUN_ID", "NEW_RUN_ID"))
    args = ap.parse_args(argv)

    import torch
    say(f"torch {torch.__version__} threads {torch.get_num_threads()}  "
        f"networkx {nx.__version__}  numpy {np.__version__}")

    if args.records:
        compare_records(*args.records)
    elif args.synthetic:
        synthetic(args, load_old(args.old))
    elif args.pair and args.dataset:
        pair = Path(args.pair)
        pair = pair if pair.is_absolute() else REPO_ROOT / pair
        if not (pair / "G1.edgelist").exists():
            say(f"no G1.edgelist in {pair}")
            return 2
        old = None if args.new_only else load_old(args.old)
        params = DATASETS[args.dataset]
        Ga, Gu, truth, stats = timed("load_pair", load_pair, pair)
        say(f"pair {pair.name}: G1 {Ga.number_of_nodes():,}n {Ga.number_of_edges():,}e  "
            f"G2 {Gu.number_of_nodes():,}n {Gu.number_of_edges():,}e  "
            f"n_anch {params['n_anch']}  m {params['m']}")
        rec = None
        if args.record:
            rec = json.loads((RECORDS_DIR / args.record / "record.json").read_text())
            from provenance import sha256_file
            ins = {Path(i["path"]).name: i["sha256"] for i in rec["inputs"]}
            for f in ("G1.edgelist", "G2.edgelist", "mapping.txt"):
                check(f"{f} sha256 == record {args.record}'s", sha256_file(pair / f) == ins.get(f))
        mn, score = run_pair_check(Ga, Gu, sorted(Ga.nodes()), sorted(Gu.nodes()),
                                   params["n_anch"], params["m"], args.csls_k, old,
                                   pair=pair, new_only=args.new_only,
                                   features=not args.skip_features,
                                   cache=not args.skip_cache, truth=truth)
        if rec:
            for k in ("anchors_found", "anchors_correct"):
                want = rec["metrics"][k]
                got = score["found" if k == "anchors_found" else "correct"]
                check(f"{k} {got} == record's {want}", got == want)
        if args.ref_anchors:
            ref = [tuple(line.split()) for line in Path(args.ref_anchors).read_text().splitlines()
                   if line.strip()]
            check(f"anchors == {Path(args.ref_anchors).name} (same order)",
                  ref == [tuple(p) for p in mn["anchors"]],
                  f"{len(ref)} in file, {len(mn['anchors'])} computed")
    else:
        ap.error("give --pair and --dataset, or --synthetic, or --records")

    say("\ntimings:")
    for label, dt in TIMES:
        say(f"  {dt:10.1f}s  {label}")
    if FAILS:
        say(f"\nFAILED ({len(FAILS)}): " + "; ".join(FAILS))
        return 1
    say("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
