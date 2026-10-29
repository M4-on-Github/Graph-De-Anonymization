#!/usr/bin/env python3
"""E4a: characterise a parent graph, one RunRecord of kind `dataset_characterization` each.

The quantities are the rows of the characterisation table in experiment_log_03 (section
E4, "Dataset characterisation"), computed on the undirected simple graph that
`src/beta_noise_generator/perturbation.py:load_parent` builds (self-loops dropped,
duplicate and reciprocal lines absorbed), because that is the graph every pair is cut from:

  nodes / edges; self-loops dropped; connected components; giant component (size, share
  of nodes); second-largest component; degree mean / median / p90 / p99 / max; share of
  degree-1 nodes; global transitivity; maximum core number; nodes in a twin class and the
  twin-adjusted chi reference.

  Twin class: a set of two or more nodes with the same open neighbourhood N(u) = N(v)
  (necessarily non-adjacent) or the same closed neighbourhood N[u] = N[v] (necessarily
  adjacent). A node cannot be in an open and a closed class at once (see `twin_stats`), so
  "nodes in a twin class" is a plain count. Twin-adjusted chi reference at beta = 0:
  (giant-component nodes - sum over the twin classes inside the giant of (size - 1)) /
  all nodes -- the expected chi of an ideal structure-only method that assigns each twin
  class at random. A reference, not a bound.

  p90 and p99 are NEAREST-RANK percentiles (numpy method="inverted_cdf": the smallest
  degree d with at least q % of nodes at degree <= d). The table's integers need this:
  numpy's default linear interpolation gives 166.32 for HepTh's p99, the table says 167.

Added for Order step 2 and the proposal's midpoint ("centrality, and PageRank"): PageRank
(alpha 0.85, as src/baselines/centrality.py) and eigenvector centrality (power iteration,
max_iter 1000, numpy fallback, as centrality.py), each summarised by its maximum, the
share of the total held by the top 50 nodes (50 = the seed count of Report 1 and E0), its
Gini coefficient and its top 5 nodes; Spearman rank correlations with degree and with each
other. The spec names the two measures but defines no summary, so these are this script's
choice.

Outputs, per dataset NAME:
  <root>/NAME/characterization.json   every quantity, the degree histogram, top nodes
                                      (bulk; default root runs/char, not tracked)
  <archive>/<run_id>/record.json      the RunRecord (default results/runs, tracked), status
                                      exploratory unless --status confirmatory is passed

    python3 scripts/e4_characterize.py cit-HepTh=data/cit-HepTH/cit-HepTh.txt \\
        ca-GrQc=data/small_ca_grqc/ca-GrQc_Graph1_edgelist.txt
    python3 scripts/e4_characterize.py fb=data/fb/fb.edgelist     # after scripts/e4_convert_fb.py
    python3 scripts/e4_characterize.py --self-test

Run it in the project container (networkx, numpy and scipy).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "adv_gd"))
sys.path.insert(0, str(REPO_ROOT / "src" / "beta_noise_generator"))

from perturbation import load_parent, resolve_edgelist_path  # noqa: E402
from provenance import (ARCHIVE, RunRecord, capture_stderr,  # noqa: E402
                        describe_file)

KIND = "dataset_characterization"
SCHEMA = "dataset_characterization/1"
DEFAULT_ROOT = REPO_ROOT / "runs" / "char"
PERCENTILE_METHOD = "inverted_cdf"       # nearest rank; see the module docstring
TOP_K = 50                               # Report 1 / E0 seed count
PAGERANK_ALPHA = 0.85                    # src/baselines/centrality.py:41
EIGEN_MAX_ITER = 1000                    # src/baselines/centrality.py:45


# ---------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------

def scan_edgelist(path: Path) -> dict:
    """Line counts under `load_parent`'s own rules (strip; skip blank; '#'/'%' comments;
    split on whitespace; fewer than two tokens is malformed; u == v is a self-loop)."""
    src = resolve_edgelist_path(path)
    lines = blank = comments = malformed = data = loops = 0
    loop_ids: set[str] = set()
    edge_ids: set[str] = set()
    with src.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            lines += 1
            line = line.strip()
            if not line:
                blank += 1
                continue
            if line[0] in "#%":
                comments += 1
                continue
            parts = line.split()
            if len(parts) < 2:
                malformed += 1
                continue
            data += 1
            if parts[0] == parts[1]:
                loops += 1
                loop_ids.add(parts[0])
            else:
                edge_ids.update((parts[0], parts[1]))
    return {"lines": lines, "blank_lines": blank, "comment_lines": comments,
            "malformed_lines": malformed, "edge_lines": data,
            "self_loop_lines": loops,
            "nodes_only_in_self_loops": len(loop_ids - edge_ids),
            "nodes_with_a_non_loop_edge": len(edge_ids)}


# ---------------------------------------------------------------------------------------
# quantities
# ---------------------------------------------------------------------------------------

def twin_stats(G: nx.Graph, giant: set, n: int) -> dict:
    """Twin classes (open or closed neighbourhood) and the twin-adjusted chi reference.

    Open twins u, v (N(u) = N(v)) are never adjacent: u ~ v would put v in N(u) = N(v).
    Closed twins (N[u] = N[v]) are always adjacent. A node with an open twin v and a closed
    twin w would have w in N(u) = N(v), so v ~ w, so v in N[w] = N[u], so v ~ u: a
    contradiction. The two kinds of class are therefore disjoint, and the number of nodes
    in a twin class is the sum of the two counts.
    """
    by_open: dict = defaultdict(list)
    by_closed: dict = defaultdict(list)
    for u in G:
        nb = frozenset(G[u])
        by_open[nb].append(u)
        by_closed[nb | {u}].append(u)
    open_cls = [g for g in by_open.values() if len(g) > 1]
    closed_cls = [g for g in by_closed.values() if len(g) > 1]
    del by_open, by_closed
    open_nodes = {u for g in open_cls for u in g}
    closed_nodes = {u for g in closed_cls for u in g}
    if open_nodes & closed_nodes:
        raise RuntimeError("a node is in an open and a closed twin class; impossible in "
                           "a simple graph, so the graph is not simple")
    classes = open_cls + closed_cls
    # every member of a class has the same neighbour, so a class lies in one component
    in_giant = [g for g in classes if g[0] in giant]
    removed = sum(len(g) - 1 for g in in_giant)
    nodes = len(open_nodes) + len(closed_nodes)
    return {
        "twin_nodes": nodes, "twin_share": nodes / n,
        "open_twin_nodes": len(open_nodes), "closed_twin_nodes": len(closed_nodes),
        "open_twin_classes": len(open_cls), "closed_twin_classes": len(closed_cls),
        "largest_twin_class": max((len(g) for g in classes), default=0),
        "twin_nodes_in_giant": sum(len(g) for g in in_giant),
        "twin_classes_in_giant": len(in_giant),
        "twin_chi_reference": (len(giant) - removed) / n,
    }


def gini(values: np.ndarray) -> float:
    x = np.sort(np.clip(values, 0.0, None))
    total = x.sum()
    if total <= 0:
        return 0.0
    n = len(x)
    return float((n + 1 - 2 * np.cumsum(x).sum() / total) / n)


def centrality_summary(values: np.ndarray, nodes: list, deg: np.ndarray, k: int = TOP_K) -> dict:
    v = np.clip(values, 0.0, None)
    order = np.argsort(-v, kind="stable")
    total = float(v.sum())
    top = order[:k]
    return {
        "max": float(v[order[0]]),
        f"top{k}_share": float(v[top].sum() / total) if total > 0 else 0.0,
        "gini": gini(v),
        "nodes_below_1e-9_of_max": int((v < 1e-9 * v[order[0]]).sum()),
        "top5": [{"node": nodes[i], "value": float(v[i]), "degree": int(deg[i])}
                 for i in order[:5]],
    }


def characterize(G: nx.Graph, centrality: bool = True, log=print) -> tuple[dict, dict]:
    """Every quantity for one graph. Returns (result, seconds per step)."""
    secs: dict[str, float] = {}

    def timed(name, fn):
        t0 = time.time()
        out = fn()
        secs[name] = round(time.time() - t0, 2)
        log(f"  {name}: {secs[name]}s")
        return out

    n, m = G.number_of_nodes(), G.number_of_edges()
    nodes = list(G)
    deg = np.array([G.degree(u) for u in nodes], dtype=np.int64)

    comps = timed("components", lambda: sorted(nx.connected_components(G), key=len, reverse=True))
    sizes = [len(c) for c in comps]
    giant = comps[0]
    res: dict = {
        "nodes": n, "edges": m, "edges_per_node": m / n,
        "components": {"count": len(comps), "giant_nodes": sizes[0],
                       "giant_share": sizes[0] / n,
                       "second_largest": sizes[1] if len(sizes) > 1 else 0,
                       "largest_ten": sizes[:10]},
    }
    p90, p99 = (float(np.percentile(deg, q, method=PERCENTILE_METHOD)) for q in (90, 99))
    values, counts = np.unique(deg, return_counts=True)
    res["degree"] = {
        "mean": float(deg.mean()), "median": float(np.median(deg)),
        "p90": p90, "p99": p99, "max": int(deg.max()), "min": int(deg.min()),
        "share_degree_1": float((deg == 1).mean()),
        "percentile_method": PERCENTILE_METHOD,
        "p90_linear": float(np.percentile(deg, 90)), "p99_linear": float(np.percentile(deg, 99)),
        "histogram": {str(int(d)): int(c) for d, c in zip(values, counts)},
    }
    res["transitivity"] = timed("transitivity", lambda: float(nx.transitivity(G)))
    core = timed("core_number", lambda: nx.core_number(G))
    kmax = max(core.values())
    res["max_core"] = {"k": int(kmax), "nodes_in_max_core": sum(1 for c in core.values() if c == kmax)}
    res["twins"] = timed("twins", lambda: twin_stats(G, giant, n))

    if centrality:
        from scipy.stats import spearmanr
        pr = timed("pagerank", lambda: nx.pagerank(G, alpha=PAGERANK_ALPHA))
        try:
            ev = timed("eigenvector", lambda: nx.eigenvector_centrality(G, max_iter=EIGEN_MAX_ITER))
            solver = "power_iteration"
        except nx.PowerIterationFailedConvergence:
            ev = timed("eigenvector_numpy_fallback", lambda: nx.eigenvector_centrality_numpy(G))
            solver = "numpy_arpack_fallback"
        prv = np.array([pr[u] for u in nodes])
        evv = np.array([ev[u] for u in nodes])
        res["pagerank"] = {"alpha": PAGERANK_ALPHA, **centrality_summary(prv, nodes, deg)}
        res["eigenvector"] = {"solver": solver, **centrality_summary(evv, nodes, deg)}
        res["spearman"] = {
            "degree_pagerank": float(spearmanr(deg, prv).statistic),
            "degree_eigenvector": float(spearmanr(deg, evv).statistic),
            "pagerank_eigenvector": float(spearmanr(prv, evv).statistic)}
    return res, secs


def flat_metrics(scan: dict, res: dict, secs: dict) -> dict:
    """The numbers that go in the record's `metrics` (small, flat, numeric)."""
    c, d, t = res["components"], res["degree"], res["twins"]
    out = {
        "nodes": res["nodes"], "edges": res["edges"], "edges_per_node": res["edges_per_node"],
        "self_loops_dropped": scan["self_loop_lines"],
        "duplicate_or_reciprocal_lines_absorbed":
            scan["edge_lines"] - scan["self_loop_lines"] - res["edges"],
        "components": c["count"], "giant_nodes": c["giant_nodes"], "giant_share": c["giant_share"],
        "second_largest_component": c["second_largest"],
        "degree_mean": d["mean"], "degree_median": d["median"], "degree_p90": d["p90"],
        "degree_p99": d["p99"], "degree_max": d["max"], "degree1_share": d["share_degree_1"],
        "transitivity": res["transitivity"], "max_core": res["max_core"]["k"],
        "twin_nodes": t["twin_nodes"], "twin_share": t["twin_share"],
        "twin_chi_reference": t["twin_chi_reference"],
    }
    if "pagerank" in res:
        out.update({
            "pagerank_max": res["pagerank"]["max"],
            f"pagerank_top{TOP_K}_share": res["pagerank"][f"top{TOP_K}_share"],
            "pagerank_gini": res["pagerank"]["gini"],
            "eigenvector_max": res["eigenvector"]["max"],
            f"eigenvector_top{TOP_K}_share": res["eigenvector"][f"top{TOP_K}_share"],
            "eigenvector_gini": res["eigenvector"]["gini"],
            "spearman_degree_pagerank": res["spearman"]["degree_pagerank"],
            "spearman_degree_eigenvector": res["spearman"]["degree_eigenvector"],
            "spearman_pagerank_eigenvector": res["spearman"]["pagerank_eigenvector"]})
    out.update({f"seconds_{k}": v for k, v in secs.items()})
    return out


# ---------------------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------------------

def table_rows(name_res: list[tuple[str, dict, dict]]) -> list[tuple[str, list[str]]]:
    """The spec's characterisation table, one column per dataset, plus the additions."""
    rows: list[tuple[str, list[str]]] = []

    def row(label, fn):
        rows.append((label, [fn(r, s) for _, r, s in name_res]))

    row("nodes / edges (undirected, simple)", lambda r, s: f"{r['nodes']:,} / {r['edges']:,}")
    row("self-loop lines dropped", lambda r, s: f"{s['self_loop_lines']:,}")
    row("edges per node", lambda r, s: f"{r['edges_per_node']:.2f}")
    row("components; giant share (size)", lambda r, s: (
        f"{r['components']['count']:,}; {r['components']['giant_share']:.4f} "
        f"({r['components']['giant_nodes']:,})"))
    row("second-largest component", lambda r, s: f"{r['components']['second_largest']:,}")
    row("degree mean / median / p90 / p99 / max", lambda r, s: (
        f"{r['degree']['mean']:.2f} / {r['degree']['median']:g} / {r['degree']['p90']:g} / "
        f"{r['degree']['p99']:g} / {r['degree']['max']:,}"))
    row("share of degree-1 nodes", lambda r, s: f"{r['degree']['share_degree_1']:.3f}")
    row("global transitivity", lambda r, s: f"{r['transitivity']:.3f}")
    row("maximum core number", lambda r, s: f"{r['max_core']['k']}")
    row("nodes in a twin class (share)", lambda r, s: (
        f"{r['twins']['twin_nodes']:,} ({r['twins']['twin_share']:.3f})"))
    row("twin-adjusted chi reference, beta=0", lambda r, s: f"{r['twins']['twin_chi_reference']:.4f}")
    if any("pagerank" in r for _, r, _ in name_res):
        k = TOP_K
        row(f"PageRank: max / top-{k} share / Gini", lambda r, s: (
            f"{r['pagerank']['max']:.2e} / {r['pagerank'][f'top{k}_share']:.3f} / "
            f"{r['pagerank']['gini']:.3f}" if "pagerank" in r else "-"))
        row(f"eigenvector: max / top-{k} share / Gini", lambda r, s: (
            f"{r['eigenvector']['max']:.2e} / {r['eigenvector'][f'top{k}_share']:.3f} / "
            f"{r['eigenvector']['gini']:.3f}" if "pagerank" in r else "-"))
        row("Spearman deg-PR / deg-eig / PR-eig", lambda r, s: (
            f"{r['spearman']['degree_pagerank']:.3f} / {r['spearman']['degree_eigenvector']:.3f} / "
            f"{r['spearman']['pagerank_eigenvector']:.3f}" if "pagerank" in r else "-"))
    return rows


def render_table(name_res) -> str:
    rows = table_rows(name_res)
    names = [n for n, _, _ in name_res]
    w0 = max(len(label) for label, _ in rows)
    widths = [max(len(n), *(len(vals[i]) for _, vals in rows)) for i, n in enumerate(names)]
    out = [f"{'':<{w0}}  " + "  ".join(f"{n:>{w}}" for n, w in zip(names, widths))]
    out += [f"{label:<{w0}}  " + "  ".join(f"{v:>{w}}" for v, w in zip(vals, widths))
            for label, vals in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------
# one dataset: compute, write, record
# ---------------------------------------------------------------------------------------

def run_one(name: str, path: Path, root: Path, archive: Path, status: str, note: str,
            record: bool, centrality: bool, identity: Path | None) -> tuple[dict, dict]:
    src = resolve_edgelist_path(path)
    upload = None
    if identity is None:
        guess = src.with_suffix(".identity.json")
        identity = guess if guess.is_file() else None
    if identity is not None:
        ident = json.loads(Path(identity).read_text())
        if ident.get("schema") != "e4_fb_identity/1":
            raise SystemExit(f"{identity} is not an identity JSON from scripts/e4_convert_fb.py "
                             f"(schema {ident.get('schema')!r})")
        upload = {"file_name": ident["source"]["file_name"], "format": ident["source"]["format"],
                  "size_bytes": ident["source"]["size_bytes"], "sha256": ident["source"]["sha256"],
                  "identity_status": ident["identity_check"]["status"],
                  "identity_verdict": ident["identity_check"]["verdict"],
                  "converted_sha256": (ident.get("converted") or {}).get("sha256")}

    rec = None
    if record:
        rec = RunRecord(KIND, name, status, note or f"E4a characterisation of {name}", archive=archive)
        rec.set(seed=0, dataset=name, parent=Path(path).as_posix(),
                randomness="none: every quantity is a deterministic function of the edge list",
                config={"percentile_method": PERCENTILE_METHOD, "top_k": TOP_K,
                        "pagerank_alpha": PAGERANK_ALPHA, "eigenvector_max_iter": EIGEN_MAX_ITER,
                        "eigenvector_fallback": "nx.eigenvector_centrality_numpy",
                        "centrality": centrality,
                        "graph": "load_parent: undirected simple graph, self-loops dropped",
                        "twin_class": "same open neighbourhood or same closed neighbourhood"},
                upload=upload)
        rec.add_inputs(path)
        rec.seal_id()

    status_exit, error = "ok", None
    res = scan = secs = None
    out_json = None
    with capture_stderr() as console:
        try:
            print(f"[{name}] {src}", file=sys.stderr)
            scan = scan_edgelist(path)
            G = load_parent(path, verbose=False)
            if G.number_of_nodes() != scan["nodes_with_a_non_loop_edge"]:
                raise RuntimeError(
                    f"scan and load_parent disagree: {scan['nodes_with_a_non_loop_edge']} nodes "
                    f"with a non-loop edge against {G.number_of_nodes()} in the loaded graph")
            res, secs = characterize(G, centrality=centrality, log=lambda s: print(s, file=sys.stderr))
            out_dir = Path(root) / name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_json = out_dir / "characterization.json"
            doc = {"schema": SCHEMA, "name": name, "source": describe_file(src, REPO_ROOT),
                   "upload": upload, "scan": scan,
                   "self_loops_dropped": scan["self_loop_lines"],
                   "duplicate_or_reciprocal_lines_absorbed":
                       scan["edge_lines"] - scan["self_loop_lines"] - res["edges"],
                   **res}
            out_json.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        except BaseException:
            status_exit, error = "error", traceback.format_exc()
            raise
        finally:
            if rec:
                if out_json is not None and out_json.exists():
                    rec.add_outputs(out_json)
                if res is not None:
                    rec.metric(**flat_metrics(scan, res, secs))
                rec.finish(status_exit, error, console.getvalue())
    return res, scan


# ---------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------

def _self_test() -> int:
    import tempfile
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    def stats(G):
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        return twin_stats(G, comps[0], G.number_of_nodes())

    # hand-checked graphs
    P4 = nx.path_graph(4)
    s = stats(P4)
    check("path of 4: no twins, reference 1.0", (s["twin_nodes"], s["twin_chi_reference"]) == (0, 1.0), str(s))
    star = nx.star_graph(3)                      # centre 0, leaves 1-3 are open twins
    s = stats(star)
    check("star with 3 leaves: 3 open twins, reference (4-2)/4", (s["open_twin_nodes"], s["closed_twin_nodes"],
          s["twin_chi_reference"]) == (3, 0, 0.5), str(s))
    K4 = nx.complete_graph(4)                    # all four are closed twins
    s = stats(K4)
    check("K4: 4 closed twins, reference (4-3)/4", (s["open_twin_nodes"], s["closed_twin_nodes"],
          s["twin_chi_reference"]) == (0, 4, 0.25), str(s))
    H = nx.Graph([(0, 1), (2, 3), (3, 4)])       # K2 (closed twins, outside the giant) + P3 (2,4 open twins)
    s = stats(H)
    check("K2 + P3: 4 twin nodes, only the P3 class is in the giant: reference (3-1)/5",
          (s["twin_nodes"], s["twin_classes_in_giant"], s["twin_chi_reference"]) == (4, 1, 0.4), str(s))

    # grouping against a brute-force pairwise definition on random graphs
    rng = np.random.default_rng(3)
    agree = True
    for seed in range(40):
        G = nx.gnp_random_graph(int(rng.integers(8, 40)), float(rng.uniform(0.04, 0.3)), seed=seed)
        G.remove_nodes_from([u for u in list(G) if G.degree(u) == 0])
        if G.number_of_nodes() < 2:
            continue
        parent = {u: u for u in G}

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        nodes = list(G)
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if set(G[u]) == set(G[v]) or set(G[u]) | {u} == set(G[v]) | {v}:
                    parent[find(u)] = find(v)
        classes: dict = defaultdict(list)
        for u in nodes:
            classes[find(u)].append(u)
        brute = sum(len(c) for c in classes.values() if len(c) > 1)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        giant = comps[0]
        removed = sum(len(c) - 1 for c in classes.values() if len(c) > 1 and c[0] in giant)
        got = twin_stats(G, giant, G.number_of_nodes())
        agree &= (got["twin_nodes"] == brute
                  and abs(got["twin_chi_reference"] - (len(giant) - removed) / G.number_of_nodes()) < 1e-12)
    check("twin grouping equals the pairwise definition on 40 random graphs", agree)

    # nearest-rank percentiles and the other degree rows on a graph with known degrees
    deg = np.arange(1, 101)
    check("nearest-rank p90/p99 of degrees 1..100 are 90 and 99 (linear would give 90.1, 99.01)",
          (float(np.percentile(deg, 90, method=PERCENTILE_METHOD)),
           float(np.percentile(deg, 99, method=PERCENTILE_METHOD))) == (90.0, 99.0))
    check("gini: equal values 0, one holder (n-1)/n",
          abs(gini(np.ones(10))) < 1e-12 and abs(gini(np.array([0.0] * 9 + [1.0])) - 0.9) < 1e-12)

    # an end-to-end run on a tiny file, record and bulk output into a temp dir
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        f = td / "tiny.txt"
        f.write_text("# tiny\n1 2\n2 1\n2 3\n3 4\n4 4\n5 6\n7 7\n6 5\n")
        res, scan = run_one("tiny", f, td / "char", td / "arch", "exploratory", "", True, True, None)
        check("end to end: 6 nodes / 4 edges, 2 self-loop lines, node 7 only in a loop",
              (res["nodes"], res["edges"], scan["self_loop_lines"], scan["nodes_only_in_self_loops"])
              == (6, 4, 2, 1), f"{res['nodes']} {res['edges']} {scan}")
        recs = sorted((td / "arch").glob("*/record.json"))
        d = json.loads(recs[0].read_text()) if recs else {}
        check("end to end: one record of kind dataset_characterization, exploratory, complete",
              len(recs) == 1 and d.get("kind") == KIND and d.get("status") == "exploratory"
              and "record_incomplete" not in d and d["metrics"]["nodes"] == 6,
              recs[0].parent.name if recs else "no record")
        out = td / "char" / "tiny" / "characterization.json"
        check("end to end: bulk JSON written and hashed into the record",
              out.is_file() and d["outputs"][0]["sha256"] == __import__("hashlib").sha256(out.read_bytes()).hexdigest())
        res2, _ = run_one("tiny", f, td / "char", td / "arch2", "confirmatory", "", True, False, None)
        d2 = json.loads(next((td / "arch2").glob("*/record.json")).read_text())
        check("--status confirmatory is honoured; --no-centrality skips PageRank",
              d2["status"] == "confirmatory" and "pagerank" not in res2)
    print("e4_characterize self-test " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="E4a dataset characterisation: one RunRecord of kind "
                    "dataset_characterization per dataset.")
    ap.add_argument("datasets", nargs="*", metavar="NAME=EDGELIST",
                    help="datasets to characterise; NAME defaults to the file stem")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"bulk output root (default {DEFAULT_ROOT.relative_to(REPO_ROOT)}); "
                         "writes <root>/NAME/characterization.json")
    ap.add_argument("--archive", type=Path, default=ARCHIVE,
                    help="RunRecord archive (default results/runs, the policy's tracked archive)")
    ap.add_argument("--status", choices=("exploratory", "confirmatory"), default="exploratory",
                    help="exploratory unless a numeric expectation was written down first")
    ap.add_argument("--note", default="", help="goes in the record")
    ap.add_argument("--identity", type=Path, default=None,
                    help="identity JSON from scripts/e4_convert_fb.py (default: "
                         "<edgelist>.identity.json if it exists); its upload facts go in the record")
    ap.add_argument("--no-centrality", action="store_true", help="skip PageRank and eigenvector")
    ap.add_argument("--no-record", action="store_true", help="debugging only: no RunRecord")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.datasets:
        ap.error("give at least one NAME=EDGELIST (or --self-test)")

    results = []
    for spec in args.datasets:
        name, _, p = spec.partition("=")
        if not p:
            name, p = Path(spec).stem.replace(".", "_"), spec
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            ap.error(f"dataset name {name!r} must be letters, digits, '_' or '-' (it names a "
                     "directory and is part of the run id)")
        res, scan = run_one(name, Path(p), args.root, args.archive, args.status, args.note,
                            not args.no_record, not args.no_centrality, args.identity)
        results.append((name, res, scan))
    print(render_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
