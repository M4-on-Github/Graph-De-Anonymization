#!/usr/bin/env python3
"""Generate evaluation graph pairs by the Li et al. (CIKM '20) perturbation protocol.

Takes one real parent graph and cuts two independently damaged copies from it, so the
correct node mapping is known by construction. See `perturbation_protocol.md` in this
directory for the paper provenance of every step and for what this protocol cannot show.

Library use:
    from perturbation import load_parent, make_pair
    G = load_parent('data/raw/cit-HepTh.txt')
    pair = make_pair(G, beta=0.1, seed=0)

CLI:
    python src/beta_noise_generator/perturbation.py --self-test
    python src/beta_noise_generator/perturbation.py --parent <edgelist> --beta 0.1 0.3 0.5 --repeats 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------
# Parent loading
# --------------------------------------------------------------------------------------

def resolve_edgelist_path(path) -> Path:
    """Return the actual file to read.

    SNAP's Google+ archive unpacks as a *directory* named `gplus_combined.txt` holding a
    file of the same name, so a naive open() on the given path fails. Handle that, and the
    general case of a directory with exactly one file in it.
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        inner = p / p.name
        if inner.is_file():
            return inner
        files = [f for f in sorted(p.iterdir()) if f.is_file()]
        if len(files) == 1:
            return files[0]
        raise FileNotFoundError(
            f"{p} is a directory with {len(files)} files; point --parent at one of them")
    raise FileNotFoundError(f"no such edge list: {p}")


class _NodeSampler:
    """Deterministic single-pass node sampling by stable hash.

    A two-pass approach would need the node set up front, which is exactly what we are
    trying to avoid on a 30M-line file. Hashing the id gives the same decision every run
    for a given seed, and costs one dict lookup per repeated node.
    """

    def __init__(self, frac: float, seed: int):
        self.frac = frac
        self.key = f"gda-node-sample-{seed}".encode()[:64]
        self.cache: dict[str, bool] = {}

    def keep(self, node: str) -> bool:
        hit = self.cache.get(node)
        if hit is None:
            digest = hashlib.blake2b(node.encode(), digest_size=8, key=self.key).digest()
            hit = int.from_bytes(digest, "big") / 2 ** 64 < self.frac
            self.cache[node] = hit
        return hit


def load_parent(path, node_sample: float | None = None, seed: int = 0,
                progress_every: int = 2_000_000, verbose: bool = True) -> nx.Graph:
    """Stream an edge list into an undirected simple graph.

    **Collapses to undirected before anything else.** This ordering is the whole point:
    perturbing a symmetric (directed) line list and collapsing afterwards squares the
    intended deletion rate, because an undirected edge only disappears when both of its
    lines are dropped. That is the bug baked into `data/labeled_dev`.

    Duplicate lines and reciprocal pairs are absorbed by nx.Graph; self-loops are dropped.
    """
    src = resolve_edgelist_path(path)
    sampler = _NodeSampler(node_sample, seed) if node_sample else None

    G = nx.Graph()
    lines = comments = selfloops = skipped = malformed = 0
    t0 = time.time()

    with src.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            lines += 1
            if verbose and progress_every and lines % progress_every == 0:
                # Never call G.number_of_edges() in here -- it sums degrees over every
                # node and turns a 1.6 s load into a 230 s one.
                print(f"  ... {lines:,} lines in {time.time() - t0:.0f}s", file=sys.stderr)
            line = line.strip()
            if not line:
                continue
            if line[0] in "#%":
                comments += 1
                continue
            parts = line.split()
            if len(parts) < 2:
                malformed += 1
                continue
            u, v = parts[0], parts[1]
            if u == v:
                selfloops += 1
                continue
            if sampler and not (sampler.keep(u) and sampler.keep(v)):
                skipped += 1
                continue
            G.add_edge(u, v)

    if verbose:
        print(f"parent: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} undirected "
              f"edges from {lines:,} lines in {time.time() - t0:.1f}s", file=sys.stderr)
        print(f"        {comments:,} comment, {selfloops:,} self-loop, "
              f"{malformed:,} malformed, {skipped:,} out-of-sample lines", file=sys.stderr)
    return G


# --------------------------------------------------------------------------------------
# Perturbation
# --------------------------------------------------------------------------------------

def perturb(G: nx.Graph, beta: float, rng: np.random.Generator,
            mode: str = "bernoulli") -> nx.Graph:
    """Delete a beta fraction of edges, then drop the nodes left with degree 0.

    Nodes are never deleted directly; the node set shrinks only as a side effect. That is
    the protocol as stated, and it is also why node overlap is not a free parameter here.
    """
    edges = list(G.edges())
    m = len(edges)
    if mode == "bernoulli":
        kept = [e for e, k in zip(edges, rng.random(m) >= beta) if k]
    elif mode == "exact":
        n_remove = int(round(beta * m))
        kept = [edges[i] for i in rng.permutation(m)[n_remove:]]
    else:
        raise ValueError(f"mode must be 'bernoulli' or 'exact', got {mode!r}")

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(kept)
    H.remove_nodes_from([n for n, d in H.degree() if d == 0])
    return H


def _sort_key(node: str):
    try:
        return (0, int(node), "")
    except ValueError:
        return (1, 0, node)


def relabel_random(G: nx.Graph, rng: np.random.Generator) -> tuple[nx.Graph, dict]:
    """Give every node a fresh id drawn from a random permutation of 0..n-1.

    The paper leaves ground truth as the identity. Permuting costs nothing and removes a
    silent failure mode: with identity ids, any matcher bug that compares raw node ids
    scores 100% and looks like a result.
    """
    nodes = sorted(G.nodes(), key=_sort_key)
    perm = rng.permutation(len(nodes))
    mapping = {n: str(int(p)) for n, p in zip(nodes, perm)}
    return nx.relabel_nodes(G, mapping, copy=True), mapping


def _overlap_stats(Ga: nx.Graph, Gu: nx.Graph, m_parent: int, beta: float) -> dict:
    """Measured edge overlap against what the protocol predicts.

    Each edge survives both draws with probability (1-beta)^2, so
        |E^a cap E^u| / |E|        -> (1-beta)^2
        Jaccard(E^a, E^u)          -> (1-beta)^2 / (1-beta^2) = (1-beta)/(1+beta)
    These are the protocol's own correctness check, stated in the paper alongside it.
    """
    ma, mu = Ga.number_of_edges(), Gu.number_of_edges()
    small, large = (Ga, Gu) if ma <= mu else (Gu, Ga)
    shared = sum(1 for u, v in small.edges() if large.has_edge(u, v))
    union = ma + mu - shared

    p = (1.0 - beta) ** 2
    sigma = math.sqrt(p * (1.0 - p) / m_parent) if m_parent else 0.0
    return {
        "parent_edges": m_parent,
        "edges_a": ma,
        "edges_u": mu,
        "shared_edges": shared,
        "retention_a": ma / m_parent if m_parent else 0.0,
        "retention_u": mu / m_parent if m_parent else 0.0,
        "retention_predicted": 1.0 - beta,
        "overlap_ratio": shared / m_parent if m_parent else 0.0,
        "overlap_predicted": p,
        "overlap_sigma": sigma,
        "jaccard": shared / union if union else 0.0,
        "jaccard_predicted": (1.0 - beta) / (1.0 + beta),
    }


def check_overlap(stats: dict, tol: float | None = None) -> tuple[bool, str]:
    """Self-test: does the measured overlap match (1-beta)^2?

    Tolerance is four binomial standard deviations, floored at 0.01 so small sampled
    graphs do not trip. The failure this is really guarding against -- perturbing a
    directed line list and collapsing afterwards -- misses by a mile, not by a sigma.
    """
    if tol is None:
        tol = max(0.01, 4.0 * stats["overlap_sigma"])
    delta = abs(stats["overlap_ratio"] - stats["overlap_predicted"])
    ok = delta <= tol
    msg = (f"overlap {stats['overlap_ratio']:.4f} vs predicted "
           f"{stats['overlap_predicted']:.4f} (delta {delta:.4f}, tol {tol:.4f})")
    return ok, msg


def make_pair(G: nx.Graph, beta: float, seed: int = 0, mode: str = "bernoulli",
              permute: bool = True) -> dict:
    """Cut one parent graph into (G^a, G^u) plus exact ground truth.

    Returns a dict with `Ga`, `Gu`, `pairs` (ground truth over V^a cap V^u) and `stats`.
    """
    m_parent = G.number_of_edges()
    Ga = perturb(G, beta, np.random.default_rng([seed, 1]), mode)
    Gu = perturb(G, beta, np.random.default_rng([seed, 2]), mode)

    # Overlap must be measured on the shared id space, i.e. before any relabelling.
    stats = _overlap_stats(Ga, Gu, m_parent, beta)

    common = sorted(set(Ga.nodes()) & set(Gu.nodes()), key=_sort_key)
    stats.update({
        "beta": beta,
        "seed": seed,
        "mode": mode,
        "permuted": permute,
        "parent_nodes": G.number_of_nodes(),
        "nodes_a": Ga.number_of_nodes(),
        "nodes_u": Gu.number_of_nodes(),
        "isolated_dropped_a": G.number_of_nodes() - Ga.number_of_nodes(),
        "isolated_dropped_u": G.number_of_nodes() - Gu.number_of_nodes(),
        "matchable_nodes": len(common),
    })

    if permute:
        Ga, map_a = relabel_random(Ga, np.random.default_rng([seed, 3]))
        Gu, map_u = relabel_random(Gu, np.random.default_rng([seed, 4]))
        pairs = [(map_a[n], map_u[n]) for n in common]
    else:
        pairs = [(n, n) for n in common]

    return {"Ga": Ga, "Gu": Gu, "pairs": pairs, "stats": stats}


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------

def write_pair(outdir: Path, pair: dict) -> Path:
    """Write G1.edgelist / G2.edgelist / mapping.txt / stats.json.

    Filenames match what the inherited baseline looks for. Note that `src/seed_based.py`
    and `src/seed_free.py` also gate loading on the literal substring `labeled_dev` being
    in the *directory* path, so a generated directory will not load under them unless it
    is named accordingly -- see this package's README note.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for name, G in (("G1.edgelist", pair["Ga"]), ("G2.edgelist", pair["Gu"])):
        edges = sorted((min(e, key=_sort_key), max(e, key=_sort_key)) for e in G.edges())
        with (outdir / name).open("w", encoding="utf-8") as fh:
            for u, v in sorted(edges, key=lambda e: (_sort_key(e[0]), _sort_key(e[1]))):
                fh.write(f"{u} {v}\n")

    with (outdir / "mapping.txt").open("w", encoding="utf-8") as fh:
        for a, u in pair["pairs"]:
            fh.write(f"{a} {u}\n")

    with (outdir / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(pair["stats"], fh, indent=2)
        fh.write("\n")

    return outdir


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------

def self_test(verbose: bool = True) -> bool:
    """Run the protocol's own checks on a synthetic parent. No dataset needed."""
    ok = True

    def report(label, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}"
                  + (f" -- {detail}" if detail else ""))

    G = nx.gnm_random_graph(4000, 40000, seed=7)
    G = nx.relabel_nodes(G, {n: str(n) for n in G})
    if verbose:
        print(f"synthetic parent: {G.number_of_nodes():,} nodes / "
              f"{G.number_of_edges():,} edges")

    print("\n(1-beta)^2 overlap check, beta swept over the paper's range:")
    for beta in (0.1, 0.2, 0.3, 0.4, 0.5):
        pair = make_pair(G, beta, seed=1)
        passed, msg = check_overlap(pair["stats"])
        report(f"beta={beta}", passed, msg)

    print("\nground truth is a bijection over V^a cap V^u:")
    pair = make_pair(G, 0.3, seed=2)
    s = pair["stats"]
    a_ids = [a for a, _ in pair["pairs"]]
    u_ids = [u for _, u in pair["pairs"]]
    report("no duplicate ids on either side",
           len(set(a_ids)) == len(a_ids) and len(set(u_ids)) == len(u_ids))
    report("every mapped id exists in its graph",
           set(a_ids) <= set(pair["Ga"].nodes()) and set(u_ids) <= set(pair["Gu"].nodes()))
    report("pair count == |V^a cap V^u|", len(pair["pairs"]) == s["matchable_nodes"],
           f"{len(pair['pairs']):,} pairs")

    print("\npermutation actually permutes (guards the identity-leak failure mode):")
    identity_hits = sum(1 for a, u in pair["pairs"] if a == u)
    report("near-zero identity pairs", identity_hits < max(10, 0.01 * len(pair["pairs"])),
           f"{identity_hits} of {len(pair['pairs']):,}")

    print("\nregression guard: collapse BEFORE perturbing, never after")
    # Perturbing a symmetric line list and collapsing afterwards loses an undirected edge
    # only when both directions are dropped, so the effective rate is beta^2, not beta.
    # This is what produced data/labeled_dev's ~1-2% effective noise from a nominal 10%.
    beta = 0.1
    rng = np.random.default_rng(11)
    lines = [(u, v) for u, v in G.edges()] + [(v, u) for u, v in G.edges()]
    kept = [e for e, k in zip(lines, rng.random(len(lines)) >= beta) if k]
    H = nx.Graph()
    H.add_edges_from(kept)
    wrong_beta = 1.0 - H.number_of_edges() / G.number_of_edges()
    report("collapse-after yields beta^2, i.e. the wrong graph",
           abs(wrong_beta - beta ** 2) < 0.01,
           f"nominal {beta}, effective {wrong_beta:.4f}, beta^2 = {beta ** 2}")
    right = perturb(G, beta, np.random.default_rng(11))
    report("collapse-before yields beta",
           abs((1.0 - right.number_of_edges() / G.number_of_edges()) - beta) < 0.01,
           f"effective {1.0 - right.number_of_edges() / G.number_of_edges():.4f}")

    print(f"\n{'all checks passed' if ok else 'SELF-TEST FAILED'}")
    return ok


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate (G^a, G^u, ground truth) pairs by the Li et al. "
                    "edge-deletion protocol. See perturbation_protocol.md.")
    ap.add_argument("--parent", type=str,
                    help="Edge list of the parent graph. May be a directory holding a "
                         "single file (SNAP's gplus_combined.txt unpacks that way).")
    ap.add_argument("--beta", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5],
                    help="Edge deletion rate(s). Paper sweeps 0.1-0.5. Default: all five.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Pairs per beta, each with its own seed. Paper averages 5.")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    ap.add_argument("--mode", choices=["bernoulli", "exact"], default="bernoulli",
                    help="Independent coin per edge, or exactly round(beta*|E|) removals. "
                         "The paper's wording is ambiguous; declare which you used.")
    ap.add_argument("--no-permute", action="store_true",
                    help="Keep identity ground truth, as the paper does. Not recommended.")
    ap.add_argument("--node-sample", type=float, default=None,
                    help="Keep this fraction of parent nodes (hash-based, reproducible). "
                         "Use on Google+ rather than loading all 30M lines.")
    ap.add_argument("--out", type=str, default=str(REPO_ROOT / "data" / "generated"),
                    help="Directory to write the generated pairs into.")
    ap.add_argument("--compat-name", action="store_true",
                    help="Suffix output dirs with '_labeled_dev' so the inherited "
                         "baseline's substring path guard accepts them.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Write the pair even if the (1-beta)^2 check fails.")
    ap.add_argument("--self-test", action="store_true",
                    help="Run the protocol checks on a synthetic graph and exit.")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1
    if not args.parent:
        ap.error("--parent is required (or use --self-test)")

    G = load_parent(args.parent, node_sample=args.node_sample, seed=args.seed)
    if G.number_of_edges() == 0:
        print("parent graph has no edges", file=sys.stderr)
        return 1

    stem = resolve_edgelist_path(args.parent).stem.replace(".", "_")
    out_root = Path(args.out)
    failures = 0

    for beta in args.beta:
        for r in range(args.repeats):
            seed = args.seed + r
            pair = make_pair(G, beta, seed=seed, mode=args.mode,
                             permute=not args.no_permute)
            passed, msg = check_overlap(pair["stats"])
            if not passed:
                failures += 1
                print(f"  VERIFY FAILED beta={beta} seed={seed}: {msg}", file=sys.stderr)
                if not args.no_verify:
                    continue

            name = f"{stem}_b{round(beta * 100):02d}_s{seed}"
            if args.compat_name:
                name += "_labeled_dev"
            outdir = write_pair(out_root / name, pair)
            s = pair["stats"]
            print(f"{outdir.relative_to(REPO_ROOT) if outdir.is_relative_to(REPO_ROOT) else outdir}"
                  f"  |V^a cap V^u|={s['matchable_nodes']:,}"
                  f"  E^a={s['edges_a']:,} E^u={s['edges_u']:,}"
                  f"  {msg}")

    if failures and not args.no_verify:
        print(f"\n{failures} pair(s) skipped: measured overlap did not match (1-beta)^2. "
              f"That usually means the parent was not collapsed to undirected first.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
