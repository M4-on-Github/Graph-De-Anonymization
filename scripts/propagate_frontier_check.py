#!/usr/bin/env python3
"""Check that the skipping `propagate` (src/adv_gd/propagation.py) equals the full rescan.

Runs the OLD implementation (--old, a propagation.py that rescans every unmapped node in
every sweep) and the NEW one from src/adv_gd/propagation.py on many synthetic pairs and
compares, per case:
  * the sha256 of the sorted "<a> <u>" lines of the mapping, and the mapping's item order
    (ignition_grid.py writes the mapping in insertion order, so that order is in the
    stored hashes too);
  * the sweep lines on stderr, as (sweep, stage, +N, mapped) tuples (the time is dropped);
  * the exception, if one is raised.
It prints old/new seconds per case and, from NEW's propagation.LAST_EVALUATIONS, how many
forward evaluations the new code made against how many the full rescan makes.

Cases vary: seed; n in 200..3000; gnm or powerlaw-cluster parent (a few directed gnm);
5-25% of the parent's edges deleted independently per side; some nodes dropped from one
side; shuffled node and edge insertion order; isolated nodes and self-loops; disjoint or
shared id spaces; anchors by degree or at random, 0-30% wrong, a few naming missing
nodes, some duplicated on the a or u side; candidates {} / {a: [u]} per anchor /
partly-true lists for 40% of nodes (so stage one accepts nodes and the flag flip matters);
support "all" and "nonzero"; theta 0.5 in most cases, a few other values; max_sweeps 100
or 2-3.

Usage (from the repo root, in the container; no install needed):
    python3 scripts/propagate_frontier_check.py \\
        --old /data/mmyatmau/Graph-De-Anonymization/scratch/propagation_old.py
    ... [--cases 240] [--seed 0] [--max-n 3000] [--case K]
Exit status 1 on any mismatch, 2 on setup error.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_PATH = REPO_ROOT / "src" / "adv_gd" / "propagation.py"
SWEEP_RE = re.compile(r"^\s*sweep\s+(\d+)\s+stage\s+(\d+)\s+\+([\d,]+)\s+->\s+([\d,]+)"
                      r" mapped\s+\d+s$")
SIZES = [200, 300, 500, 800, 1200, 2000, 3000]


def load_module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sorted_hash(mapping):
    lines = sorted(f"{a} {u}\n" for a, u in mapping.items())
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def run(mod, Ga, Gu, anchors, candidates, theta, support, max_sweeps):
    buf = io.StringIO()
    t = time.time()
    exc, m = None, None
    with contextlib.redirect_stderr(buf):
        try:
            m = mod.propagate(Ga, Gu, anchors, candidates, theta=theta,
                              support=support, max_sweeps=max_sweeps, verbose=True)
        except Exception as e:                       # compared, not hidden
            exc = f"{type(e).__name__}: {e}"
    dt = time.time() - t
    sweeps, other = [], []
    for line in buf.getvalue().splitlines():
        mt = SWEEP_RE.match(line)
        if mt:
            sweeps.append((int(mt.group(1)), int(mt.group(2)),
                           int(mt.group(3).replace(",", "")),
                           int(mt.group(4).replace(",", ""))))
        else:
            other.append(line)
    return m, sweeps, other, exc, dt


def build_side(nx, nodes, edges, rng, directed, label):
    H = nx.DiGraph() if directed else nx.Graph()
    ns = list(nodes)
    rng.shuffle(ns)
    H.add_nodes_from(label(n) for n in ns)
    es = list(edges)
    rng.shuffle(es)
    for x, y in es:
        if not directed and rng.random() < 0.5:
            x, y = y, x
        H.add_edge(label(x), label(y))
    return H


def make_case(nx, k, base_seed, max_n):
    """Return (description, Ga, Gu, anchors, candidates, theta, support, max_sweeps)."""
    rng = random.Random(base_seed * 1_000_003 + k)
    gseed = rng.randrange(2**31)
    n = rng.choice([s for s in SIZES if s <= max_n] or [max_n])
    directed = rng.random() < 0.05
    if directed or rng.random() < 0.5:
        deg = rng.choice([4, 8, 16])
        G = nx.gnm_random_graph(n, n * deg // 2, seed=gseed, directed=directed)
        model = f"gnm{'-dir' if directed else ''}(n={n},deg={deg})"
    else:
        m_ = rng.choice([2, 4, 8])
        p_ = rng.choice([0.1, 0.5])
        G = nx.powerlaw_cluster_graph(n, m_, p_, seed=gseed)
        model = f"plc(n={n},m={m_},p={p_})"

    edges = list(G.edges())
    da, du = rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.25)
    ea = [e for e in edges if rng.random() >= da]
    eu = [e for e in edges if rng.random() >= du]
    nodes = list(G.nodes())
    drop = rng.choice([0.0, 0.0, 0.05, 0.1])
    na = [v for v in nodes if rng.random() >= drop]
    nu = [v for v in nodes if rng.random() >= drop]
    sa, su = set(na), set(nu)
    ea = [(x, y) for x, y in ea if x in sa and y in sa]
    eu = [(x, y) for x, y in eu if x in su and y in su]
    if rng.random() < 0.1:                       # self-loops
        ea += [(v, v) for v in rng.sample(na, max(1, len(na) // 100))]
        eu += [(v, v) for v in rng.sample(nu, max(1, len(nu) // 100))]
    shared = rng.random() < 0.15
    la = (lambda v: f"n{v}") if shared else (lambda v: f"a{v}")
    lu = (lambda v: f"n{v}") if shared else (lambda v: f"u{v}")
    Ga = build_side(nx, na, ea, rng, directed, la)
    Gu = build_side(nx, nu, eu, rng, directed, lu)
    for i in range(rng.choice([0, 3, 10])):     # isolated nodes
        Ga.add_node(f"iso_a{i}")
        Gu.add_node(f"iso_u{i}")

    common = [v for v in nodes if v in sa and v in su]
    frac = rng.choice([0.01, 0.03, 0.08])
    n_anch = max(5, int(frac * len(common)))
    if rng.random() < 0.6:
        pick = sorted(common, key=lambda v: -G.degree(v))[:n_anch]
    else:
        pick = rng.sample(common, min(n_anch, len(common)))
    wrong = rng.choice([0.0, 0.1, 0.3])
    anchors = []
    for v in pick:
        if rng.random() < wrong:
            anchors.append((la(v), lu(rng.choice(nu))))
        else:
            anchors.append((la(v), lu(v)))
    anchors.append(("missing_a", lu(rng.choice(nu))))
    anchors.append((la(rng.choice(na)), "missing_u"))
    dup = rng.random() < 0.2
    if dup and len(anchors) > 4:
        for _ in range(3):                       # same a, other u; same u, other a
            a, _u = rng.choice(anchors[:-2])
            anchors.append((a, lu(rng.choice(nu))))
            _a, u = rng.choice(anchors[:-2])
            anchors.append((la(rng.choice(na)), u))
    rng.shuffle(anchors)

    cmode = rng.choice(["empty", "anchors", "partial_truth"])
    if cmode == "empty":
        candidates = {}
    elif cmode == "anchors":
        candidates = {a: [u] for a, u in anchors}
    else:
        candidates = {}
        for v in na:
            if rng.random() < 0.4:
                lst = [lu(rng.choice(nu))]
                if v in su and rng.random() < 0.8:
                    lst.append(lu(v))
                rng.shuffle(lst)
                candidates[la(v)] = lst
    support = rng.choice(["all", "nonzero"])
    theta = 0.5 if rng.random() < 0.8 else rng.choice([0.2, 1.0, 2.0])
    max_sweeps = 100 if rng.random() < 0.9 else rng.choice([2, 3])
    desc = (f"{model} del={da:.2f}/{du:.2f} drop={drop} shared={int(shared)} "
            f"anch={len(anchors)} wrong={wrong} dup={int(dup)} cand={cmode} "
            f"sup={support} theta={theta} ms={max_sweeps}")
    return desc, Ga, Gu, anchors, candidates, theta, support, max_sweeps


def full_rescan_evals(Ga, sweeps):
    """Forward evaluations the full rescan makes: the unmapped count at each sweep start."""
    out = []
    for _, _, plus, mapped in sweeps:
        out.append(Ga.number_of_nodes() - (mapped - plus))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, metavar="PATH",
                    help="full-rescan propagation.py to compare against")
    ap.add_argument("--cases", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=3000)
    ap.add_argument("--case", type=int, default=None, help="run only case K")
    args = ap.parse_args(argv)

    try:
        import networkx as nx
        old = load_module_from(args.old, "propagation_old")
        new = load_module_from(NEW_PATH, "propagation_new")
    except Exception as e:
        print(f"setup error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if not hasattr(new, "LAST_EVALUATIONS"):
        print(f"{NEW_PATH} has no LAST_EVALUATIONS; is it the new code?", file=sys.stderr)
        return 2
    print(f"networkx {nx.__version__}  python {sys.version.split()[0]}")
    print(f"old: {args.old}\nnew: {NEW_PATH}")

    ks = [args.case] if args.case is not None else range(args.cases)
    bad = 0
    t_old_tot = t_new_tot = 0.0
    ev_new_tot = ev_full_tot = 0
    print("case\tresult\told_s\tnew_s\tspeedup\tsweeps\tmapped\tevals_new/full\tdesc")
    for k in ks:
        desc, Ga, Gu, anchors, cands, theta, support, ms = make_case(nx, k, args.seed,
                                                                     args.max_n)
        m_o, sw_o, oth_o, ex_o, t_o = run(old, Ga, Gu, anchors, cands, theta, support, ms)
        m_n, sw_n, oth_n, ex_n, t_n = run(new, Ga, Gu, anchors, cands, theta, support, ms)
        ev_n = list(new.LAST_EVALUATIONS)
        problems = []
        if ex_o != ex_n:
            problems.append(f"exception old={ex_o!r} new={ex_n!r}")
        if m_o is not None and m_n is not None:
            if sorted_hash(m_o) != sorted_hash(m_n):
                problems.append("mapping sha256 differs")
            elif list(m_o.items()) != list(m_n.items()):
                problems.append("mapping insertion order differs")
        if sw_o != sw_n:
            problems.append(f"sweeps old={sw_o} new={sw_n}")
        if oth_o or oth_n:
            problems.append(f"unparsed stderr old={oth_o[:3]} new={oth_n[:3]}")
        if not sw_n and ex_n is None:
            problems.append("no sweep lines")
        full = full_rescan_evals(Ga, sw_o)
        if ex_n is None and len(ev_n) != len(sw_n):
            problems.append(f"LAST_EVALUATIONS has {len(ev_n)} entries for "
                            f"{len(sw_n)} sweeps")
        if ex_n is None and any(e > f for e, f in zip(ev_n, full)):
            problems.append(f"new evaluated more than the rescan: {ev_n} vs {full}")
        bad += bool(problems)
        t_old_tot += t_o
        t_new_tot += t_n
        ev_new_tot += sum(ev_n)
        ev_full_tot += sum(full)
        mapped = sw_n[-1][3] if sw_n else "-"
        print(f"{k}\t{'MISMATCH' if problems else 'ok'}\t{t_o:.2f}\t{t_n:.2f}\t"
              f"{(t_o / t_n) if t_n else float('nan'):.1f}x\t{len(sw_n)}\t{mapped}\t"
              f"{sum(ev_n)}/{sum(full)}\t{desc}", flush=True)
        if problems:
            for p in problems:
                print(f"    {p}", flush=True)
            print(f"    per-sweep evals new={ev_n} full={full}", flush=True)
    n_cases = len(list(ks))
    print(f"{n_cases} cases, {bad} mismatches; old {t_old_tot:.1f}s new {t_new_tot:.1f}s; "
          f"forward evaluations new {ev_new_tot} vs full rescan {ev_full_tot}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
