"""E2 runner: does Phase-3 `propagate` ignite from a synthetic anchor set?

Experiment Log 3, section "E2: Phase-3 ignition-threshold grid". Read that
first; this file implements its "Design: grid" and "Metrics" items and nothing
else. The question is what decides ignition: the number of correct anchors k,
the fraction p = k/n, the number of wrong ones, or the pair's overlap.

One call runs one array task (`--block B --task-id T`): one instance (overlap
level, pair, draw) and one n group, 1 to 9 runs. A run builds an anchor set of
n pairs of which k are correct, hands it to the existing `propagate` unchanged
(propagation.py:90), scores the mapping with the existing `evaluate`, and writes
one RunRecord of kind `ignition_grid`. `ignition_plan.py` says which runs a task
holds; `--plan` prints them.

The anchor set (spec, "Design: grid"):

  pool        the 2,000 highest-degree nodes of each graph (`top_degree`,
              alignment.py:298; ties by node id). placement "uniform" makes the
              pool every node instead (exploratory block).
  correct     k pool nodes of G^a whose true partner is also in the G^u pool,
              drawn uniformly, each with its true partner. Under placement
              "uniform": k matchable nodes, drawn uniformly.
  wrong       w = n - k further G^a pool nodes, each with a partner that is not
              its true partner, one use per G^u node (and not a partner already
              used by a correct anchor, so the anchor set is one-to-one, which
              `propagate`'s psi dictionary needs). Scheme "uniform": the partner
              is a uniform draw from the unused G^u pool. Scheme
              "degree_matched": the unused G^u pool node whose degree is closest
              to the G^a node's (ties broken at random), assigned greedily in
              the order the wrong G^a nodes were drawn.
  candidates  the empty dict (base), or {a: [partner(a)]} for the anchors only
              (the control; `centrality_rank_match.py:122` does the same).
              Neither adds stage-0 acceptances: stage 0 only tests unmapped
              nodes (propagation.py:118, 137).
  settings    theta 0.5, support "all" (adv_gd.py:229-230, 448).

Draws use numpy.random.default_rng seeded from the list
[draw, n, k, scheme code, placement code] (integer codes, in ignition_plan.py),
so a run is reproducible from its cell. Every draw is made with `rng.random()`
and an argsort, the most stable part of numpy's Generator API, rather than
`choice` or `permutation`, whose streams numpy reserves the right to change.
The anchor list is written next to the mapping and both digests are in the
record, so a drift would show.

Candidate mode is not in the seed, on purpose: the singleton and empty runs of a
cell draw the same anchors, which is what P2.6 compares.

Output. The mapping and the anchor list go under `--out` (default
runs/ignition_grid/) as <out>/<pair root>/<pair>/out_mapping_ignition_grid_<run_id>.txt
and anchors_ignition_grid_<run_id>.txt; both are hashed into the record. The
runner writes no pairs. Records go to results/runs/ (`--archive` redirects).

    PYTHONHASHSEED=0 python3 src/ignition_grid/ignition_grid.py --block base --task-id 0
    python3 src/ignition_grid/ignition_grid.py --block base --plan
    PYTHONHASHSEED=0 python3 src/ignition_grid/ignition_grid.py --block base --task-id 2 --cells 1000:5,1000:200
    python3 src/ignition_grid/ignition_grid.py --self-test
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "adv_gd"))
sys.path.insert(0, str(HERE))

import networkx as nx            # noqa: E402
import numpy as np               # noqa: E402

import ignition_plan as plan     # noqa: E402
from adv_gd import DATASETS, load_pair                                   # noqa: E402
from alignment import evaluate_anchors, top_degree                       # noqa: E402
from propagation import evaluate, propagate                              # noqa: E402
from provenance import ARCHIVE, RunRecord, capture_stderr, sha256_file   # noqa: E402

KIND = "ignition_grid"


class Terminated(BaseException):
    """Raised by the SIGTERM handler, so that the `finally` that writes the
    record still runs. A BaseException so that no `except Exception` eats it."""


def _on_sigterm(signum, frame):
    raise Terminated(f"signal {signum}")


# ---------------------------------------------------------------------------------------
# The anchor set
# ---------------------------------------------------------------------------------------

def _shuffled(rng, items: list) -> list:
    """`items` in a uniformly random order, from rng.random() and a stable argsort."""
    if not items:
        return []
    order = np.argsort(rng.random(len(items)), kind="stable")
    return [items[i] for i in order]


def _pools(Ga: nx.Graph, Gu: nx.Graph, placement: str, pool_size: int):
    """(pool_a, pool_u): node ids in a fixed order. "pool": the `pool_size`
    highest-degree nodes of each graph, in degree order, ties by id (the order
    `top_degree` returns). "uniform": every node, in id order."""
    if placement == "pool":
        na, nu = list(Ga.nodes()), list(Gu.nodes())
        return ([na[i] for i in top_degree(Ga, na, pool_size)],
                [nu[i] for i in top_degree(Gu, nu, pool_size)])
    if placement == "uniform":
        return sorted(Ga.nodes()), sorted(Gu.nodes())
    raise ValueError(f"placement must be one of {plan.PLACEMENTS}, got {placement!r}")


def _wrong_partners(Ga, Gu, truth, wrong_a: list, free_u: list, scheme: str) -> list:
    """One G^u partner per wrong G^a node, never its true partner, never reused.
    `free_u` is the unused G^u pool in random order; it is what makes ties
    random and the uniform scheme uniform."""
    out = []
    if scheme == "uniform":
        free = list(free_u)
        for a in wrong_a:
            j = 1 if free and free[0] == truth.get(a) else 0
            if j >= len(free):
                raise ValueError(f"no wrong partner left for {a}")
            out.append((a, free.pop(j)))
        return out
    if scheme == "degree_matched":
        degs = np.array([Gu.degree(u) for u in free_u], dtype=np.int64)
        pos = {u: i for i, u in enumerate(free_u)}
        alive = np.ones(len(free_u), dtype=bool)
        big = np.iinfo(np.int64).max // 2
        for a in wrong_a:
            gap = np.abs(degs - Ga.degree(a))
            gap[~alive] = big
            if truth.get(a) in pos:
                gap[pos[truth[a]]] = big          # the partner has to be wrong
            j = int(np.argmin(gap))               # first minimum: random among ties
            if gap[j] >= big:
                raise ValueError(f"no wrong partner left for {a}")
            alive[j] = False
            out.append((a, free_u[j]))
        return out
    raise ValueError(f"scheme must be one of {plan.SCHEMES}, got {scheme!r}")


def build_anchors(Ga: nx.Graph, Gu: nx.Graph, truth: dict, n: int, k: int,
                  scheme: str, placement: str, draw: int,
                  pool_size: int = plan.POOL_SIZE) -> tuple[list, dict]:
    """The anchor list [(a, u), ...] of n pairs, k of them correct, and a dict of
    facts about the draw. Deterministic in its arguments and the pair."""
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k} n={n}")
    entropy = [draw, n, k, plan.SCHEME_CODE[scheme], plan.PLACEMENT_CODE[placement]]
    rng = np.random.default_rng(entropy)

    pool_a, pool_u = _pools(Ga, Gu, placement, pool_size)
    if placement == "pool":
        pool_u_set = set(pool_u)
        eligible = [a for a in pool_a if a in truth and truth[a] in pool_u_set]
    else:
        eligible = [a for a in sorted(truth) if a in Ga and truth[a] in Gu]
    if k > len(eligible):
        raise ValueError(f"k={k} correct anchors asked for, but only {len(eligible)} "
                         f"pool nodes have their partner in the other pool")

    correct_a = _shuffled(rng, eligible)[:k]
    correct = [(a, truth[a]) for a in correct_a]

    w = n - k
    taken_a = set(correct_a)
    taken_u = {u for _, u in correct}
    wrong_a_pool = [a for a in pool_a if a not in taken_a]
    free_u = [u for u in pool_u if u not in taken_u]
    if w > len(wrong_a_pool) or w > len(free_u):
        raise ValueError(f"w={w} wrong anchors asked for, but {len(wrong_a_pool)} G^a "
                         f"and {len(free_u)} G^u pool nodes are unused")
    wrong_a = _shuffled(rng, wrong_a_pool)[:w]
    wrong = _wrong_partners(Ga, Gu, truth, wrong_a, _shuffled(rng, free_u), scheme)

    anchors = correct + wrong
    if (len(anchors) != n or len({a for a, _ in anchors}) != n
            or len({u for _, u in anchors}) != n):
        raise RuntimeError("anchor set is not a one-to-one list of n pairs")
    if any(truth.get(a) == u for a, u in wrong):
        raise RuntimeError("a wrong anchor is its node's true partner")
    gaps = [abs(Ga.degree(a) - Gu.degree(u)) for a, u in wrong]
    info = {"rng_entropy": entropy, "pool_a": len(pool_a), "pool_u": len(pool_u),
            "eligible": len(eligible),
            "wrong_abs_degree_gap_mean": float(np.mean(gaps)) if gaps else None}
    return anchors, info


# ---------------------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------------------

SWEEP_RE = re.compile(r"^\s*sweep\s+(\d+)\s+stage\s+(\d)\s+\+([\d,]+)\s+->\s+([\d,]+)\s+mapped",
                      re.M)


def parse_sweeps(text: str) -> dict:
    """What `propagate` printed, since it returns only the mapping: the number of
    sweeps, and how much each stage added. `stage0_added` is the direct check on
    the claim that candidates keyed on the anchors leave stage 0 nothing to do."""
    rows = [(int(a), int(b), int(c.replace(",", "")))
            for a, b, c, _ in SWEEP_RE.findall(text)]
    return {"sweeps": rows[-1][0] if rows else None,
            "sweeps_stage0": sum(1 for r in rows if r[1] == 0),
            "stage0_added": sum(r[2] for r in rows if r[1] == 0),
            "stage1_added": sum(r[2] for r in rows if r[1] == 1)}


def run_one(ctx: dict, spec: dict, index: int, total: int) -> dict:
    """One run of one task: build the anchors, propagate, score, record."""
    args, task = ctx["args"], ctx["task"]
    Ga, Gu, truth, stats = ctx["Ga"], ctx["Gu"], ctx["truth"], ctx["stats"]
    pair_dir = Path(ctx["pair_dir"])
    n, k = spec["n"], spec["k"]
    w, p = n - k, k / n
    draw = task["draw"]
    note = ((args.note + " ") if args.note else "") + (
        f"E2 block={task['block']} task={task['task_id']}: level {task['level']} "
        f"{task['pair']} ({task['mode']}) draw {draw}; n={n} k={k} w={w} p={p:.4g}; "
        f"scheme {spec['scheme']}, placement {spec['placement']}, "
        f"candidates {spec['candidates']}.")

    rec = None
    if args.record:
        rec = RunRecord(KIND, pair_dir.name, args.status, note, archive=Path(args.archive))
        # Config only: block and task id are in the note and argv, not here, so that
        # a rerun of one cell from another block has the same trailing run-id hash.
        rec.set(seed=draw, beta=stats["beta"], beta_nominal=stats.get("beta_nominal"),
                beta_mode=stats.get("beta_mode"), dataset=args.dataset,
                theta=args.theta, eccen_support=args.eccen_support,
                pool_size=args.pool_size, n=n, k=k, w=w, p=p,
                overlap=stats["overlap_ratio"], level=task["level"],
                scheme=spec["scheme"], placement=spec["placement"],
                candidate_mode=spec["candidates"], draw=draw,
                rng_entropy=[draw, n, k, plan.SCHEME_CODE[spec["scheme"]],
                             plan.PLACEMENT_CODE[spec["placement"]]],
                matchable_nodes=stats.get("matchable_nodes"),
                pair_stats=stats, pair_dir=str(pair_dir))
        rec.add_inputs(pair_dir / "G1.edgelist", pair_dir / "G2.edgelist",
                       pair_dir / "mapping.txt", pair_dir / "stats.json")
        rec.seal_id()

    tag = rec.run_id if rec else (f"unrecorded_d{draw}_n{n}_k{k}_{spec['scheme']}_"
                                  f"{spec['placement']}_{spec['candidates']}")
    out_dir = Path(args.out) / pair_dir.parent.name / pair_dir.name
    anchors_path = out_dir / f"anchors_{KIND}_{tag}.txt"
    mapping_path = out_dir / f"out_mapping_{KIND}_{tag}.txt"

    t0 = time.time()
    m: dict = {}
    result = {"n": n, "k": k, "spec": spec}
    exit_status, error = "ok", None
    with capture_stderr() as console:
        try:
            print(f"run {index}/{total}: {pair_dir.name} draw {draw}  n={n} k={k} w={w}  "
                  f"{spec['scheme']}/{spec['placement']}/{spec['candidates']}",
                  file=sys.stderr)
            anchors, info = build_anchors(Ga, Gu, truth, n, k, spec["scheme"],
                                          spec["placement"], draw, args.pool_size)
            score = evaluate_anchors(anchors, truth)
            if score["found"] != n or score["correct"] != k:
                raise RuntimeError(f"built {score['correct']}/{score['found']} correct "
                                   f"anchors, wanted {k}/{n}")
            out_dir.mkdir(parents=True, exist_ok=True)
            anchors_path.write_text("".join(f"{a} {u}\n" for a, u in anchors))
            m.update(anchors_found=score["found"], anchors_correct=score["correct"],
                     anchor_precision=score["precision"],
                     pool_eligible=info["eligible"],
                     wrong_abs_degree_gap_mean=info["wrong_abs_degree_gap_mean"],
                     anchors_sha256=sha256_file(anchors_path))
            print(f"  anchors {score['correct']}/{score['found']}  "
                  f"({info['eligible']} eligible of {info['pool_a']} pool nodes)",
                  file=sys.stderr)

            candidates = ({} if spec["candidates"] == "empty"
                          else {a: [u] for a, u in anchors})
            t_prop = time.time()
            mapping = propagate(Ga, Gu, anchors, candidates, theta=args.theta,
                                support=args.eccen_support, verbose=True)
            m["propagate_seconds"] = round(time.time() - t_prop, 1)
            final = evaluate(mapping, truth, stats.get("matchable_nodes"))
            mapping_path.write_text("".join(f"{a} {u}\n" for a, u in mapping.items()))
            m.update(mapped=final["mapped"], correct=final["correct"],
                     chi=final["chi"], gamma=final["gamma"],
                     mapping_sha256=sha256_file(mapping_path),
                     **parse_sweeps(console.getvalue()))
            print(f"  chi {final['chi']:.4f}  gamma {final['gamma']:.4f}  "
                  f"{final['correct']:,}/{final['mapped']:,} correct/mapped  "
                  f"{m.get('sweeps')} sweeps  {time.time() - t0:.0f}s", file=sys.stderr)
        except Terminated as exc:
            exit_status, error = "killed", f"Terminated: {exc}\n{traceback.format_exc()}"
            raise
        except KeyboardInterrupt:
            exit_status, error = "killed", traceback.format_exc()
            raise
        except BaseException:
            exit_status, error = "error", traceback.format_exc()
            raise
        finally:
            if rec:
                rec.add_outputs(*[p_ for p_ in (anchors_path, mapping_path) if p_.exists()])
                rec.metric(**m)
                rec.finish(exit_status, error, console.getvalue(), True)
                result["run_id"] = rec.run_id
    result.update(m, seconds=round(time.time() - t0, 1))
    return result


# ---------------------------------------------------------------------------------------
# One task
# ---------------------------------------------------------------------------------------

def parse_cells(text: str | None) -> set | None:
    if not text:
        return None
    cells = set()
    for part in text.split(","):
        try:
            n, k = part.split(":")
            cells.add((int(n), int(k)))
        except ValueError:
            raise SystemExit(f"--cells wants n:k pairs like 100:5,1000:200, got {part!r}")
    return cells


def run_task(args) -> int:
    tasks = plan.tasks(args.block)
    if not 0 <= args.task_id < len(tasks):
        print(f"--task-id must be in [0, {len(tasks)}); block {args.block} has "
              f"{len(tasks)} tasks", file=sys.stderr)
        return 2
    task = tasks[args.task_id]
    try:
        pair_dir, stats0 = plan.resolve_pair(args.pair_root, task["mode"], task["pair"])
    except LookupError as exc:
        print(f"{exc}\n(--pair-root should list the nominal and the divergence root)",
              file=sys.stderr)
        return 2
    bad = plan.check_overlap(task, stats0)
    if bad:
        print(f"{task['pair']}: {bad}", file=sys.stderr)
        return 2
    runs = task["runs"]
    cells = parse_cells(args.cells)
    if cells is not None:
        runs = [r for r in runs if (r["n"], r["k"]) in cells]
        if not runs:
            print(f"--cells {args.cells} matches no run of task {args.task_id}",
                  file=sys.stderr)
            return 2

    Ga, Gu, truth, stats = load_pair(Path(pair_dir))
    if len(truth) != stats.get("matchable_nodes"):
        print(f"{task['pair']}: mapping.txt has {len(truth)} pairs but stats.json says "
              f"{stats.get('matchable_nodes')} matchable nodes", file=sys.stderr)
        return 2
    print(f"E2 {args.block} task {args.task_id}: {len(runs)} runs on {pair_dir}  "
          f"overlap {stats['overlap_ratio']:.4f}  status {args.status}", file=sys.stderr)

    ctx = {"args": args, "task": task, "Ga": Ga, "Gu": Gu, "truth": truth,
           "stats": stats, "pair_dir": pair_dir}
    done, failed = [], []
    for i, spec in enumerate(runs, 1):
        try:
            done.append(run_one(ctx, spec, i, len(runs)))
        except Terminated:
            print("terminated; the record of the run in flight was written as 'killed'",
                  file=sys.stderr)
            return 143
        except KeyboardInterrupt:
            return 130
        except Exception as exc:       # recorded by run_one; the other cells still run
            failed.append((spec, exc))
            print(f"run {i}/{len(runs)} (n={spec['n']} k={spec['k']}) FAILED: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"\ntask {args.task_id} ({args.block}, {task['pair']}, draw {task['draw']}): "
          f"{len(done)} ok, {len(failed)} failed")
    for r in done:
        print(f"  n={r['n']:<5} k={r['k']:<4} chi={r['chi']:.4f}  mapped={r['mapped']:>6,}  "
              f"sweeps={r.get('sweeps')}  {r['seconds']:.0f}s  {r.get('run_id', '')}")
    return 1 if failed else 0


# ---------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------

def _synthetic_pair(seed: int = 1, n_nodes: int = 1500, m: int = 9000, keep: float = 0.9):
    """A small noisy pair with disjoint id spaces ("a<i>" against "u<perm i>"), so
    that a G^a/G^u mix-up cannot pass unnoticed (propagation.py's self-test does the same)."""
    G = nx.gnm_random_graph(n_nodes, m, seed=seed)
    edges = list(G.edges())

    def thin(tag):
        mask = np.random.default_rng([seed, tag]).random(len(edges)) < keep
        return [e for e, kept in zip(edges, mask) if kept]

    perm = np.random.default_rng(seed).permutation(n_nodes)
    Ga, Gu = nx.Graph(), nx.Graph()
    Ga.add_edges_from((f"a{u}", f"a{v}") for u, v in thin(1))
    Gu.add_edges_from((f"u{perm[u]}", f"u{perm[v]}") for u, v in thin(2))
    truth = {f"a{i}": f"u{perm[i]}" for i in range(n_nodes)
             if f"a{i}" in Ga and f"u{perm[i]}" in Gu}
    return Ga, Gu, truth


def self_test() -> bool:
    """Anchor construction and the control, on a synthetic pair; no dataset needed."""
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}",
              file=sys.stderr)
        ok = ok and bool(cond)

    assert plan.self_test() == 0
    Ga, Gu, truth = _synthetic_pair()
    ps = 300
    pa, pu = _pools(Ga, Gu, "pool", ps)
    pu_set = set(pu)

    anchors, info = build_anchors(Ga, Gu, truth, 100, 20, "uniform", "pool", 0, ps)
    wrong = [(a, u) for a, u in anchors if truth.get(a) != u]
    check("n and k are what was asked", len(anchors) == 100 and len(wrong) == 80
          and evaluate_anchors(anchors, truth)["correct"] == 20)
    check("one-to-one on both sides", len({a for a, _ in anchors}) == 100
          and len({u for _, u in anchors}) == 100)
    check("pool placement keeps every anchor in the pools",
          all(a in set(pa) and u in pu_set for a, u in anchors))
    check("correct anchors have their partner in the G^u pool",
          all(truth[a] in pu_set for a, u in anchors if truth.get(a) == u))
    check("a wrong anchor is never its node's partner",
          all(truth.get(a) != u for a, u in wrong))

    again, _ = build_anchors(Ga, Gu, truth, 100, 20, "uniform", "pool", 0, ps)
    other, _ = build_anchors(Ga, Gu, truth, 100, 20, "uniform", "pool", 1, ps)
    check("same cell, same anchors", again == anchors)
    check("another draw, other anchors", other != anchors)
    check("a cell's seed does not leak into its neighbour",
          build_anchors(Ga, Gu, truth, 100, 50, "uniform", "pool", 0, ps)[0][:20] != anchors[:20])

    full, _ = build_anchors(Ga, Gu, truth, 60, 60, "uniform", "pool", 0, ps)
    check("w = 0: every anchor correct", all(truth[a] == u for a, u in full))

    gap = {}
    for scheme in plan.SCHEMES:
        a_, i_ = build_anchors(Ga, Gu, truth, 200, 10, scheme, "pool", 0, ps)
        gap[scheme] = i_["wrong_abs_degree_gap_mean"]
        check(f"{scheme}: 190 wrong, none correct", evaluate_anchors(a_, truth)["correct"] == 10)
    check("degree-matched partners are closer in degree than uniform ones",
          gap["degree_matched"] < 0.5 * gap["uniform"],
          f"mean |dDeg| {gap['degree_matched']:.2f} against {gap['uniform']:.2f}")

    up, _ = build_anchors(Ga, Gu, truth, 100, 20, "uniform", "uniform", 0, ps)
    check("uniform placement leaves the pool", any(a not in set(pa) for a, _ in up)
          and evaluate_anchors(up, truth)["correct"] == 20)
    try:
        build_anchors(Ga, Gu, truth, 400, 300, "uniform", "pool", 0, ps)
        check("k above the eligible pool nodes is refused", False)
    except ValueError:
        check("k above the eligible pool nodes is refused", True)

    # The control, end to end on the small pair: empty against singleton candidates.
    anchors, _ = build_anchors(Ga, Gu, truth, 40, 40, "uniform", "pool", 0, ps)
    with capture_stderr() as buf_e:
        map_e = propagate(Ga, Gu, anchors, {}, verbose=True)
    with capture_stderr() as buf_s:
        map_s = propagate(Ga, Gu, anchors, {a: [u] for a, u in anchors}, verbose=True)
    sw = parse_sweeps(buf_e.getvalue())
    check("singleton and empty candidates give the same mapping, in the same order",
          list(map_e.items()) == list(map_s.items()))
    check("stage 0 added nothing", sw["stage0_added"] == 0 and parse_sweeps(
        buf_s.getvalue())["stage0_added"] == 0, str(sw))
    res = evaluate(map_e, truth, len(truth))
    check("40 correct anchors ignite on the small pair", res["chi"] > 0.5,
          f"chi {res['chi']:.3f}, {sw['sweeps']} sweeps")
    check("parse_sweeps reads the printed sweep lines", parse_sweeps(
        "  sweep   1  stage 0  +0 -> 1,000 mapped  1s\n  sweep   2  stage 1  +1,234 -> 2,234 mapped  9s\n")
          == {"sweeps": 2, "sweeps_stage0": 1, "stage0_added": 0, "stage1_added": 1234})

    print("all checks passed" if ok else "CHECKS FAILED", file=sys.stderr)
    return ok


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="E2 runner: Phase-3 ignition from synthetic anchor sets.")
    ap.add_argument("--block", choices=plan.GRID_BLOCKS,
                    help="Which block of the design; see ignition_plan.py.")
    ap.add_argument("--task-id", type=int, default=None,
                    help="Run exactly this array task (one instance, one n group).")
    ap.add_argument("--cells", default=None,
                    help="Only these runs of the task, as n:k,n:k. For a rerun or a probe.")
    ap.add_argument("--plan", action="store_true", help="Print the block's tasks and exit.")
    ap.add_argument("--pair-root", nargs="+", default=plan.default_roots(),
                    help="The existing pair roots (default runs/sweep_nominal and "
                         "runs/sweep_divergence). A pair is found by name and by the "
                         "beta_mode in its stats.json. The runner writes no pairs.")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "ignition_grid"),
                    help="Root for the output mappings and anchor lists.")
    ap.add_argument("--archive", default=str(ARCHIVE),
                    help="Where records go (default results/runs/).")
    ap.add_argument("--status", choices=("exploratory", "confirmatory"), default="exploratory",
                    help="Declare before the run. confirmatory only for the blocks the "
                         "spec's Covered records names, with the design's settings.")
    ap.add_argument("--note", default="", help="Goes in front of the generated note.")
    ap.add_argument("--dataset", default="hepth", choices=sorted(DATASETS),
                    help="Recorded only; the design is the same for any dataset.")
    ap.add_argument("--theta", type=float, default=plan.THETA)
    ap.add_argument("--eccen-support", default=plan.ECCEN_SUPPORT, choices=("all", "nonzero"))
    ap.add_argument("--pool-size", type=int, default=plan.POOL_SIZE)
    ap.add_argument("--no-record", action="store_true",
                    help="Skip the record. Debugging only; an unrecorded run cannot be reported.")
    ap.add_argument("--allow-unset-hashseed", action="store_true",
                    help="Run without PYTHONHASHSEED set. Only for a throwaway run.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    args.record = not args.no_record

    if args.self_test:
        return 0 if self_test() else 1
    if not args.block:
        ap.error("--block is required (or --self-test)")
    if args.plan:
        return plan.main(["--block", args.block, "--plan"])
    if args.task_id is None:
        ap.error("--task-id is required (or --plan)")

    args.pair_root = [plan.abs_root(r) for r in args.pair_root]
    args.out = plan.abs_root(args.out)
    args.archive = plan.abs_root(args.archive)
    if args.status == "confirmatory":
        if plan.COVERAGE[args.block] != "confirmatory":
            print(f"block {args.block} is not covered by a prediction (Covered records); "
                  f"it is exploratory. Refusing --status confirmatory.", file=sys.stderr)
            return 2
        if ((args.theta, args.eccen_support, args.pool_size)
                != (plan.THETA, plan.ECCEN_SUPPORT, plan.POOL_SIZE)):
            print("confirmatory runs use the design's theta, eccentricity support and pool "
                  "size; refusing --status confirmatory with other values.", file=sys.stderr)
            return 2
    elif plan.COVERAGE[args.block] == "confirmatory":
        print(f"note: block {args.block} is covered by confirmatory predictions; these "
              f"records are labelled exploratory because --status was not confirmatory.",
              file=sys.stderr)
    # An empty value is not a set value: PYTHONHASHSEED= passes a get() test and is
    # rejected by the interpreter itself (same check as sweep.py:285-294).
    if not os.environ.get("PYTHONHASHSEED") and not args.allow_unset_hashseed:
        print("PYTHONHASHSEED is not set. Set it (0 is conventional here) or pass "
              "--allow-unset-hashseed.", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _on_sigterm)
    return run_task(args)


if __name__ == "__main__":
    raise SystemExit(main())
