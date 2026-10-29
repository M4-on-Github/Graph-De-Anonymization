"""E2 plan: which runs exist, and how they are cut into SLURM array tasks.

Experiment Log 3, section "E2: Phase-3 ignition-threshold grid" (Design: grid,
Design: step E2.0, Covered records). Standard library only and Python 3.6
compatible on purpose: submit_ignition_grid.sh runs this on the head node, whose
python3 is 3.6 and has none of the project's dependencies (the same reason
scripts/moes-cluster-scripts/grid.py is written that way). The runner and the
audit import it, so an array's size and a task's content come from one place
and cannot disagree.

Blocks. Names are this file's, the counts are the spec's.

    block          runs   tasks  what                                         status
    base            432     54   18 instances x 24 cells, cut by n group      confirmatory
    scheme           54      9   degree-matched wrong anchors, n=1000         confirmatory
    control           6      2   singleton candidates, o in {1.0, 0.81}       confirmatory
    placement        54      9   uniform placement, n=1000                    exploratory
    lowoverlap       12      3   o=0.36 probe, n=1000                         exploratory
    control_rerun     6      2   NOT in the spec's 558. See below.            confirmatory-eligible
    audit_high       45      9   E2.0, pairs with o >= 0.7985 (5 measures)    confirmatory
    audit_low        65     13   E2.0, pairs with o <= 0.6985 (5 measures)    exploratory

base + scheme + control + placement + lowoverlap = 432 + 54 + 6 + 54 + 12 = 558
runs in 77 tasks. The audit is 110 records in 22 tasks (one per pair).

An instance is (overlap level, pair directory, draw index). The draw index is
the pair seed for the nominal levels (so the pair and the draw change together)
and 0, 1, 2 on the single pair for the two divergence levels, which is the
spec's "replicate" column.

A task is one instance and one "n group". Base-grid cells are grouped by the
smallest of n in {100, 300, 1000} that is >= the cell's n, so the w = 0 cells
(n = k = 20, 50, 200, 500) fall into groups 100, 100, 300, 1000 and the three
tasks of an instance hold 8, 7 and 9 runs: the spec's "7--9 runs each".

control_rerun. P2.6 asks for "a rerun of the same run" as well as singleton
against empty candidates, but the spec counts only 6 extra runs (the singleton
ones; the empty member of each pair is the base-grid run). control_rerun is the
same six cells with the empty dict, run again, so that the rerun clause has
runs behind it. It is not part of the 558 and is never in a default selection;
name it with --only. If a singleton mapping differs from its base-grid twin,
it says whether the difference is nondeterminism.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -- the design's fixed settings ------------------------------------------------------------
POOL_SIZE = 2000               # "the 2,000 highest-degree nodes of each graph"
THETA = 0.5                    # adv_gd.py:229, 448
ECCEN_SUPPORT = "all"          # adv_gd.py:230
SCHEMES = ("uniform", "degree_matched")
PLACEMENTS = ("pool", "uniform")
CANDIDATE_MODES = ("empty", "singleton")
# Integers, not strings, in the RNG entropy: hash() of a str is salted, and the
# draw must not depend on anything but the cell.
SCHEME_CODE = {"uniform": 0, "degree_matched": 1}
PLACEMENT_CODE = {"pool": 0, "uniform": 1}

# -- overlap levels (spec: "Overlap levels (six; o from stats.json)", plus the low probe) ----
LEVEL_ORDER = ("1.0", "0.8994", "0.81", "0.7985", "0.64", "0.49", "0.36")
LEVELS = {
    "1.0":    {"overlap": 1.0,    "mode": "nominal",
               "pairs": ["cit-HepTh_b00_s0", "cit-HepTh_b00_s1", "cit-HepTh_b00_s2"]},
    "0.8994": {"overlap": 0.8994, "mode": "divergence", "pairs": ["cit-HepTh_b05_s0"] * 3},
    "0.81":   {"overlap": 0.81,   "mode": "nominal",
               "pairs": ["cit-HepTh_b10_s0", "cit-HepTh_b10_s1", "cit-HepTh_b10_s2"]},
    "0.7985": {"overlap": 0.7985, "mode": "divergence", "pairs": ["cit-HepTh_b11_s0"] * 3},
    "0.64":   {"overlap": 0.64,   "mode": "nominal",
               "pairs": ["cit-HepTh_b20_s0", "cit-HepTh_b20_s1", "cit-HepTh_b20_s2"]},
    "0.49":   {"overlap": 0.49,   "mode": "nominal",
               "pairs": ["cit-HepTh_b30_s0", "cit-HepTh_b30_s1", "cit-HepTh_b30_s2"]},
    "0.36":   {"overlap": 0.36,   "mode": "nominal",
               "pairs": ["cit-HepTh_b40_s0", "cit-HepTh_b40_s1", "cit-HepTh_b40_s2"]},
}
REPS = (0, 1, 2)
# The level's stated overlap against the pair's measured overlap_ratio. The
# measured values lie within 0.0014 of the stated ones (0.8086--0.8108 for
# "0.81", and so on); a wrong --pair-root misses by far more.
OVERLAP_TOL = 0.005

# -- cells ----------------------------------------------------------------------------------
N_GROUPS = (100, 300, 1000)
BASE_ROWS = ((100, (2, 5, 10, 20, 50, 100)),
             (300, (5, 10, 20, 50, 100, 200)),
             (1000, (2, 5, 10, 20, 50, 100, 200, 500)))
W0_K = (20, 50, 200, 500)                  # the w = 0 row, n = k (100 is in the n=100 row)
BLOCK_K = (5, 10, 20, 50, 100, 200)        # scheme and placement blocks, n = 1000
LOW_K = (50, 100, 200, 500)                # low-overlap probe, n = 1000
CONTROL_CELLS = ((100, 10), (1000, 50), (300, 100))

BASE_LEVELS = ("1.0", "0.8994", "0.81", "0.7985", "0.64", "0.49")
BLOCK_LEVELS = ("1.0", "0.81", "0.49")     # scheme and placement blocks
CONTROL_LEVELS = ("1.0", "0.81")
LOW_LEVELS = ("0.36",)

GRID_BLOCKS = ("base", "scheme", "control", "placement", "lowoverlap", "control_rerun")
AUDIT_BLOCKS = ("audit_high", "audit_low")
ALL_BLOCKS = GRID_BLOCKS + AUDIT_BLOCKS

# Which label the spec gives each block ("Covered records"). The runner and the
# submit script refuse --status confirmatory for anything not covered.
COVERAGE = {
    "base": "confirmatory", "scheme": "confirmatory", "control": "confirmatory",
    "placement": "exploratory", "lowoverlap": "exploratory",
    "control_rerun": "confirmatory",
    "audit_high": "confirmatory", "audit_low": "exploratory",
}
# What a call selects when it does not say --only. control_rerun is never in it.
DEFAULT_BLOCKS = {
    "confirmatory": ("base", "scheme", "control", "audit_high"),
    "exploratory": ("placement", "lowoverlap", "audit_low"),
}

MEASURES = ("degree", "eigenvector", "k_core", "k_truss", "pagerank")   # sorted(MEASURES)
# The 22 pairs with centrality_rank records (16 nominal, 6 divergence). b00_s0
# exists under both roots and has records under both. Group by overlap: the
# spec's "o >= 0.7985" and "o <= 0.6985" are rounded labels (b11_s0 measures
# 0.798498), so the split is made in the gap, at 0.75.
AUDIT_PAIRS = {
    "high": [("nominal", "cit-HepTh_b00_s0"), ("nominal", "cit-HepTh_b00_s1"),
             ("nominal", "cit-HepTh_b00_s2"), ("divergence", "cit-HepTh_b00_s0"),
             ("divergence", "cit-HepTh_b05_s0"),
             ("nominal", "cit-HepTh_b10_s0"), ("nominal", "cit-HepTh_b10_s1"),
             ("nominal", "cit-HepTh_b10_s2"), ("divergence", "cit-HepTh_b11_s0")],
    "low": [("divergence", "cit-HepTh_b16_s0"), ("divergence", "cit-HepTh_b23_s0"),
            ("divergence", "cit-HepTh_b29_s0"),
            ("nominal", "cit-HepTh_b20_s0"), ("nominal", "cit-HepTh_b20_s1"),
            ("nominal", "cit-HepTh_b20_s2"),
            ("nominal", "cit-HepTh_b30_s0"), ("nominal", "cit-HepTh_b30_s1"),
            ("nominal", "cit-HepTh_b30_s2"),
            ("nominal", "cit-HepTh_b40_s0"), ("nominal", "cit-HepTh_b40_s1"),
            ("nominal", "cit-HepTh_b40_s2"),
            ("nominal", "cit-HepTh_b50_s0")],
}
AUDIT_SPLIT = 0.75        # overlap >= this: group "high"; the low group is <= 0.70


def n_group(n):
    """The task group of a cell: the smallest of 100, 300, 1000 that is >= n."""
    for g in N_GROUPS:
        if n <= g:
            return g
    raise ValueError("n={} is above the largest group".format(n))


def base_cells():
    """The 24 (n, k) cells: three rows, then the w = 0 row."""
    cells = []
    for n, ks in BASE_ROWS:
        cells += [(n, k) for k in ks]
    cells += [(k, k) for k in W0_K]
    return cells


def _instances(level_keys, reps):
    out = []
    for key in level_keys:
        lv = LEVELS[key]
        for r in reps:
            out.append({"level": key, "overlap": lv["overlap"], "mode": lv["mode"],
                        "pair": lv["pairs"][r], "draw": r})
    return out


def _run(n, k, scheme, placement, candidates):
    return {"n": n, "k": k, "scheme": scheme, "placement": placement,
            "candidates": candidates}


# block -> (levels, replicates, cells, scheme, placement, candidates)
_ONE_TASK_PER_INSTANCE = {
    "scheme": (BLOCK_LEVELS, REPS, [(1000, k) for k in BLOCK_K],
               "degree_matched", "pool", "empty"),
    "placement": (BLOCK_LEVELS, REPS, [(1000, k) for k in BLOCK_K],
                  "uniform", "uniform", "empty"),
    "control": (CONTROL_LEVELS, (0,), list(CONTROL_CELLS), "uniform", "pool", "singleton"),
    "control_rerun": (CONTROL_LEVELS, (0,), list(CONTROL_CELLS), "uniform", "pool", "empty"),
    "lowoverlap": (LOW_LEVELS, REPS, [(1000, k) for k in LOW_K], "uniform", "pool", "empty"),
}


def tasks(block):
    """The array tasks of a block, in task-id order. Each is a dict: block,
    task_id, level, overlap, mode, pair, draw, group (None unless base), runs."""
    out = []
    if block == "base":
        for inst in _instances(BASE_LEVELS, REPS):
            for g in N_GROUPS:
                runs = [_run(n, k, "uniform", "pool", "empty")
                        for (n, k) in base_cells() if n_group(n) == g]
                t = dict(inst)
                t.update(block=block, task_id=len(out), group=g, runs=runs)
                out.append(t)
        return out
    if block in _ONE_TASK_PER_INSTANCE:
        levels, reps, cells, scheme, placement, cand = _ONE_TASK_PER_INSTANCE[block]
        for inst in _instances(levels, reps):
            t = dict(inst)
            t.update(block=block, task_id=len(out), group=None,
                     runs=[_run(n, k, scheme, placement, cand) for (n, k) in cells])
            out.append(t)
        return out
    if block in AUDIT_BLOCKS:
        cls = block.split("_")[1]
        for mode, pair in AUDIT_PAIRS[cls]:
            out.append({"block": block, "task_id": len(out), "overlap_class": cls,
                        "mode": mode, "pair": pair,
                        "runs": [{"measure": m} for m in MEASURES]})
        return out
    raise ValueError("unknown block {!r}; one of {}".format(block, ", ".join(ALL_BLOCKS)))


def n_tasks(block):
    return len(tasks(block))


def n_runs(block):
    return sum(len(t["runs"]) for t in tasks(block))


def describe(t):
    if t["block"] in AUDIT_BLOCKS:
        return "task {:2d}  {}  {}  ({} measures)".format(
            t["task_id"], t["mode"], t["pair"], len(t["runs"]))
    cells = " ".join("{}:{}".format(r["n"], r["k"]) for r in t["runs"])
    r0 = t["runs"][0]
    return ("task {:2d}  o={:<6} {:<10} {} draw {}  group {}  {} runs "
            "[{}/{}/{}]  cells {}".format(
                t["task_id"], t["level"], t["mode"], t["pair"], t["draw"],
                t["group"] if t["group"] else "-", len(t["runs"]),
                r0["scheme"], r0["placement"], r0["candidates"], cells))


# -- pairs ----------------------------------------------------------------------------------
def default_roots():
    return [os.path.join(REPO_ROOT, "runs", "sweep_nominal"),
            os.path.join(REPO_ROOT, "runs", "sweep_divergence")]


def abs_root(path):
    """A relative root is relative to the repo, as sweep.py treats its pair paths."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def resolve_pair(roots, mode, name):
    """(pair_dir, stats) for the one directory called `name` under `roots` whose
    stats.json says beta_mode == mode. The mode is read from the pair, not from
    the root's name: cit-HepTh_b00_s0 exists under both roots."""
    hits, seen = [], set()
    for root in roots:
        d = os.path.join(abs_root(root), name)
        sp = os.path.join(d, "stats.json")
        real = os.path.realpath(d)
        if real in seen or not os.path.isfile(sp):
            continue
        seen.add(real)
        with open(sp) as fh:
            stats = json.load(fh)
        if stats.get("beta_mode") == mode:
            hits.append((d, stats))
    if len(hits) != 1:
        raise LookupError("{} pair {!r}: found {} matching directories under {}"
                          .format(mode, name, len(hits), [abs_root(r) for r in roots]))
    return hits[0]


def check_overlap(task, stats):
    """Return an error string, or '' if the pair's overlap is the task's."""
    o = stats["overlap_ratio"]
    if task["block"] in AUDIT_BLOCKS:
        if task["overlap_class"] == "high" and o < AUDIT_SPLIT:
            return "overlap {:.4f} is below {} but the pair is in the high group".format(o, AUDIT_SPLIT)
        if task["overlap_class"] == "low" and o > 0.70:
            return "overlap {:.4f} is above 0.70 but the pair is in the low group".format(o)
        return ""
    if abs(o - task["overlap"]) > OVERLAP_TOL:
        return ("pair overlap {:.4f} is not the level's {} (tolerance {})"
                .format(o, task["overlap"], OVERLAP_TOL))
    return ""


def check_pairs(blocks, roots):
    """Resolve and check every pair the blocks need. Returns the error strings."""
    errors, seen = [], set()
    for b in blocks:
        for t in tasks(b):
            key = (t["mode"], t["pair"])
            if key in seen:
                continue
            seen.add(key)
            try:
                _, stats = resolve_pair(roots, t["mode"], t["pair"])
            except LookupError as exc:
                errors.append(str(exc))
                continue
            msg = check_overlap(t, stats)
            if msg:
                errors.append("{} {}: {}".format(t["mode"], t["pair"], msg))
    return errors


# -- CLI ------------------------------------------------------------------------------------
def summary():
    lines = ["{:<14} {:>5} {:>6}  {:<13} {}".format("block", "runs", "tasks", "status", "default call")]
    total_runs = total_tasks = 0
    for b in ALL_BLOCKS:
        dflt = [s for s, bl in sorted(DEFAULT_BLOCKS.items()) if b in bl]
        lines.append("{:<14} {:>5} {:>6}  {:<13} {}".format(
            b, n_runs(b), n_tasks(b), COVERAGE[b], dflt[0] if dflt else "(--only only)"))
        if b in ("base", "scheme", "control", "placement", "lowoverlap"):
            total_runs += n_runs(b)
            total_tasks += n_tasks(b)
    lines.append("grid blocks counted in the spec's 558: {} runs in {} tasks".format(total_runs, total_tasks))
    return "\n".join(lines)


def self_test():
    """The plan algebra against the spec's counts. Runs under python 3.6."""
    expect = {"base": (54, 432), "scheme": (9, 54), "control": (2, 6),
              "placement": (9, 54), "lowoverlap": (3, 12), "control_rerun": (2, 6),
              "audit_high": (9, 45), "audit_low": (13, 65)}
    for b, (nt, nr) in sorted(expect.items()):
        assert (n_tasks(b), n_runs(b)) == (nt, nr), (b, n_tasks(b), n_runs(b))
    assert sum(n_runs(b) for b in ("base", "scheme", "control", "placement", "lowoverlap")) == 558
    assert sum(n_tasks(b) for b in ("base", "scheme", "control", "placement", "lowoverlap")) == 77
    assert n_runs("audit_high") + n_runs("audit_low") == 110
    assert n_tasks("audit_high") + n_tasks("audit_low") == 22
    assert len(base_cells()) == 24 and len(set(base_cells())) == 24
    sizes = sorted(set(len(t["runs"]) for t in tasks("base")))
    assert sizes == [7, 8, 9], sizes
    assert len(set((t["mode"], t["pair"], t["draw"]) for t in tasks("base"))) == 18
    # every base cell appears exactly once per instance
    per = {}
    for t in tasks("base"):
        for r in t["runs"]:
            per.setdefault((t["pair"], t["mode"], t["draw"]), []).append((r["n"], r["k"]))
    assert all(sorted(v) == sorted(base_cells()) for v in per.values())
    # the scheme/placement/control cells are base cells, so the comparisons have a base twin
    bc = set(base_cells())
    for b in ("scheme", "placement", "control", "control_rerun"):
        assert all((r["n"], r["k"]) in bc for t in tasks(b) for r in t["runs"]), b
    assert all(len(set(t["pair"] for t in tasks(b))) >= 1 for b in GRID_BLOCKS)
    assert set(COVERAGE) == set(ALL_BLOCKS)
    for status, blocks in DEFAULT_BLOCKS.items():
        assert all(COVERAGE[b] == status for b in blocks), (status, blocks)
    assert "control_rerun" not in DEFAULT_BLOCKS["confirmatory"] + DEFAULT_BLOCKS["exploratory"]
    assert [t["task_id"] for t in tasks("base")] == list(range(54))
    print("ignition_plan self-test passed: " + "; ".join(
        "{} {}t/{}r".format(b, n_tasks(b), n_runs(b)) for b in ALL_BLOCKS))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="E2 plan (stdlib only).")
    ap.add_argument("--block", choices=ALL_BLOCKS)
    ap.add_argument("--count", action="store_true", help="print the number of array tasks")
    ap.add_argument("--runs", action="store_true", help="print the number of runs (audit: records)")
    ap.add_argument("--plan", action="store_true", help="print the tasks")
    ap.add_argument("--coverage", action="store_true",
                    help="print the status the spec gives the block")
    ap.add_argument("--default-blocks", choices=("confirmatory", "exploratory"),
                    help="print the blocks a call with this --status selects")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--check-pairs", action="store_true",
                    help="resolve every pair of --block (or of every block) and check its overlap")
    ap.add_argument("--pair-root", nargs="+", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.summary:
        print(summary())
        return 0
    if args.default_blocks:
        print(" ".join(DEFAULT_BLOCKS[args.default_blocks]))
        return 0
    if args.check_pairs:
        roots = args.pair_root or default_roots()
        errors = check_pairs([args.block] if args.block else ALL_BLOCKS, roots)
        for e in errors:
            print("ERROR: " + e, file=sys.stderr)
        return 1 if errors else 0
    if not args.block:
        ap.error("--block is required (or --summary, --default-blocks, --self-test)")
    if args.coverage:
        print(COVERAGE[args.block])
    elif args.count:
        print(n_tasks(args.block))
    elif args.runs:
        print(n_runs(args.block))
    else:
        ts = tasks(args.block)
        print("block {}: {} tasks, {} runs, status {}".format(
            args.block, len(ts), sum(len(t["runs"]) for t in ts), COVERAGE[args.block]))
        for t in ts:
            print("  " + describe(t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
