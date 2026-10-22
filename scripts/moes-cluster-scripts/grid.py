#!/usr/bin/env python3
"""The Report 1 grid on cit-HepTh: the cells, and nothing else.

`docs/Project_Report_1_Seed_Based_Seed_Free_.pdf` reports three things on a
single graph pair with no noise parameter: seed-based propagation at one
training fraction and one eccentricity threshold (92.13% at 20% / theta=0.5),
six seed-free rankings scored on their own (Table 2), and the integration of
the top-k of a ranking back into the propagator (88.88% at top-50 PageRank).
Reimplementing it on generated HepTh pairs adds the axis the report does not
have -- noise -- and turns its two one-dimensional figures into cells.

This module owns the cross product and nothing else. It computes no result and
reads no graph. It exists so that the array size a submit script asks for and
the configuration a task resolves cannot disagree: the submit script freezes
the plan into a manifest, the array job reads cell `$SLURM_ARRAY_TASK_ID` out of
that same file, and editing a default here cannot retarget a queued job.

    python3 grid.py --kind pairs      --gen-argv
    python3 grid.py --kind seed_based --plan
    python3 grid.py --kind seed_free  --count
    python3 grid.py --kind seed_based --manifest /data/$USER/logs/sb.json

Three axis conventions, each of which is a choice and not the paper's:

- beta is a divergence by default (`--beta-mode divergence`): the pair is
  generated at 1-sqrt(1-beta) per copy so that the two released graphs differ
  by beta. `nominal` deletes at beta from each copy, so they differ by ~2*beta.
  The two modes write into separate roots because pair directory names round
  beta to whole percent and would collide -- the same argument as
  `src/adv_gd/sweep.py`.
- beta=0.0 is in the default sweep as a control. It is the cell that says
  whether the reimplementation reproduces the report at all; a pair at beta=0
  is the parent graph twice over, relabelled, so anything below ~90% there is a
  bug and not a finding.
- the training seed set is drawn with an explicit RNG seed, not taken as the
  first 20% of `mapping.txt`. The generator writes ground truth sorted by
  parent node id, i.e. by arXiv id, so a prefix slice is a structured seed set
  (arXiv-id order puts the 2000 papers first) rather than an arbitrary one.
  `--seed-draws -1` restores the prefix split that `src/seed_based.py`
  performs.
"""
import argparse
import json
import math
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PARENT = REPO_ROOT / "data" / "cit-HepTH" / "cit-HepTh.txt"
STEM = "cit-HepTh"

# node2vec is in the report's Table 2 and not in `src/seed_free.py`: the module
# dispatches on five method names and has no node2vec branch. It scored 0.01%
# there, by its own account because the all-pairs cosine did not fit the
# machine, so it is excluded here rather than reimplemented.
METHODS = ("pagerank", "eigen", "degree", "k-core", "k-truss")

ANCHOR_THETA = 0.5          # the report's eccentricity threshold
ANCHOR_TRAIN = 0.20         # the report's training fraction


def divergence_beta(beta):
    """Per-copy deletion rate that makes the two copies diverge by `beta`.

    Independent deletion at rate b from each copy leaves an edge in both with
    probability (1-b)^2; solving (1-b)^2 = 1-beta gives this. Identical to
    `src/adv_gd/sweep.py`, deliberately duplicated rather than imported so that
    this file has no import path into the GPU module.
    """
    return 1.0 - math.sqrt(1.0 - beta)


def generated_beta(beta, mode):
    return divergence_beta(beta) if mode == "divergence" else beta


def pair_name(gen_beta, pair_seed):
    """The directory name the generator will choose. Kept in sync by hand.

    `perturbation.py` composes this as f"{stem}_b{round(beta*100):02d}_s{seed}"
    and does not expose it as a function. If that format changes, generation
    still works and every cell stops finding its pair, loudly.
    """
    return f"{STEM}_b{round(gen_beta * 100):02d}_s{pair_seed}"


def default_root(mode):
    """Where the pairs live. Separate from adv_gd's roots for two reasons.

    The names collide across beta modes (nominal 0.05 and divergence 0.1 both
    round to b05), and these pairs are written with `--compat-layout`, which
    nests the four files one directory deeper than `src/adv_gd` expects. A
    shared root would hand one module the other's layout.
    """
    return REPO_ROOT / "runs" / f"report1_{mode}"


def _pairs(args):
    out = []
    for beta in args.betas:
        gen = generated_beta(beta, args.beta_mode)
        for ps in args.pair_seeds:
            out.append({"beta_nominal": beta, "beta_generated": round(gen, 6),
                        "pair_seed": ps, "pair": pair_name(gen, ps)})
    return out


def _cell_base(args, p):
    root = Path(args.out) if args.out else default_root(args.beta_mode)
    pair_dir = root / p["pair"]
    try:
        pair_rel = str(pair_dir.relative_to(REPO_ROOT))
    except ValueError:
        pair_rel = str(pair_dir)
    return {
        "dataset": "hepth",
        "beta_nominal": p["beta_nominal"],
        "beta_generated": p["beta_generated"],
        "beta_mode": args.beta_mode,
        "pair_seed": p["pair_seed"],
        "pair": p["pair"],
        # The directory the baseline's --path would point at: it appends
        # 'labeled_dev' itself, and --compat-layout is what puts it there.
        "pair_dir": pair_rel,
    }


def seed_based_cells(args):
    """Two slices through (train fraction x theta), not their full cross.

    The report varies one at a time around its anchor, and so do we: the cross
    would be 20 cells per pair to answer two one-dimensional questions. Slice A
    holds theta at the anchor and sweeps the training fraction (its Figure 1b);
    slice B holds the training fraction and sweeps theta (its Figure 1a). The
    anchor cell is in both and emitted once.
    """
    combos = []
    for tp in args.train_portions:
        combos.append((tp, ANCHOR_THETA))
    for th in args.thetas:
        if (ANCHOR_TRAIN, th) not in combos:
            combos.append((ANCHOR_TRAIN, th))

    cells = []
    for p in _pairs(args):
        for tp, th in combos:
            for draw in args.seed_draws:
                cells.append({**_cell_base(args, p), "kind": "seed_based",
                              "task_id": len(cells), "train_portion": tp,
                              "theta": th, "seed_draw": draw})
    return cells


def seed_free_cells(args):
    """One cell per (pair, ranking method); the top-k axis is inside the cell.

    A cell writes one record for the ranking alone and one per integrated
    top-k, which is 1 + len(top_ks) records from one centrality computation.
    Splitting top-k into its own task would recompute the ranking for each --
    four more k-truss decompositions of a 350k-edge graph to re-slice a list
    whose prefixes are already stable under top_k.
    """
    cells = []
    for p in _pairs(args):
        for method in args.methods:
            cells.append({**_cell_base(args, p), "kind": "seed_free",
                          "task_id": len(cells), "method": method,
                          "top_ks": list(args.top_ks),
                          "theta": ANCHOR_THETA})
    return cells


def build(args):
    if args.kind == "pairs":
        return _pairs(args)
    if args.kind == "seed_based":
        return seed_based_cells(args)
    return seed_free_cells(args)


def gen_argv(args):
    """The single `perturbation.py` invocation that makes every pair.

    One process, because loading the 350k-edge parent once and cutting 18 pairs
    from it is strictly cheaper than 18 processes that each load it, and
    because `--beta` and `--repeats` already express this grid exactly.
    `--beta-nominal` takes one value per `--beta`, so the same call also records
    on every pair which nominal beta and which mode produced it.
    """
    seeds = sorted(args.pair_seeds)
    if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
        raise SystemExit(
            f"--pair-seeds must be contiguous to generate in one call; got "
            f"{seeds}. The generator derives its seeds as --seed + 0..repeats-1. "
            f"Run it once per non-contiguous seed, or use a contiguous range.")
    betas = [f"{generated_beta(b, args.beta_mode):.6f}" for b in args.betas]
    nominal = [str(b) for b in args.betas]
    root = Path(args.out) if args.out else default_root(args.beta_mode)
    return ["--parent", str(PARENT),
            "--beta", *betas,
            "--beta-nominal", *nominal,
            "--beta-mode", args.beta_mode,
            "--seed", str(seeds[0]),
            "--repeats", str(len(seeds)),
            "--mode", args.gen_mode,
            "--out", str(root),
            "--compat-layout",
            "--status", "confirmatory",
            "--note", (f"Report 1 reimplementation grid on HepTh, "
                       f"{args.beta_mode} beta mode, nominal betas "
                       f"{' '.join(nominal)}")]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Enumerate the Report 1 cells on cit-HepTh.")
    ap.add_argument("--kind", choices=("pairs", "seed_based", "seed_free"),
                    help="Required except with --self-test.")
    ap.add_argument("--beta-mode", choices=("divergence", "nominal"),
                    default="divergence",
                    help="divergence: generate at 1-sqrt(1-beta) so the two "
                         "graphs differ by beta. nominal: delete at beta from "
                         "each copy. Separate roots; see default_root.")
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    help="0.0 is the control cell, not padding.")
    ap.add_argument("--pair-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Noise draws per beta. The report has one pair and so "
                         "no interval on the axis that matters most.")
    ap.add_argument("--train-portions", type=float, nargs="+",
                    default=[0.05, 0.10, 0.20, 0.40],
                    help="Fraction of ground truth given as seeds (report: 0.20).")
    ap.add_argument("--thetas", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7, 0.9],
                    help="Eccentricity thresholds (report: 0.5).")
    ap.add_argument("--seed-draws", type=int, nargs="+", default=[0],
                    help="RNG seed for choosing which ground-truth pairs are "
                         "given. -1 means the baseline's prefix slice.")
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=list(METHODS),
                    help="Seed-free rankings. node2vec is absent from "
                         "src/seed_free.py and so from here.")
    ap.add_argument("--top-ks", type=int, nargs="+", default=[25, 50, 100, 150],
                    help="Top-k of the ranking fed back as seeds (report: "
                         "25-150, best at 50).")
    ap.add_argument("--gen-mode", choices=("bernoulli", "exact"),
                    default="bernoulli",
                    help="Passed through to the generator's --mode.")
    ap.add_argument("--out", default=None,
                    help="Pair root. Defaults to runs/report1_<beta-mode>/.")

    ap.add_argument("--count", action="store_true",
                    help="Print the number of cells and exit. This is what a "
                         "submit script asks for the array size.")
    ap.add_argument("--plan", action="store_true", help="Print the cells.")
    ap.add_argument("--json", type=int, default=None, metavar="TASK_ID",
                    help="Print one cell as JSON.")
    ap.add_argument("--manifest", default=None, metavar="PATH",
                    help="Write the frozen plan here. An array job reads its "
                         "cell from this file, not from these defaults.")
    ap.add_argument("--gen-argv", action="store_true",
                    help="With --kind pairs: print the perturbation.py "
                         "arguments, shell-quoted, for one invocation.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.kind:
        ap.error("--kind is required (pairs, seed_based or seed_free)")

    if args.gen_argv:
        if args.kind != "pairs":
            ap.error("--gen-argv only applies to --kind pairs")
        print(" ".join(shlex.quote(a) for a in gen_argv(args)))
        return 0

    cells = build(args)

    if args.manifest:
        man = Path(args.manifest)
        man.parent.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps(
            {"kind": args.kind, "beta_mode": args.beta_mode,
             "argv": sys.argv, "n_cells": len(cells), "cells": cells},
            indent=2) + "\n")
        print(f"manifest {man}  ({len(cells)} cells)")

    if args.json is not None:
        if not 0 <= args.json < len(cells):
            print(f"--json must be in [0, {len(cells)})", file=sys.stderr)
            return 2
        print(json.dumps(cells[args.json], indent=2))
        return 0

    if args.count:
        print(len(cells))
        return 0

    if args.plan or not args.manifest:
        if args.kind == "pairs":
            for c in cells:
                print(f"  beta {c['beta_nominal']:.2f} -> "
                      f"{c['beta_generated']:.4f}  {c['pair']}")
        elif args.kind == "seed_based":
            for c in cells:
                print(f"  {c['task_id']:4d}  {c['pair']:24s}  "
                      f"train {c['train_portion']:.2f}  theta {c['theta']:.1f}  "
                      f"draw {c['seed_draw']}")
        else:
            for c in cells:
                print(f"  {c['task_id']:4d}  {c['pair']:24s}  "
                      f"{c['method']:8s}  top-k {c['top_ks']}")
        print(f"{len(cells)} cells")
    return 0


def _self_test():
    """Checks the plan algebra. Reads nothing and writes nothing."""
    def ns(**over):
        base = dict(kind="seed_based", beta_mode="divergence",
                    betas=[0.0, 0.1], pair_seeds=[0, 1],
                    train_portions=[0.05, 0.20], thetas=[0.3, 0.5],
                    seed_draws=[0], methods=list(METHODS),
                    top_ks=[25, 50],
                    gen_mode="bernoulli", out=None)
        base.update(over)
        return argparse.Namespace(**base)

    # The divergence transform, against the relation the generator checks.
    for beta in (0.1, 0.2, 0.3, 0.4, 0.5):
        b = divergence_beta(beta)
        assert abs((1 - b) ** 2 - (1 - beta)) < 1e-12, beta
    assert divergence_beta(0.0) == 0.0

    # Slices, not the cross: 2 train x 2 theta would be 4 per pair; the anchor
    # is shared, so it is 2 + 2 - 1 = 3.
    sb = seed_based_cells(ns())
    assert len(sb) == 4 * 3, len(sb)
    assert [c["task_id"] for c in sb] == list(range(len(sb)))
    per_pair = [(c["train_portion"], c["theta"]) for c in sb
                if c["pair"] == sb[0]["pair"]]
    assert len(per_pair) == len(set(per_pair)) == 3, per_pair
    assert (ANCHOR_TRAIN, ANCHOR_THETA) in per_pair

    # beta=0 is a real cell with a real directory name.
    assert pair_name(0.0, 0) == "cit-HepTh_b00_s0"
    assert pair_name(divergence_beta(0.1), 0) == "cit-HepTh_b05_s0"

    # The two modes must not be able to share a pair root: at nominal 0.05 and
    # divergence 0.1 the directory names are equal.
    assert pair_name(0.05, 0) == pair_name(divergence_beta(0.1), 0)
    assert default_root("nominal") != default_root("divergence")

    # Seed-free: one cell per (pair, method), top-k inside.
    sf = seed_free_cells(ns(kind="seed_free"))
    assert len(sf) == 4 * len(METHODS), len(sf)
    assert all(c["top_ks"] == [25, 50] for c in sf)

    # Pairs: one per (beta, seed), and one generator call covers them.
    pr = _pairs(ns(kind="pairs"))
    assert len(pr) == 4 and len({c["pair"] for c in pr}) == 4
    argv = gen_argv(ns(kind="pairs"))
    assert "--compat-layout" in argv and argv[argv.index("--repeats") + 1] == "2"

    # The mode and the nominal betas travel with the generated ones, one each.
    def values(argv, flag):
        i = argv.index(flag) + 1
        j = i
        while j < len(argv) and not argv[j].startswith("--"):
            j += 1
        return argv[i:j]
    assert values(argv, "--beta") == ["0.000000", "0.051317"]
    assert values(argv, "--beta-nominal") == ["0.0", "0.1"]
    assert values(argv, "--beta-mode") == ["divergence"]
    nom = gen_argv(ns(kind="pairs", beta_mode="nominal"))
    assert values(nom, "--beta") == ["0.000000", "0.100000"]
    assert values(nom, "--beta-nominal") == ["0.0", "0.1"]
    assert values(nom, "--beta-mode") == ["nominal"]
    try:
        gen_argv(ns(kind="pairs", pair_seeds=[0, 2]))
    except SystemExit:
        pass
    else:
        raise AssertionError("non-contiguous pair seeds must be refused")

    print(f"  seed-based {len(sb)} cells (slices, anchor emitted once)")
    print(f"  seed-free  {len(sf)} cells, {len(METHODS)} methods")
    print(f"  parent {'present' if PARENT.exists() else 'MISSING'}: {PARENT}")
    print("grid self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
