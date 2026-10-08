"""Multi-cell driver for the Adv-GD reproduction: beta x alignment seed.

`adv_gd.py` runs one pair at one seed. This builds the cross product, generates
the pairs it needs, and runs exactly one cell per process so that a cluster
array can own the parallelism. See the "Sweeping beta across seeds" section of
`adv_gd_protocol.md` for why each of the four design decisions below is what it
is; the short version:

- the GAE seed is pinned while the alignment seed varies, because Phase 2's
  bimodality is in the adversarial initialization and re-training Phase 1 per
  cell costs ~20 GPU-minutes to vary something the question is not about;
- one cell per process, so a death at cell 14 of 20 costs one cell;
- `--beta-mode divergence` generates at beta' = 1 - sqrt(1 - beta), which is
  what makes a cell comparable to the paper's Table 2;
- the two beta modes generate into separate roots, because pair directory names
  round beta to whole percent and would collide.

Nothing here computes a result. Every number comes out of `adv_gd.run`, and
every cell leaves a record under `results/runs/`.

    python src/adv_gd/sweep.py --plan
    python src/adv_gd/sweep.py --generate
    PYTHONHASHSEED=0 python src/adv_gd/sweep.py --task-id 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from adv_gd import DATASETS, GAE_EPOCHS, run  # noqa: E402

GENERATOR = REPO_ROOT / "src" / "beta_noise_generator" / "perturbation.py"
PARENTS = {
    "hepth": REPO_ROOT / "data" / "cit-HepTH" / "cit-HepTh.txt",
}


def divergence_beta(beta: float) -> float:
    """Per-copy deletion rate that makes the two copies diverge by `beta`.

    Independent deletion at rate b from each copy leaves an edge in both with
    probability (1-b)^2, so solving (1-b)^2 = 1-beta gives this. The inverse of
    the overlap prediction the generator already checks itself against.
    """
    return 1.0 - math.sqrt(1.0 - beta)


def pair_name(stem: str, gen_beta: float, pair_seed: int) -> str:
    """The directory name the generator will choose. Kept in sync by hand.

    `perturbation.py` composes this as f"{stem}_b{round(beta*100):02d}_s{seed}"
    and does not expose it as a function. If that format changes, --generate
    will still work and --task-id will stop finding its pair, loudly.
    """
    return f"{stem}_b{round(gen_beta * 100):02d}_s{pair_seed}"


def build_plan(args) -> list[dict]:
    """The cells, in a fixed order that --task-id indexes into.

    Ordered beta-major so that consecutive task ids share a pair, and therefore
    a Phase 1 embedding cache: cell 0 pays for the GAEs and cells 1..k-1 of the
    same beta read them. On a cluster that runs the array concurrently this
    saves nothing (they all miss the cache at once) and is still the right order
    for a serial loop, which is how a GPU-poor run of this will happen.
    """
    stem = PARENTS[args.dataset].stem.replace(".", "_")
    root = Path(args.out)
    cells = []
    for beta in args.betas:
        gen_beta = divergence_beta(beta) if args.beta_mode == "divergence" else beta
        for pair_seed in args.pair_seeds:
            name = pair_name(stem, gen_beta, pair_seed)
            for align_seed in args.align_seeds:
                cells.append({
                    "task_id": len(cells),
                    "dataset": args.dataset,
                    "beta_nominal": beta,
                    "beta_generated": round(gen_beta, 6),
                    "beta_mode": args.beta_mode,
                    "pair_seed": pair_seed,
                    "pair": str((root / name).relative_to(REPO_ROOT))
                            if (root / name).is_relative_to(REPO_ROOT)
                            else str(root / name),
                    "align_seed": align_seed,
                    "gae_seed": args.gae_seed,
                    "gae_epochs": args.gae_epochs,
                    "n_restarts": args.n_restarts,
                    "theta": args.theta,
                    "eccen_support": args.eccen_support,
                })
    return cells


def generate(args, cells: list[dict]) -> int:
    """Create every pair the plan needs, skipping pairs already present.

    Delegates to the generator as a subprocess rather than importing it, so that
    the pair gets the generator's own run record with its own argv rather than
    one that says the sweep made it. Each beta is a separate invocation because
    the generator names its output from beta and we need to know which pair
    directory corresponds to which cell.
    """
    want: dict[str, dict] = {}
    for c in cells:
        want.setdefault(c["pair"], c)

    rc = 0
    for pair, c in sorted(want.items()):
        target = REPO_ROOT / pair if not Path(pair).is_absolute() else Path(pair)
        if (target / "stats.json").exists():
            print(f"  have {pair}")
            continue
        cmd = [sys.executable, str(GENERATOR),
               "--parent", str(PARENTS[c["dataset"]]),
               "--beta", str(c["beta_generated"]),
               "--seed", str(c["pair_seed"]),
               "--out", str(args.out),
               "--status", "confirmatory",
               "--note", (f"sweep cell for nominal beta={c['beta_nominal']} "
                          f"in {c['beta_mode']} mode")]
        print("  " + " ".join(cmd[1:]))
        if args.dry_run:
            continue
        res = subprocess.run(cmd, cwd=REPO_ROOT)
        rc |= res.returncode
        if not (target / "stats.json").exists():
            print(f"  generator did not produce {pair}", file=sys.stderr)
            rc |= 1
    return rc


def run_cell(cell: dict, args) -> int:
    pair = Path(cell["pair"])
    if not pair.is_absolute():
        pair = REPO_ROOT / pair
    if not (pair / "stats.json").exists():
        print(f"pair {cell['pair']} does not exist; run --generate first",
              file=sys.stderr)
        return 2

    cfg = {"n_restarts": cell["n_restarts"]}
    note = (args.note or
            f"sweep task {cell['task_id']}: nominal beta={cell['beta_nominal']} "
            f"({cell['beta_mode']}, generated at {cell['beta_generated']}), "
            f"align seed {cell['align_seed']}, gae seed {cell['gae_seed']}")

    t0 = time.time()
    out = run(pair, cell["dataset"], seed=cell["align_seed"], cfg=cfg,
              theta=cell["theta"], eccen_support=cell["eccen_support"],
              gae_epochs=cell["gae_epochs"], gae_seed=cell["gae_seed"],
              status=args.status, note=note, record=not args.no_record)
    print(f"task {cell['task_id']} done in {time.time() - t0:.0f}s  "
          f"chi={out.get('chi')}  anchors={out.get('anchors_correct')}/"
          f"{out.get('anchors_found')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the Adv-GD beta x seed sweep, one cell per process.")
    ap.add_argument("--dataset", default="hepth", choices=sorted(PARENTS),
                    help="Only datasets with a parent graph on disk.")
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5],
                    help="Nominal noise levels, as the paper states them.")
    ap.add_argument("--beta-mode", choices=("nominal", "divergence"),
                    default="divergence",
                    help="nominal: delete at beta from each copy (copies "
                         "diverge by ~2*beta). divergence: delete at "
                         "1-sqrt(1-beta), so the copies diverge by beta. Only "
                         "divergence cells are comparable to Table 2.")
    ap.add_argument("--pair-seeds", type=int, nargs="+", default=[0],
                    help="Noise draws per beta. More than one is the only way "
                         "to get an interval over the beta axis.")
    ap.add_argument("--align-seeds", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="Phase 2 seeds. Phase 2 is bimodal; fewer than ~4 "
                         "cannot distinguish a mode from a bad draw.")
    ap.add_argument("--gae-seed", type=int, default=0,
                    help="Pinned, so Phase 1 is trained once per pair rather "
                         "than once per cell.")
    ap.add_argument("--gae-epochs", type=int, default=GAE_EPOCHS)
    ap.add_argument("--n-restarts", type=int, default=4,
                    help="Restarts inside each cell, selected by mean match "
                         "cosine. This is model selection, not replication.")
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--eccen-support", default="all", choices=("all", "nonzero"))
    ap.add_argument("--out", default=None,
                    help="Pair root. Defaults to runs/sweep_<beta-mode>/, "
                         "which must differ per mode: pair names round beta to "
                         "whole percent and would collide.")
    ap.add_argument("--status", choices=("exploratory", "confirmatory"),
                    default="exploratory",
                    help="A sweep is exploratory. Say confirmatory only when "
                         "re-running a cell against a stated prior expectation.")
    ap.add_argument("--note", default="",
                    help="Overrides the generated per-cell note in the record.")
    ap.add_argument("--no-record", action="store_true")

    ap.add_argument("--plan", action="store_true",
                    help="Print the cells and exit. Writes nothing.")
    ap.add_argument("--manifest", action="store_true",
                    help="With --plan, also write the plan as JSON under --out.")
    ap.add_argument("--generate", action="store_true",
                    help="Create the pairs the plan needs. Idempotent.")
    ap.add_argument("--dry-run", action="store_true",
                    help="With --generate, print the generator commands only.")
    ap.add_argument("--task-id", type=int, default=None,
                    help="Run exactly this cell of the plan and exit.")
    ap.add_argument("--allow-unset-hashseed", action="store_true",
                    help="Run a cell without PYTHONHASHSEED set. Only for a "
                         "throwaway run; the value belongs in the record.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.dataset not in DATASETS:
        print(f"{args.dataset} is not a known Adv-GD dataset", file=sys.stderr)
        return 2
    if args.out is None:
        args.out = str(REPO_ROOT / "runs" / f"sweep_{args.beta_mode}")

    if args.self_test:
        return _self_test()

    cells = build_plan(args)

    if args.plan or (not args.generate and args.task_id is None):
        print(f"{len(cells)} cells  mode={args.beta_mode}  out={args.out}")
        for c in cells:
            print(f"  {c['task_id']:3d}  beta {c['beta_nominal']:.2f} -> "
                  f"{c['beta_generated']:.4f}  {Path(c['pair']).name}  "
                  f"align {c['align_seed']}  gae {c['gae_seed']}")
        if args.manifest:
            root = Path(args.out)
            root.mkdir(parents=True, exist_ok=True)
            man = root / "sweep_plan.json"
            man.write_text(json.dumps(
                {"argv": sys.argv, "beta_mode": args.beta_mode,
                 "cells": cells}, indent=2))
            print(f"manifest {man}")
        if not args.generate and args.task_id is None and not args.plan:
            print("nothing to do: pass --plan, --generate or --task-id",
                  file=sys.stderr)
        return 0

    if args.generate:
        return generate(args, cells)

    if not 0 <= args.task_id < len(cells):
        print(f"--task-id must be in [0, {len(cells)}); the plan has "
              f"{len(cells)} cells", file=sys.stderr)
        return 2
    # An empty value is not a set value: PYTHONHASHSEED= passes a get()
    # test and is rejected by the interpreter itself.
    if not os.environ.get("PYTHONHASHSEED") and not args.allow_unset_hashseed:
        # Not superstition: the inherited baseline is hash-order sensitive and
        # this module was measured not to be. That measurement is only
        # meaningful beside the value it was taken under, and the record stores
        # whatever is set here.
        print("PYTHONHASHSEED is not set. Set it (0 is conventional here) or "
              "pass --allow-unset-hashseed.", file=sys.stderr)
        return 2
    return run_cell(cells[args.task_id], args)


def _self_test() -> int:
    """Checks the plan algebra and the collision argument. Runs no cell."""
    ns = argparse.Namespace(
        dataset="hepth", betas=[0.1, 0.2], beta_mode="divergence",
        pair_seeds=[0], align_seeds=[0, 1, 2, 3], gae_seed=0,
        gae_epochs=600, n_restarts=4, theta=0.5, eccen_support="all",
        out=str(REPO_ROOT / "runs" / "sweep_divergence"))
    cells = build_plan(ns)
    assert len(cells) == 8, len(cells)
    assert [c["task_id"] for c in cells] == list(range(8))
    # Beta-major: consecutive ids share a pair until the beta changes.
    assert len({c["pair"] for c in cells[:4]}) == 1
    assert cells[0]["pair"] != cells[4]["pair"]
    assert [c["align_seed"] for c in cells[:4]] == [0, 1, 2, 3]

    # The divergence transform, against the relation the generator checks.
    for beta in (0.1, 0.2, 0.3, 0.4, 0.5):
        b = divergence_beta(beta)
        assert abs((1 - b) ** 2 - (1 - beta)) < 1e-12, beta
    assert abs(divergence_beta(0.1) - 0.0513167) < 1e-6

    # Nominal and divergence must not be able to share a root: at beta=0.05
    # nominal and beta=0.1 divergence produce the same directory name.
    assert pair_name("cit-HepTh", 0.05, 0) == pair_name(
        "cit-HepTh", divergence_beta(0.1), 0) == "cit-HepTh_b05_s0"
    ns_nom = argparse.Namespace(**{**vars(ns), "beta_mode": "nominal",
                                   "out": str(REPO_ROOT / "runs" / "sweep_nominal")})
    nom = build_plan(ns_nom)
    assert not ({c["pair"] for c in cells} & {c["pair"] for c in nom}), \
        "the two modes must not write into one pair directory"
    assert nom[0]["beta_generated"] == 0.1

    # Pinning the GAE seed is the point; it must not track align_seed.
    assert {c["gae_seed"] for c in cells} == {0}

    assert PARENTS["hepth"].exists(), f"parent graph missing: {PARENTS['hepth']}"
    print(f"  plan {len(cells)} cells, beta-major, gae seed pinned")
    print(f"  divergence beta(0.1) = {divergence_beta(0.1):.6f}, "
          f"modes use disjoint roots")
    print("sweep self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
