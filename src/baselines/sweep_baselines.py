"""Multi-cell driver for the two classic baselines: beta x pair seed.

Mirrors `adv_gd/sweep.py`'s shape (same CLI conventions, same --plan/
--generate/--task-id flow, same pair-generation delegation) but is deliberately
simpler in one respect: neither baseline here has anything analogous to Adv-GD's
alignment seed -- structural-fingerprint matching and IsoRank's recurrence are
both fully deterministic given a pair, so the cell grid is beta x pair_seed
only, not beta x pair_seed x align_seed. One cell = one (pair, method) run.

Reuses `adv_gd.sweep`'s pair-math (`divergence_beta`, `pair_name`, `PARENTS`,
`generate`) directly rather than duplicating it: unlike `grid.py` (Report 1's
own planner, which deliberately duplicates the beta math so that module has no
import path into the GPU code), there is no such constraint here -- these
baselines already import GPU-adjacent code transitively through
`baselines.common` (which imports `adv_gd.propagation`/`alignment`, themselves
torch imports, though neither baseline ever touches a GPU). Duplicating the
beta math again would just be a second place for it to drift.

    python src/baselines/sweep_baselines.py --method structural_fingerprint --plan
    python src/baselines/sweep_baselines.py --method structural_fingerprint --generate
    PYTHONHASHSEED=0 python src/baselines/sweep_baselines.py --method isorank --task-id 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src" / "adv_gd"))

from sweep import PARENTS, divergence_beta, pair_name, generate as _generate  # noqa: E402
from common import DATASETS  # noqa: E402
import structural_fingerprint  # noqa: E402
import isorank  # noqa: E402

METHODS = {"structural_fingerprint": structural_fingerprint, "isorank": isorank}


def build_plan(args) -> list[dict]:
    """beta x pair_seed only -- see module docstring for why there is no
    third dimension here.
    """
    stem = PARENTS[args.dataset].stem.replace(".", "_")
    root = Path(args.out)
    cells = []
    for beta in args.betas:
        gen_beta = divergence_beta(beta) if args.beta_mode == "divergence" else beta
        for pair_seed in args.pair_seeds:
            name = pair_name(stem, gen_beta, pair_seed)
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
            })
    return cells


def run_cell(cell: dict, args) -> int:
    pair = Path(cell["pair"])
    if not pair.is_absolute():
        pair = REPO_ROOT / pair
    if not (pair / "stats.json").exists():
        print(f"pair {cell['pair']} does not exist; run --generate first",
              file=sys.stderr)
        return 2

    mod = METHODS[args.method]
    note = (args.note or
            f"baseline sweep task {cell['task_id']} ({args.method}): "
            f"nominal beta={cell['beta_nominal']} ({cell['beta_mode']}, "
            f"generated at {cell['beta_generated']})")

    kwargs = dict(phases=args.phases, theta=args.theta,
                 eccen_support=args.eccen_support, status=args.status,
                 note=note, record=not args.no_record)
    if args.method == "isorank":
        kwargs.update(m=args.isorank_m, alpha=args.isorank_alpha,
                     rounds=args.isorank_rounds)

    t0 = time.time()
    out = mod.run(pair, cell["dataset"], **kwargs)
    print(f"task {cell['task_id']} ({args.method}) done in {time.time() - t0:.0f}s  "
          f"chi={out.get('chi')}  anchors={out.get('anchors_correct')}/"
          f"{out.get('anchors_found')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Multi-cell driver for the classic baselines.")
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--dataset", default="hepth", choices=sorted(PARENTS))
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.0, 0.05, 0.11, 0.16, 0.23, 0.29])
    ap.add_argument("--beta-mode", choices=("nominal", "divergence"),
                    default="divergence")
    ap.add_argument("--pair-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--eccen-support", default="all", choices=("all", "nonzero"))
    ap.add_argument("--phases", default="123", choices=("12", "123"))
    ap.add_argument("--isorank-m", type=int, default=2500,
                    help="isorank only: top-m degree node restriction per side.")
    ap.add_argument("--isorank-alpha", type=float, default=0.6)
    ap.add_argument("--isorank-rounds", type=int, default=20)
    ap.add_argument("--out", default=None,
                    help="Pair root. Default runs/sweep_divergence or "
                         "runs/sweep_nominal depending on --beta-mode -- "
                         "the SAME root adv_gd/sweep.py uses, so pairs are "
                         "shared across Adv-GD and these baselines.")
    ap.add_argument("--status", choices=("exploratory", "confirmatory"),
                    default="exploratory")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.out is None:
        args.out = f"runs/sweep_{args.beta_mode if args.beta_mode == 'nominal' else 'divergence'}"

    if args.self_test:
        return 0 if _self_test() else 1

    cells = build_plan(args)

    if args.plan or (not args.generate and args.task_id is None):
        print(f"plan: {len(cells)} cells, {args.method}, beta-major")
        for c in cells:
            print(f"  {c['task_id']:3d}  beta {c['beta_nominal']:.2f} -> "
                  f"pair {c['pair']}")
        if not args.generate and args.task_id is None:
            print("nothing to do: pass --plan, --generate or --task-id",
                  file=sys.stderr)
        if not args.plan:
            return 0

    if args.generate:
        return _generate(args, cells)

    if args.task_id is None:
        return 0
    if not 0 <= args.task_id < len(cells):
        print(f"--task-id must be in [0, {len(cells)}); the plan has "
              f"{len(cells)} cells", file=sys.stderr)
        return 2
    return run_cell(cells[args.task_id], args)


def _self_test() -> bool:
    ns = argparse.Namespace(dataset="hepth", betas=[0.0, 0.1], beta_mode="divergence",
                            pair_seeds=[0, 1], out="runs/sweep_divergence")
    cells = build_plan(ns)
    assert len(cells) == 4, f"expected 4 cells, got {len(cells)}"
    assert [c["task_id"] for c in cells] == list(range(4))
    # beta-major: cells 0-1 share pair_seed 0/1 at beta 0.0, cells 2-3 at beta 0.1
    assert cells[0]["beta_nominal"] == cells[1]["beta_nominal"] == 0.0
    assert cells[2]["beta_nominal"] == cells[3]["beta_nominal"] == 0.1
    print("sweep_baselines self-test passed", file=sys.stderr)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
