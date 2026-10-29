"""E2.0: audit the anchor sets the centrality_rank baseline handed to propagate.

Experiment Log 3, section "E2: Phase-3 ignition-threshold grid", item "Design:
step E2.0". No record holds `centrality_rank`'s anchor precision, because
`centrality_rank_match.run` scores the rank-zip only after propagation. This
recomputes the anchors exactly as `centrality_rank_match.py:120-121` does, the
first min(n_anch, |mapping|) keys of `rank_mapping` (line 49; descending by the
measure, ties by node id, so the result is deterministic), logs n, k and
precision, and runs no `propagate`. One RunRecord of kind `anchor_audit` per
(pair, measure): n = metrics.anchors_found, k = metrics.anchors_correct,
precision = metrics.anchor_precision, overlap = the top-level `overlap` (the
pair's overlap_ratio), measure = `measure`. Join on (pair_dir, measure) to the
existing `baseline_centrality_<measure>` record for the outcome (P2.5).

A call is one array task: one pair, five measures (`--group G --task-id T`).
The 22 pairs with centrality_rank records are cut into two groups by overlap so
that they can be submitted, and labelled, separately:

    high   9 pairs, 45 records, overlap >= 0.7985 (the spec's label; the split
           is made at 0.75, because b11_s0 measures 0.798498)   confirmatory
    low   13 pairs, 65 records, overlap <= 0.6985                exploratory

`--status confirmatory` is refused for the low group: no prediction is written
for it. The anchor list of each record is written to
<out>/audit/<pair root>/<pair>/anchors_anchor_audit_<measure>_<run_id>.txt and
hashed into the record.

    PYTHONHASHSEED=0 python3 src/ignition_grid/anchor_audit.py --group high --task-id 0
    python3 src/ignition_grid/anchor_audit.py --group low --plan
    python3 src/ignition_grid/anchor_audit.py --self-test
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "adv_gd"))
sys.path.insert(0, str(REPO_ROOT / "src" / "baselines"))
sys.path.insert(0, str(HERE))

import networkx as nx            # noqa: E402

import ignition_plan as plan     # noqa: E402
from alignment import evaluate_anchors                                   # noqa: E402
from centrality import MEASURES                                          # noqa: E402
from centrality_rank_match import rank_mapping                           # noqa: E402
from common import DATASETS, load_pair                                   # noqa: E402
from provenance import ARCHIVE, RunRecord, capture_stderr, sha256_file   # noqa: E402

KIND = "anchor_audit"


class Terminated(BaseException):
    """Raised by the SIGTERM handler so that the record of the run in flight is written."""


def _on_sigterm(signum, frame):
    raise Terminated(f"signal {signum}")


def anchor_set(Ga: nx.Graph, Gu: nx.Graph, measure: str, n_anch: int) -> tuple[list, int]:
    """(anchors, |mapping|): what centrality_rank_match.py:120-121 builds.

    Keys of the rank-zip in RANK order, not re-sorted by id: the zip drops G^a's
    lowest-ranked nodes when G^a has more nodes than G^u (that file's comment)."""
    mapping_full = rank_mapping(Ga, Gu, measure)
    anchor_nodes = list(mapping_full.keys())[:min(n_anch, len(mapping_full))]
    return [(a, mapping_full[a]) for a in anchor_nodes], len(mapping_full)


def audit_one(ctx: dict, measure: str, index: int, total: int) -> dict:
    args, task = ctx["args"], ctx["task"]
    Ga, Gu, truth, stats = ctx["Ga"], ctx["Gu"], ctx["truth"], ctx["stats"]
    pair_dir = Path(ctx["pair_dir"])
    note = ((args.note + " ") if args.note else "") + (
        f"E2.0 block={task['block']} task={task['task_id']}: anchors of the "
        f"centrality_rank baseline, measure {measure}, first {args.n_anch} of the "
        f"rank-zip, on {task['pair']} ({task['mode']}); no propagate.")

    rec = None
    if args.record:
        rec = RunRecord(KIND, pair_dir.name, args.status, note, archive=Path(args.archive))
        rec.set(seed=0, beta=stats["beta"], beta_nominal=stats.get("beta_nominal"),
                beta_mode=stats.get("beta_mode"), dataset=args.dataset, measure=measure,
                n_anch=args.n_anch, overlap=stats["overlap_ratio"],
                matchable_nodes=stats.get("matchable_nodes"),
                pair_stats=stats, pair_dir=str(pair_dir))
        rec.add_inputs(pair_dir / "G1.edgelist", pair_dir / "G2.edgelist",
                       pair_dir / "mapping.txt", pair_dir / "stats.json")
        rec.seal_id()

    tag = rec.run_id if rec else f"unrecorded_{measure}"
    out_dir = Path(args.out) / "audit" / pair_dir.parent.name / pair_dir.name
    anchors_path = out_dir / f"anchors_{KIND}_{measure}_{tag}.txt"

    t0 = time.time()
    m: dict = {}
    result = {"measure": measure}
    exit_status, error = "ok", None
    with capture_stderr() as console:
        try:
            print(f"audit {index}/{total}: {pair_dir.name} ({task['mode']})  "
                  f"measure={measure}  overlap {stats['overlap_ratio']:.4f}", file=sys.stderr)
            anchors, mapping_len = anchor_set(Ga, Gu, measure, args.n_anch)
            score = evaluate_anchors(anchors, truth)
            out_dir.mkdir(parents=True, exist_ok=True)
            anchors_path.write_text("".join(f"{a} {u}\n" for a, u in anchors))
            m.update(anchors_found=score["found"], anchors_correct=score["correct"],
                     anchor_precision=score["precision"], rank_mapping_len=mapping_len,
                     anchors_sha256=sha256_file(anchors_path))
            print(f"  anchors {score['correct']}/{score['found']}  precision "
                  f"{score['precision']:.4f}  {time.time() - t0:.0f}s", file=sys.stderr)
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
                rec.add_outputs(*[p for p in (anchors_path,) if p.exists()])
                rec.metric(**m)
                rec.finish(exit_status, error, console.getvalue(), True)
                result["run_id"] = rec.run_id
    result.update(m, seconds=round(time.time() - t0, 1))
    return result


def run_task(args) -> int:
    block = f"audit_{args.group}"
    tasks = plan.tasks(block)
    if not 0 <= args.task_id < len(tasks):
        print(f"--task-id must be in [0, {len(tasks)}); group {args.group} has "
              f"{len(tasks)} pairs", file=sys.stderr)
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

    Ga, Gu, truth, stats = load_pair(Path(pair_dir))
    print(f"E2.0 {block} task {args.task_id}: {pair_dir}  overlap {stats['overlap_ratio']:.4f}"
          f"  measures {' '.join(args.measures)}  status {args.status}", file=sys.stderr)
    ctx = {"args": args, "task": task, "Ga": Ga, "Gu": Gu, "truth": truth,
           "stats": stats, "pair_dir": pair_dir}
    done, failed = [], []
    for i, measure in enumerate(args.measures, 1):
        try:
            done.append(audit_one(ctx, measure, i, len(args.measures)))
        except Terminated:
            print("terminated; the record of the audit in flight was written as 'killed'",
                  file=sys.stderr)
            return 143
        except KeyboardInterrupt:
            return 130
        except Exception as exc:       # recorded by audit_one; the other measures still run
            failed.append((measure, exc))
            print(f"audit {i}/{len(args.measures)} ({measure}) FAILED: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"\ntask {args.task_id} ({block}, {task['pair']}, {task['mode']}, overlap "
          f"{stats['overlap_ratio']:.4f}): {len(done)} ok, {len(failed)} failed")
    for r in done:
        print(f"  {r['measure']:<12} n={r['anchors_found']:<5} k={r['anchors_correct']:<5} "
              f"precision={r['anchor_precision']:.4f}  {r['seconds']:.0f}s  {r.get('run_id', '')}")
    return 1 if failed else 0


def self_test() -> bool:
    """The anchor recipe on a synthetic pair; no dataset needed."""
    import numpy as np
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}",
              file=sys.stderr)
        ok = ok and bool(cond)

    assert plan.self_test() == 0
    rng = np.random.default_rng(0)
    G = nx.gnm_random_graph(400, 2400, seed=3)
    G = nx.relabel_nodes(G, {i: str(i) for i in G})
    perm = rng.permutation(400)
    Gp = nx.relabel_nodes(G, {str(i): f"u{perm[i]}" for i in range(400)})
    truth = {str(i): f"u{perm[i]}" for i in range(400)}
    # G^a with 5 more nodes than G^u, as in a real noisy pair: the zip drops G^a's tail.
    Gb = Gp.copy()
    Gb.remove_nodes_from([f"u{perm[i]}" for i in range(5)])

    for measure in sorted(MEASURES):
        for n_anch, label in ((50, "n_anch below |mapping|"), (1000, "n_anch above |mapping|")):
            anchors, mlen = anchor_set(G, Gb, measure, n_anch)
            again, _ = anchor_set(G, Gb, measure, n_anch)
            full = rank_mapping(G, Gb, measure)
            sc = evaluate_anchors(anchors, truth)
            check(f"{measure}, {label}",
                  len(anchors) == min(n_anch, len(full)) and mlen == len(full)
                  and anchors == [(a, full[a]) for a in list(full)[:len(anchors)]]
                  and anchors == again and sc["found"] == len(anchors)
                  and 0 <= sc["correct"] <= sc["found"],
                  f"{sc['correct']}/{sc['found']}")
    print("all checks passed" if ok else "CHECKS FAILED", file=sys.stderr)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="E2.0: anchor audit of the centrality_rank baseline.")
    ap.add_argument("--group", choices=("high", "low"),
                    help="high: overlap >= 0.7985 (9 pairs). low: overlap <= 0.6985 (13 pairs).")
    ap.add_argument("--task-id", type=int, default=None,
                    help="The pair to audit: its index within the group.")
    ap.add_argument("--measures", nargs="+", choices=sorted(MEASURES), default=sorted(MEASURES))
    ap.add_argument("--plan", action="store_true", help="Print the group's pairs and exit.")
    ap.add_argument("--pair-root", nargs="+", default=plan.default_roots(),
                    help="The two existing pair roots; a pair is found by name and by the "
                         "beta_mode in its stats.json.")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "ignition_grid"),
                    help="Root for the anchor lists (under <out>/audit/).")
    ap.add_argument("--archive", default=str(ARCHIVE), help="Where records go (default results/runs/).")
    ap.add_argument("--status", choices=("exploratory", "confirmatory"), default="exploratory",
                    help="Declare before the run. confirmatory only for --group high.")
    ap.add_argument("--note", default="")
    ap.add_argument("--dataset", default="hepth", choices=sorted(DATASETS))
    ap.add_argument("--n-anch", type=int, default=None,
                    help="Anchors per set. Default: the dataset's N_anch (1000 for hepth), "
                         "which is what centrality_rank_match.run uses.")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--allow-unset-hashseed", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    args.record = not args.no_record

    if args.self_test:
        return 0 if self_test() else 1
    if not args.group:
        ap.error("--group is required (or --self-test)")
    if args.plan:
        return plan.main(["--block", f"audit_{args.group}", "--plan"])
    if args.task_id is None:
        ap.error("--task-id is required (or --plan)")
    if args.n_anch is None:
        args.n_anch = DATASETS[args.dataset]["n_anch"]

    args.pair_root = [plan.abs_root(r) for r in args.pair_root]
    args.out = plan.abs_root(args.out)
    args.archive = plan.abs_root(args.archive)
    block = f"audit_{args.group}"
    if args.status == "confirmatory" and plan.COVERAGE[block] != "confirmatory":
        print(f"group {args.group} has no prediction written for it (Covered records); it is "
              f"exploratory. Refusing --status confirmatory.", file=sys.stderr)
        return 2
    if args.status != "confirmatory" and plan.COVERAGE[block] == "confirmatory":
        print(f"note: group {args.group} is covered by P2.5; these records are labelled "
              f"exploratory because --status was not confirmatory.", file=sys.stderr)
    if not os.environ.get("PYTHONHASHSEED") and not args.allow_unset_hashseed:
        print("PYTHONHASHSEED is not set. Set it (0 is conventional here) or pass "
              "--allow-unset-hashseed.", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _on_sigterm)
    return run_task(args)


if __name__ == "__main__":
    raise SystemExit(main())
