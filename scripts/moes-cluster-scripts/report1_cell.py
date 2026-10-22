#!/usr/bin/env python3
"""Run one cell of the Report 1 grid on a generated HepTh pair, and record it.

One process per cell, so a death at cell 14 of 144 costs one cell. The cell is
read out of a manifest frozen by the submit script (see `grid.py`), never from a
default that could have changed since the array was queued.

    python3 report1_cell.py --manifest /data/$USER/logs/sb.json --task-id 0
    python3 report1_cell.py --kind seed_based --task-id 0 --betas 0.1   # ad hoc

What is reused and what is not. The algorithms are the inherited baseline's,
imported and called directly: `seed_based.SeedFreeD` is the propagator of the
report's Algorithm 1, and `seed_free.determine_seed_pairs` is its five-way
ranking dispatch. Three things are not reused, each because the baseline's
`main()` fixes them in a way a sweep cannot:

- **the split.** `src/seed_based.py` hardcodes `train_portion = 0.20` and
  `ecce_threshold = 0.5` inside `main()`, below the argument parser, and takes
  the first 20% of `mapping.txt` as the given seeds. Both of the report's
  figures vary exactly those two numbers, so the sweep has to own the split.
  The draw is also shuffled by default: the generator writes ground truth
  sorted by parent arXiv id, so a prefix slice is a structured seed set (by
  arXiv id, which puts the 2000 papers first). `--seed-draws -1` restores the
  prefix.
- **the membership test.** The baseline keeps a ground-truth pair when
  `u in G1 and u in G2`, which looks the G1 label up in G2 -- the wrong graph.
  The generator relabels each side to compact ids 0..n-1, so on these pairs
  `u in G2` is nearly always true and the test is near-vacuous, not
  destructive: it differs from the right one by 0-28 pairs on the pairs
  checked (26,647 against 26,675 kept on one generated pair, 26,942 against
  26,948 of 27,770 on `labeled_dev`). Here the test is `u in G1 and v in G2`,
  which is what the loop means.
- **the scoring container.** `get_seed_pairs` does `tup in validation_pairs`
  against a list; at the full ranking that is 27k linear scans of a 27k-element
  list. A set is passed instead. Membership is the only operation performed on
  it, so nothing else changes.

Everything else -- score function, eccentricity test, iteration to a fixed
point, the accuracy definitions -- is called, not reimplemented, so that a
difference from the report is a difference in the data and not in the code.

One hazard in calling it, worth stating because it silently corrupts a metric
rather than raising: **`SeedFreeD` extends the seed list it is handed, in
place.** `operation()` returns the same list object it was given with the newly
accepted pairs appended, and `iterativeOperation` feeds that back in. A caller
that keeps a reference to the list it passed -- to count it, or to union it with
the result -- is reading a 608-element list where it passed 25. Everything here
passes `list(...)` and scores against a set frozen beforehand.

Every cell writes a record under `results/runs/` per
`docs/experiment_logging_policy.md`, including when it fails: a cell that dies
in propagation is data about where it got to. A seed-free cell writes one
record for the ranking alone (the report's Table 2) and one per integrated
top-k (its Figure 4), from a single centrality computation.
"""
import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

# matplotlib is imported at seed_free import time and `plot_first_k_elements_two_dicts`
# calls plt.show() from inside determine_seed_pairs. Agg makes that a no-op; without
# it the call can fail or block on a node with no display.
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "adv_gd"))

import networkx as nx                                        # noqa: E402

import grid                                                  # noqa: E402
from provenance import RunRecord, capture_stderr             # noqa: E402
from seed_based import SeedFreeD                             # noqa: E402


def resolve(path_str):
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def load_pair(pair_dir):
    """Load a `--compat-layout` pair: the four files sit in `labeled_dev/`.

    Returns (G1, G2, truth, stats). `truth` keeps the file's order, which is
    the generator's: sorted by parent node id.
    """
    d = Path(pair_dir) / "labeled_dev"
    missing = [n for n in ("G1.edgelist", "G2.edgelist", "mapping.txt",
                           "stats.json") if not (d / n).exists()]
    if missing:
        raise SystemExit(
            f"pair {pair_dir} is incomplete (missing {', '.join(missing)}).\n"
            f"Generate it first: see 00_generate_pairs_job.sh. Note the four "
            f"files must be in a subdirectory literally named 'labeled_dev', "
            f"which is what the generator's --compat-layout does.")

    G1 = nx.read_edgelist(str(d / "G1.edgelist"))
    G2 = nx.read_edgelist(str(d / "G2.edgelist"))
    truth = []
    dropped = 0
    with (d / "mapping.txt").open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            u, v = parts
            if u in G1 and v in G2:
                truth.append((u, v))
            else:
                dropped += 1
    stats = json.loads((d / "stats.json").read_text())
    if dropped:
        print(f"  {dropped} ground-truth pairs name a node absent from its "
              f"graph; kept {len(truth)}", file=sys.stderr)
    if not truth:
        raise SystemExit(
            f"no usable ground truth in {d / 'mapping.txt'}. If the file is "
            f"non-empty, check that it is 'G1-id G2-id' per line and that "
            f"G1.edgelist/G2.edgelist in this directory are the pair it "
            f"describes.")
    return G1, G2, truth, stats


def split_truth(truth, train_portion, seed_draw):
    """Choose which ground-truth pairs are given as seeds.

    `seed_draw >= 0` shuffles with that seed; `-1` takes the prefix, which is
    what `src/seed_based.py` does. The prefix is reproducible and structured,
    the shuffle reproducible and not; the record stores which was used.
    """
    k = int(len(truth) * train_portion)
    if k == 0:
        raise SystemExit(
            f"train_portion {train_portion} of {len(truth)} pairs rounds to "
            f"zero seeds; propagation from no seeds has nothing to do.")
    if seed_draw < 0:
        return truth[:k], truth[k:]
    order = list(truth)
    random.Random(seed_draw).shuffle(order)
    return order[:k], order[k:]


def _new_record(kind, cell, args, note, extra):
    """Build the record and seal its id before any artifact is written."""
    rec = RunRecord(kind, cell["pair"], args.status, note,
                    archive=args.archive)
    rec.set(dataset=cell["dataset"],
            beta=cell["beta_generated"],
            beta_nominal=cell["beta_nominal"],
            beta_mode=cell["beta_mode"],
            pair_seed=cell["pair_seed"],
            pair_dir=cell["pair_dir"],
            task_id=cell["task_id"],
            manifest=str(args.manifest) if args.manifest else None,
            # No restarts and no model selection in either method: stated
            # rather than omitted, because the policy's mandatory group is
            # sample size and selection rule.
            n_restarts=1,
            selection="none",
            **extra)
    pair_dir = resolve(cell["pair_dir"]) / "labeled_dev"
    rec.add_inputs(pair_dir / "G1.edgelist", pair_dir / "G2.edgelist",
                   pair_dir / "mapping.txt", pair_dir / "stats.json")
    rec.seal_id()
    return rec


def _write_mapping(pair_dir, run_id, mapping):
    out = resolve(pair_dir) / f"out_mapping_{run_id}.txt"
    with out.open("w") as fh:
        for u, v in sorted(mapping):
            fh.write(f"{u} {v}\n")
    return out


def _score(extracted, truth_set, test_set):
    """The report's accuracy plus the two numbers it does not state.

    The report quotes recall over the held-out pairs. Precision over what was
    extracted is not derivable from it and is the thing that moves when theta
    drops, so both are recorded.
    """
    extracted = set(extracted)
    correct_heldout = len(extracted & test_set)
    return {
        "extracted": len(extracted),
        "correct_heldout": correct_heldout,
        "accuracy_report": round(100.0 * correct_heldout / len(test_set), 4)
                           if test_set else None,
        "precision_extracted": round(100.0 * correct_heldout / len(extracted), 4)
                               if extracted else None,
        "correct_vs_all_truth": len(extracted & truth_set),
    }


def run_seed_based(cell, args):
    G1, G2, truth, stats = load_pair(resolve(cell["pair_dir"]))
    train, test = split_truth(truth, cell["train_portion"], cell["seed_draw"])
    test_set, truth_set = set(test), set(truth)

    note = args.note or (
        f"Report 1 seed-based cell {cell['task_id']}: nominal beta="
        f"{cell['beta_nominal']} ({cell['beta_mode']}, generated at "
        f"{cell['beta_generated']}), train {cell['train_portion']}, theta "
        f"{cell['theta']}, seed draw {cell['seed_draw']}")

    rec = _new_record("report1_seed_based", cell, args, note, {
        "seed": cell["seed_draw"],
        "theta": cell["theta"],
        "train_portion": cell["train_portion"],
        "split": "prefix" if cell["seed_draw"] < 0 else "shuffled",
        "n_truth": len(truth), "n_train": len(train), "n_test": len(test),
        "matchable_nodes": stats.get("matchable_nodes"),
        "pair_stats": stats,
    }) if not args.no_record else None

    print(f"pair {cell['pair']}  beta={cell['beta_generated']}  "
          f"G1 {G1.number_of_nodes():,}n/{G1.number_of_edges():,}e  "
          f"G2 {G2.number_of_nodes():,}n/{G2.number_of_edges():,}e  "
          f"truth {len(truth):,}  train {len(train):,}  theta {cell['theta']}",
          file=sys.stderr)

    exit_status, error, metrics = "ok", None, {}
    with capture_stderr() as console:
        try:
            # A copy: SeedFreeD extends the list it is handed in place (see
            # the note on aliasing in this module's docstring), and `train` is
            # read again below.
            prop = SeedFreeD(G1, G2, list(train), ecce_threshold=cell["theta"])
            new_pairs, final_map = prop.iterativeOperation()
            metrics = _score(new_pairs, truth_set, test_set)
            final = {(u, v) for u, v in final_map.items()}
            metrics.update({
                "final_mapped": len(final),
                "final_correct": len(final & truth_set),
                "new_pairs": len(set(new_pairs)),
            })
            if rec:
                rec.add_outputs(_write_mapping(cell["pair_dir"], rec.run_id,
                                               final))
        except BaseException as exc:
            exit_status = "error"
            error = traceback.format_exc()
            metrics["failed_in"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if rec:
                rec.metric(**metrics)
                rec.finish(exit_status, error, console.getvalue())
    print(f"task {cell['task_id']} accuracy {metrics.get('accuracy_report')}% "
          f"({metrics.get('correct_heldout')}/{len(test)})  record "
          f"{rec.run_id if rec else 'unrecorded'}")
    return 0


def run_seed_free(cell, args):
    # Imported here rather than at module scope: seed_free imports torch and
    # prints its version on import, which a seed-based cell has no use for.
    from seed_free import determine_seed_pairs

    G1, G2, truth, stats = load_pair(resolve(cell["pair_dir"]))
    truth_set = set(truth)
    full_k = min(G1.number_of_nodes(), G2.number_of_nodes())
    method, theta = cell["method"], cell["theta"]

    print(f"pair {cell['pair']}  method {method}  full ranking k={full_k:,}  "
          f"truth {len(truth):,}", file=sys.stderr)

    # One ranking, every k. get_seed_pairs sorts both score dicts and pairs
    # them index by index, so the list it returns for k is the first k of the
    # list it returns for any larger k: slicing is exactly re-running it.
    rank_cell = dict(cell, top_ks=None)
    rec = _new_record("report1_seed_free_rank", rank_cell, args,
                      args.note or (
                          f"Report 1 Table 2 cell: {method} ranking alone on "
                          f"{cell['pair']} (nominal beta "
                          f"{cell['beta_nominal']}, {cell['beta_mode']})"),
                      {"seed": cell["pair_seed"], "method": method,
                       "top_k": full_k, "integrated": False,
                       "n_truth": len(truth),
                       "matchable_nodes": stats.get("matchable_nodes"),
                       "pair_stats": stats}) if not args.no_record else None

    all_pairs = None
    exit_status, error = "ok", None
    with capture_stderr() as console:
        try:
            all_pairs, correct_pairs, count, accuracy = determine_seed_pairs(
                G1, G2, truth_set, full_k, method_name=method)
            if rec:
                rec.metric(extracted=len(all_pairs), correct=count,
                           accuracy_report=round(accuracy * 100, 4))
        except BaseException as exc:
            exit_status = "error"
            error = traceback.format_exc()
            if rec:
                rec.metric(failed_in=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if rec:
                rec.finish(exit_status, error, console.getvalue())
    print(f"  {method} ranking alone: {count} correct of {len(truth):,} "
          f"({accuracy * 100:.4f}%)  record {rec.run_id if rec else '-'}")

    # Integration: the top-k of that ranking handed to the propagator as seeds.
    for k in (cell["top_ks"] or []):
        if k > len(all_pairs):
            print(f"  skipping top-{k}: the ranking has only "
                  f"{len(all_pairs)} pairs", file=sys.stderr)
            continue
        # Frozen before the propagator sees it, because SeedFreeD extends the
        # list it is given in place: the first version of this scored against a
        # `seeds` that had grown from 25 entries to 608 mid-run, inflating both
        # the denominator and the count. src/seed_free.py avoids it by
        # re-slicing all_pairs[:k] a second time after the run.
        seeds = all_pairs[:k]
        given = set(seeds)
        irec = _new_record("report1_seed_free_integrated",
                           dict(cell, top_ks=None), args,
                           args.note or (
                               f"Report 1 Figure 4 cell: top-{k} of the "
                               f"{method} ranking as seeds to Algorithm 1 on "
                               f"{cell['pair']}"),
                           {"seed": cell["pair_seed"], "method": method,
                            "top_k": k, "integrated": True, "theta": theta,
                            "n_truth": len(truth),
                            "seeds_given_correct": len(given & truth_set),
                            "matchable_nodes": stats.get("matchable_nodes"),
                            "pair_stats": stats}) \
            if not args.no_record else None
        exit_status, error, metrics = "ok", None, {}
        with capture_stderr() as console:
            try:
                prop = SeedFreeD(G1, G2, list(seeds), ecce_threshold=theta)
                new_pairs, final_map = prop.iterativeOperation()
                # seed_free.py's integrate==1 definition: everything extracted,
                # scored against the whole of the ground truth. There is no
                # held-out set here -- the seeds were not drawn from it.
                total = set(new_pairs) | given
                correct = len(total & truth_set)
                metrics = {
                    "extracted": len(total),
                    "correct": correct,
                    "accuracy_report": round(100.0 * correct / len(truth_set), 4),
                    "precision_extracted": round(100.0 * correct / len(total), 4)
                                           if total else None,
                    "new_pairs": len(set(new_pairs)),
                    "final_mapped": len(final_map),
                }
                if irec:
                    irec.add_outputs(_write_mapping(
                        cell["pair_dir"], irec.run_id,
                        {(u, v) for u, v in final_map.items()}))
            except BaseException as exc:
                exit_status = "error"
                error = traceback.format_exc()
                metrics["failed_in"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                if irec:
                    irec.metric(**metrics)
                    irec.finish(exit_status, error, console.getvalue())
        print(f"  {method} top-{k} integrated: "
              f"{metrics.get('accuracy_report')}%  "
              f"record {irec.run_id if irec else '-'}")
    return 0


def load_cells(args):
    if args.manifest:
        man = json.loads(Path(args.manifest).read_text())
        if args.kind and man["kind"] != args.kind:
            raise SystemExit(f"manifest is a {man['kind']} plan, not "
                             f"{args.kind}")
        return man["kind"], man["cells"]
    if not args.kind:
        raise SystemExit("pass --manifest, or --kind plus the grid flags")
    return args.kind, grid.build(args)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run one cell of the Report 1 grid and record it.")
    ap.add_argument("--manifest", default=None,
                    help="Frozen plan from grid.py. Preferred: an array job "
                         "must not re-derive its cell from live defaults.")
    ap.add_argument("--task-id", type=int, default=None,
                    help="Which cell. Defaults to $SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--status", choices=("exploratory", "confirmatory"),
                    default="exploratory",
                    help="A sweep is exploratory. Say confirmatory only when "
                         "re-running one cell against a stated expectation.")
    ap.add_argument("--note", default="",
                    help="Overrides the generated per-cell note.")
    ap.add_argument("--archive", default=None,
                    help="Record archive. Defaults to results/runs/.")
    ap.add_argument("--no-record", action="store_true",
                    help="Debugging only; an unrecorded run cannot be reported.")
    ap.add_argument("--allow-unset-hashseed", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="Run both cell kinds end to end on a synthetic pair "
                         "in a temporary directory. Needs no dataset.")
    # The grid flags, so a cell can be resolved without a manifest.
    grid_args = ap.add_argument_group("grid (only without --manifest)")
    grid_args.add_argument("--kind", choices=("seed_based", "seed_free"))
    grid_args.add_argument("--beta-mode", choices=("divergence", "nominal"),
                           default="divergence")
    grid_args.add_argument("--betas", type=float, nargs="+",
                           default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    grid_args.add_argument("--pair-seeds", type=int, nargs="+", default=[0, 1, 2])
    grid_args.add_argument("--train-portions", type=float, nargs="+",
                           default=[0.05, 0.10, 0.20, 0.40])
    grid_args.add_argument("--thetas", type=float, nargs="+",
                           default=[0.1, 0.3, 0.5, 0.7, 0.9])
    grid_args.add_argument("--seed-draws", type=int, nargs="+", default=[0])
    grid_args.add_argument("--methods", nargs="+", default=list(grid.METHODS),
                           choices=list(grid.METHODS))
    grid_args.add_argument("--top-ks", type=int, nargs="+",
                           default=[25, 50, 100, 150])
    grid_args.add_argument("--gen-mode", choices=("bernoulli", "exact"),
                           default="bernoulli")
    grid_args.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if args.archive is None:
        import provenance
        args.archive = provenance.ARCHIVE

    # An empty value is not a set value: PYTHONHASHSEED= passes a get() test
    # and is rejected by the interpreter itself. The baseline is hash-order
    # sensitive, so the value belongs in the record.
    if not os.environ.get("PYTHONHASHSEED") and not args.allow_unset_hashseed:
        print("PYTHONHASHSEED is not set. Set it (0 is conventional here) or "
              "pass --allow-unset-hashseed.", file=sys.stderr)
        return 2

    kind, cells = load_cells(args)
    task_id = args.task_id
    if task_id is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            print("pass --task-id, or run as a SLURM array task",
                  file=sys.stderr)
            return 2
        task_id = int(env)
    if not 0 <= task_id < len(cells):
        print(f"--task-id must be in [0, {len(cells)}); this plan has "
              f"{len(cells)} cells", file=sys.stderr)
        return 2

    cell = cells[task_id]
    cell.setdefault("task_id", task_id)
    return (run_seed_based if kind == "seed_based" else run_seed_free)(cell, args)


def _self_test():
    """Both kinds, end to end, on a synthetic pair. No dataset, no pollution."""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="report1_selftest_"))
    try:
        d = tmp / "pair" / "labeled_dev"
        d.mkdir(parents=True)
        # Two noisy copies of one graph, with a known relabelling. Small enough
        # to run in seconds and dense enough that propagation has work to do.
        rng = random.Random(7)
        G = nx.barabasi_albert_graph(120, 4, seed=11)
        perm = {n: f"u{i}" for i, n in enumerate(rng.sample(list(G), len(G)))}
        G1 = nx.Graph((f"a{u}", f"a{v}") for u, v in G.edges()
                      if rng.random() > 0.05)
        G2 = nx.Graph((perm[u], perm[v]) for u, v in G.edges()
                      if rng.random() > 0.05)
        nx.write_edgelist(G1, str(d / "G1.edgelist"), data=False)
        nx.write_edgelist(G2, str(d / "G2.edgelist"), data=False)
        pairs = [(f"a{n}", perm[n]) for n in sorted(G)]
        (d / "mapping.txt").write_text(
            "".join(f"{a} {u}\n" for a, u in pairs))
        (d / "stats.json").write_text(json.dumps(
            {"beta": 0.05, "seed": 0, "mode": "bernoulli",
             "matchable_nodes": len(pairs)}))

        G1r, G2r, truth, stats = load_pair(tmp / "pair")
        assert len(truth) > 100, f"membership test dropped too much: {len(truth)}"
        tr, te = split_truth(truth, 0.2, 0)
        assert len(tr) == int(len(truth) * 0.2) and not set(tr) & set(te)
        pre_tr, _ = split_truth(truth, 0.2, -1)
        assert pre_tr == truth[:len(pre_tr)], "seed_draw -1 must be the prefix"
        assert pre_tr != tr, "a shuffled draw that equals the prefix is suspect"

        cell = {"kind": "seed_based", "task_id": 0, "dataset": "hepth",
                "beta_nominal": 0.1, "beta_generated": 0.05, "pair_seed": 0,
                "beta_mode": "divergence", "pair": "selftest_b05_s0",
                "pair_dir": str(tmp / "pair"), "train_portion": 0.2,
                "theta": 0.5, "seed_draw": 0}
        args = argparse.Namespace(status="exploratory", note="self-test",
                                  archive=tmp / "runs", no_record=False,
                                  manifest=None)
        assert run_seed_based(cell, args) == 0
        recs = sorted((tmp / "runs").glob("*/record.json"))
        assert len(recs) == 1, recs
        rec = json.loads(recs[0].read_text())
        assert rec["kind"] == "report1_seed_based"
        assert rec["status_exit"] == "ok", rec.get("error")
        assert rec["metrics"]["accuracy_report"] is not None
        assert rec["train_portion"] == 0.2 and rec["theta"] == 0.5
        assert len(rec["inputs"]) == 4 and rec["outputs"], rec
        assert not rec.get("record_incomplete"), rec["record_incomplete"]
        print(f"  seed-based self-test accuracy "
              f"{rec['metrics']['accuracy_report']}%  "
              f"({rec['metrics']['correct_heldout']} held-out pairs found)")

        sf = dict(cell, kind="seed_free", method="degree", top_ks=[10, 25])
        assert run_seed_free(sf, args) == 0
        kinds = [json.loads(p.read_text())["kind"]
                 for p in sorted((tmp / "runs").glob("*/record.json"))]
        assert kinds.count("report1_seed_free_rank") == 1, kinds
        assert kinds.count("report1_seed_free_integrated") == 2, kinds
        # Regression guard for the aliasing bug: SeedFreeD extends the seed
        # list in place, so scoring against the list rather than a frozen copy
        # inflates `extracted` past what the run could possibly have produced.
        for path in sorted((tmp / "runs").glob("*/record.json")):
            r = json.loads(path.read_text())
            if r["kind"] != "report1_seed_free_integrated":
                continue
            bound = r["metrics"]["new_pairs"] + r["top_k"]
            assert r["metrics"]["extracted"] <= bound, (
                f"extracted {r['metrics']['extracted']} exceeds "
                f"new_pairs + top_k = {bound}: the seed list was mutated "
                f"under the scorer")

        ledger = (tmp / "runs" / "index.jsonl").read_text().splitlines()
        assert len(ledger) == 4, f"{len(ledger)} ledger lines for 4 runs"
        print(f"  seed-free self-test wrote {len(kinds)} records "
              f"(1 ranking + 2 integrated)")
        print("report1_cell self-test passed")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
