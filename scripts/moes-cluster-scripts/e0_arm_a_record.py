#!/usr/bin/env python3
"""E0 arm (a): run Report 1's own seed-free algorithm unchanged, and record it.

Log 3, section "E0: Report 1 reproduction". The question is whether the
inherited `src/seed_free.py` gives Report 1's 88.88% when run, unmodified, on
Report 1's own pair (`data/labeled_dev`) in this container. This script is the
recorder around it. It does not reimplement or edit anything: it sets
`sys.argv` and calls the inherited `main()`, then reads back what that wrote.

    PYTHONHASHSEED=0 python3 e0_arm_a_record.py --mode integrated --out DIR
    PYTHONHASHSEED=0 python3 e0_arm_a_record.py --mode pure       --out DIR

The two modes are the two inherited command lines (both with `--method_name
pagerank --path <repo>/data`, `--out` a fresh directory per run):

    integrated   --integrate_with_Seed_Based 1 --number_of_top_index_pairs 50
    pure         --integrate_with_Seed_Based 0

and write two record kinds, `report1_seed_free_integrated` and
`report1_seed_free_rank`, in the field layout of `report1_cell.py`
(:277-297 and :311-366), plus fields for what is specific to this pair:

- top level: `seed` is the hash seed (there is no pair seed), `beta` and
  `beta_mode` are null, `dataset` is labeled_dev, `n_truth` and
  `matchable_nodes` are the 26,948 pairs whose two nodes exist, and
  `n_validation_inherited` (26,942), `validation_filter` ("u in G1 and u in
  G2") and `pagerank_alpha` (1) describe the inherited code's own scoring list.
  `seeds_given_correct` (integrated) is set after the run, so it is not part of
  the run id.
- metrics: `accuracy_inherited` (what the script printed, denominator 26,942),
  `accuracy_report` (A_full: the same pairs over 26,948, in percent),
  `correct` (over the 26,948-pair truth), `correct_inherited` (over the
  inherited list; the number the script prints as "# of Correct pairs"),
  `extracted`, `precision_extracted`, `new_pairs` (integrated),
  `mapping_set_sha256` (the extracted pairs as a set; the file's own bytes
  depend on the hash seed, because the script writes a set in iteration order)
  and `seconds_main` (the inherited main() alone).

Checks. The run fails, with its record written and `status_exit` error, unless
(1) the recorder's count of the saved mapping over the inherited list equals the
file's "# of Correct pairs" (the check the log asks for), (2) the file's
"Accuracy" is that count over the list's length, and, for integrated, (3) the 50
seeds recomputed with `determine_seed_pairs` (deterministic, no hash dependence)
are all in the saved mapping and their correct count reproduces the file's
"Method Accuracy". Checks 2 and 3 go beyond the log; they cost nothing and tie
the recount to the inherited code's own output.

It refuses to start unless PYTHONHASHSEED is set to an integer, as
`report1_cell.py` (:437-443) refuses an unset one (and, unlike it, has no
override): the inherited propagation orders sets of strings, so the hash seed is
the replicate, and its value goes in the record. `random` is refused too, since
it cannot be repeated.

Details worth knowing:

- `seed_free.main()` `os.chdir`s into `--path` (:414), so every path given to
  it is absolute; this script restores the working directory afterwards.
- The accuracy file is named for `number_of_top_index_pairs`, which the pure
  branch resets to 0 (:440), and the mapping file for the integrate flag. So an
  integrated run writes accuracy_top50.txt and seed_free_final_mapping_1.txt,
  and a pure run writes accuracy_top0.txt and seed_free_final_mapping_0.txt.
- `--out` must not exist or must be empty (the inherited file names are fixed,
  so a reused directory would be read as this run's output) and must not sit
  inside `labeled_dev/`, whose files the inherited loader picks by substring.
- SIGTERM (scancel, or the walltime) is turned into an exception so that a
  killed run still writes its record, with `status_exit` killed.
- The inherited stdout (every correct pair, every added pair) is not captured
  into the record; it is in the job log.

`--self-test` runs both modes end to end on a small synthetic pair, in a
temporary directory with a temporary archive, and needs no dataset. It is a
test of this script, not an E0 result.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

# The inherited script imports matplotlib and calls plt.show() from inside
# determine_seed_pairs (seed_free.py:96). Agg makes that a no-op on a node with
# no display. Set before seed_free is imported.
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "adv_gd"))

from provenance import RunRecord, capture_stderr             # noqa: E402

KINDS = {"integrated": "report1_seed_free_integrated",
         "pure": "report1_seed_free_rank"}
DATASET = "labeled_dev"
METHOD = "pagerank"
TOP_K = 50                       # --number_of_top_index_pairs, integrated
THETA = 0.5                      # seed_free.py:459; not an argument of main()
PAGERANK_ALPHA = 1               # seed_free.py:272-273
VALIDATION_FILTER = "u in G1 and u in G2"        # seed_free.py:385
# Log 3, E0 "What the repo has": warned about, not enforced (the synthetic
# self-test pair has other counts).
EXPECT_N_TRUTH = 26948
EXPECT_N_INHERITED = 26942
# The inherited loader finds its files by substring ('G1.edgelist',
# 'G2.edgelist', 'mapping'), so the directory must hold exactly these.
INPUT_FILES = ("validation_G1.edgelist", "validation_G2.edgelist",
               "validation_seed_mapping.txt")


class Killed(BaseException):
    """SIGTERM as an exception. A BaseException, so no `except Exception` in
    the inherited code can swallow it."""


@contextlib.contextmanager
def term_as_exception():
    def handler(signum, frame):
        raise Killed(f"received signal {signum}")
    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def hash_seed_from_env(environ=None):
    """The hash seed this process is running under, or ValueError saying why
    it is not a usable replicate. An empty value is not a set value:
    PYTHONHASHSEED= passes a get() test and is rejected by the interpreter."""
    environ = os.environ if environ is None else environ
    raw = environ.get("PYTHONHASHSEED", "")
    if not raw:
        raise ValueError(
            "PYTHONHASHSEED is not set. The inherited propagation is hash-order "
            "sensitive and the hash seed is E0's replicate: set it to an "
            "integer (0 is conventional here).")
    try:
        seed = int(raw)
    except ValueError:
        raise ValueError(
            f"PYTHONHASHSEED={raw!r} is not an integer; 'random' cannot be "
            "repeated, so it cannot be a replicate.") from None
    if not 0 <= seed <= 4294967295:
        raise ValueError(f"PYTHONHASHSEED={seed} is outside [0, 4294967295].")
    return seed


# -- reading what the inherited code wrote --------------------------------------

def read_pairs(path):
    """'u v' per line, as seed_free.py writes and reads them (one space)."""
    pairs = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            parts = line.rstrip("\n").split(" ")
            if len(parts) != 2 or not all(parts):
                raise ValueError(f"{path} line {n}: expected 'u v', got {line!r}")
            pairs.append((parts[0], parts[1]))
    return pairs


def parse_accuracy_file(path):
    """The three lines seed_free.py:476-480 writes."""
    lines = Path(path).read_text().splitlines()
    want = ("Accuracy: ", "# of Correct pairs: ",
            f"Method Accuracy ({METHOD}): ")
    if len(lines) != 3 or not all(l.startswith(w) for l, w in zip(lines, want)):
        raise ValueError(f"{path}: unexpected layout {lines!r}")
    return {"accuracy": float(lines[0][len(want[0]):]),
            "correct": int(lines[1][len(want[1]):]),
            "method_accuracy": float(lines[2][len(want[2]):])}


def mapping_set_sha256(pairs):
    """Digest of the pairs as a set: unlike the file's bytes, it does not
    depend on the hash seed's effect on iteration order."""
    text = "".join(f"{u} {v}\n" for u, v in sorted(set(pairs)))
    return hashlib.sha256(text.encode()).hexdigest()


# -- scoring --------------------------------------------------------------------

def load_pair(labeled_dev):
    """(G1, G2, L, T) through the inherited loader.

    L is the inherited scoring list ('u in G1 and u in G2'); T is the truth the
    log scores against ('u in G1 and v in G2'), as a set. Both come from the
    same mapping file, which the inherited loader picks by substring.
    """
    import seed_free
    names = sorted(os.listdir(labeled_dev))
    if names != sorted(INPUT_FILES):
        raise ValueError(f"{labeled_dev} must hold exactly {sorted(INPUT_FILES)} "
                         f"(the inherited loader picks files by substring); "
                         f"found {names}")
    G1, G2, L = seed_free.process_files_and_get_initial_data(str(labeled_dev))
    truth = {(u, v) for u, v in read_pairs(labeled_dev / INPUT_FILES[2])
             if u in G1 and v in G2}
    return G1, G2, L, truth


def score_mapping(extracted, truth, inherited):
    """Counts for a set of extracted pairs against both truths.

    `correct` is over T (what A_full is), `correct_inherited` over the
    inherited list (what the script prints). Every extracted pair has u in G1
    and v in G2, so the first is never smaller than the second; it exceeds it
    by the extracted pairs among T minus L, which holds 14 pairs on
    labeled_dev (|T| - |L| is 6, but L is not a subset of T: 8 pairs are in L
    only).
    """
    X = set(extracted)
    return {"extracted": len(X), "correct": len(X & truth),
            "correct_inherited": len(X & inherited)}


def verify_against_file(acc, counted, n_inherited, integrated, seed_check=None):
    """Raise AssertionError, naming every disagreement, unless the recount
    reproduces the inherited script's own output.

    `acc` is parse_accuracy_file's dict, `counted` is score_mapping's, and
    `seed_check` (integrated only) is {"in_mapping": bool, "correct_inherited":
    int}, the recomputed seeds' relation to the saved mapping and their correct
    count over the inherited list.
    """
    bad = []
    if counted["correct_inherited"] != acc["correct"]:
        bad.append(f"recount over the inherited list is "
                   f"{counted['correct_inherited']}, the file says "
                   f"'# of Correct pairs: {acc['correct']}'")
    want = counted["correct_inherited"] * 100 / n_inherited
    if not math.isclose(want, acc["accuracy"], rel_tol=1e-12):
        bad.append(f"the file's Accuracy {acc['accuracy']!r} is not "
                   f"{counted['correct_inherited']} / {n_inherited} * 100 = "
                   f"{want!r}")
    if integrated:
        if not seed_check["in_mapping"]:
            bad.append("the recomputed 50 seeds are not all in the saved "
                       "mapping")
        want_m = (seed_check["correct_inherited"] / n_inherited) * 100
        if not math.isclose(want_m, acc["method_accuracy"], rel_tol=1e-12):
            bad.append(f"the file's Method Accuracy {acc['method_accuracy']!r} "
                       f"is not the recomputed seeds' "
                       f"{seed_check['correct_inherited']} / {n_inherited} * "
                       f"100 = {want_m!r}")
    if bad:
        raise AssertionError("; ".join(bad))


# -- the run --------------------------------------------------------------------

def inherited_argv(mode, data_dir, out_dir):
    argv = ["seed_free.py", "--method_name", METHOD, "--path", str(data_dir),
            "--out", str(out_dir)]
    if mode == "integrated":
        argv += ["--integrate_with_Seed_Based", "1",
                 "--number_of_top_index_pairs", str(TOP_K)]
    else:
        argv += ["--integrate_with_Seed_Based", "0"]
    return argv


def run_arm_a(mode, out_dir, data_dir, hash_seed, status="exploratory",
              run_index=None, pair_label="e0_labeled_dev", note="",
              archive=None):
    """One run, one record. Returns the finished RunRecord."""
    integrated = mode == "integrated"
    labeled_dev = data_dir / "labeled_dev"
    cwd = os.getcwd()
    note = note or (f"E0 arm (a), {mode}, run "
                    f"{'?' if run_index is None else run_index}: inherited "
                    f"seed_free.py main() unchanged, pagerank alpha=1"
                    f"{', top 50 as seeds, theta 0.5' if integrated else ', full ranking alone'}"
                    f", PYTHONHASHSEED={hash_seed} (Log 3, E0)")

    kw = {} if archive is None else {"archive": archive}
    rec = RunRecord(KINDS[mode], pair_label, status, note, **kw)
    # Configuration only. Paths are left out on purpose: they differ between a
    # run and its repeat, and a repeat must share its run id's digest.
    rec.set(seed=hash_seed, dataset=DATASET, beta=None, beta_mode=None,
            method=METHOD, integrated=integrated,
            n_restarts=1, selection="none",
            pagerank_alpha=PAGERANK_ALPHA, validation_filter=VALIDATION_FILTER,
            inherited_args={"method_name": METHOD,
                            "integrate_with_Seed_Based": int(integrated),
                            **({"number_of_top_index_pairs": TOP_K}
                               if integrated else {})})
    if integrated:
        rec.set(theta=THETA)
    rec.add_inputs(*[labeled_dev / n for n in INPUT_FILES])

    exit_status, error, metrics = "ok", None, {}
    with capture_stderr() as console, term_as_exception():
        try:
            G1, G2, L, truth = load_pair(labeled_dev)
            inherited = set(L)
            rec.set(n_truth=len(truth), matchable_nodes=len(truth),
                    n_validation_inherited=len(L),
                    top_k=TOP_K if integrated else min(len(G1), len(G2)))
            rec.seal_id()
            print(f"pair labeled_dev  G1 {len(G1):,} nodes  G2 {len(G2):,} nodes  "
                  f"truth {len(truth):,}  inherited list {len(L):,}  "
                  f"hash seed {hash_seed}", file=sys.stderr)
            if (len(truth), len(L)) != (EXPECT_N_TRUTH, EXPECT_N_INHERITED):
                print(f"  WARNING: Log 3 expects {EXPECT_N_TRUTH:,} truth pairs "
                      f"and an inherited list of {EXPECT_N_INHERITED:,}; these "
                      f"are not Report 1's files", file=sys.stderr)

            # -- the inherited program, unmodified --------------------------------
            import seed_free
            saved_argv = sys.argv
            sys.argv = inherited_argv(mode, data_dir, out_dir)
            t0 = time.time()
            try:
                seed_free.main()
            finally:
                sys.argv = saved_argv
                os.chdir(cwd)             # main() chdirs into --path (:414)
            metrics["seconds_main"] = round(time.time() - t0, 1)
            print(f"  inherited main() finished in {metrics['seconds_main']}s",
                  file=sys.stderr)

            # -- read back, recount -----------------------------------------------
            acc_path = Path(out_dir) / ("accuracy_top50.txt" if integrated
                                        else "accuracy_top0.txt")
            map_path = Path(out_dir) / ("seed_free_final_mapping_1.txt"
                                        if integrated
                                        else "seed_free_final_mapping_0.txt")
            rec.add_outputs(acc_path, map_path)
            acc = parse_accuracy_file(acc_path)
            extracted = read_pairs(map_path)
            counted = score_mapping(extracted, truth, inherited)
            if counted["extracted"] != len(extracted):
                raise ValueError(f"{map_path} lists a pair twice")
            metrics.update(
                accuracy_inherited=acc["accuracy"],
                correct_inherited=counted["correct_inherited"],
                correct_inherited_file=acc["correct"],
                extracted=counted["extracted"],
                correct=counted["correct"],
                accuracy_report=round(100.0 * counted["correct"] / len(truth), 4),
                precision_extracted=(round(100.0 * counted["correct"]
                                           / counted["extracted"], 4)
                                     if counted["extracted"] else None),
                mapping_set_sha256=mapping_set_sha256(extracted))

            seed_check = None
            if integrated:
                # The 50 seeds again, by the inherited ranking code. It is
                # deterministic (PageRank returns edge-file order and the sort
                # is stable), so this is the list main() handed to SeedFreeD.
                seeds, _, count, _ = seed_free.determine_seed_pairs(
                    G1, G2, L, TOP_K, method_name=METHOD)
                seed_set = set(seeds)
                seed_check = {"in_mapping": seed_set <= set(extracted),
                              "correct_inherited": count}
                metrics.update(
                    new_pairs=len(set(extracted) - seed_set),
                    seeds_correct_inherited=count)
                # Top level, as report1_cell.py has it, but measured after the
                # run: set once the id is sealed, so not part of the id.
                rec.set(seeds_given_correct=len(seed_set & truth))
                print(f"  seeds: {len(seed_set & truth)} of {len(seed_set)} "
                      f"correct over the truth, {count} over the inherited "
                      f"list", file=sys.stderr)

            print(f"  accuracy printed {acc['accuracy']!r}  A_full "
                  f"{metrics['accuracy_report']}%  correct {counted['correct']} "
                  f"(inherited list {counted['correct_inherited']})  extracted "
                  f"{counted['extracted']:,}", file=sys.stderr)
            verify_against_file(acc, counted, len(L), integrated, seed_check)
        except BaseException as exc:
            exit_status = ("killed" if isinstance(exc, (Killed, KeyboardInterrupt))
                           else "error")
            error = traceback.format_exc()
            metrics["failed_in"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            os.chdir(cwd)
            rec.metric(**metrics)
            rec.finish(exit_status, error, console.getvalue())
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E0 arm (a): run the inherited seed_free.py unchanged on "
                    "data/labeled_dev and record it.")
    ap.add_argument("--mode", choices=sorted(KINDS),
                    help="integrated: top 50 PageRank pairs into SeedFreeD; "
                         "pure: the PageRank ranking scored alone.")
    ap.add_argument("--out", help="A fresh directory for this run's accuracy "
                                  "file and final mapping.")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data"),
                    help="Directory holding labeled_dev/ (default data/).")
    ap.add_argument("--status", choices=("exploratory", "confirmatory"),
                    default="exploratory",
                    help="Default exploratory. Say confirmatory only for the "
                         "runs the approved predictions cover.")
    ap.add_argument("--run-index", type=int, default=None,
                    help="Which of E0's 11 runs this is. Goes in the note only, "
                         "so a repeat of a configuration shares its digest.")
    ap.add_argument("--pair-label", default="e0_labeled_dev",
                    help="The record's `pair`, and so part of its run id. "
                         "Default matches the staged arm-(b) pair directory.")
    ap.add_argument("--note", default="", help="Overrides the generated note.")
    ap.add_argument("--archive", default=None,
                    help="Record archive. Defaults to results/runs/.")
    ap.add_argument("--self-test", action="store_true",
                    help="Both modes end to end on a synthetic pair, in a "
                         "temporary directory. Needs PYTHONHASHSEED too.")
    args = ap.parse_args(argv)

    try:
        hash_seed = hash_seed_from_env()
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    if args.self_test:
        return _self_test()
    if not args.mode or not args.out:
        ap.error("--mode and --out are required (or pass --self-test)")

    out_dir = Path(os.path.abspath(args.out))
    data_dir = Path(os.path.abspath(args.data_dir))
    labeled_dev = data_dir / "labeled_dev"
    if not labeled_dev.is_dir():
        print(f"{labeled_dev} is not a directory", file=sys.stderr)
        return 2
    if out_dir == labeled_dev or labeled_dev in out_dir.parents:
        print(f"--out {out_dir} is inside {labeled_dev}, whose files the "
              f"inherited loader picks by substring ('mapping' would match "
              f"its own output)", file=sys.stderr)
        return 2
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        print(f"--out {out_dir} exists and is not an empty directory. The "
              f"inherited file names are fixed, so each run needs its own.",
              file=sys.stderr)
        return 2

    rec = run_arm_a(args.mode, out_dir, data_dir, hash_seed, args.status,
                    args.run_index, args.pair_label, args.note,
                    archive=args.archive)
    m = rec.data["metrics"]
    print(f"e0 arm (a) {args.mode}: accuracy_inherited "
          f"{m.get('accuracy_inherited')}  A_full {m.get('accuracy_report')}%  "
          f"correct {m.get('correct')}  record {rec.run_id}")
    return 0


# -- self-test ------------------------------------------------------------------

def _write_synthetic_pair(data_dir, n=400, seed=5):
    """Two noisy copies of one graph with unrelated labels, as labeled_dev has.

    G2's labels are a random n-subset of 0..n+39, so a G1 label is often not a
    G2 label at all: the inherited list ('u in G1 and u in G2') comes out
    clearly smaller than the truth ('u in G1 and v in G2'), which is the
    difference the real pair has at a scale of 6 pairs.
    """
    import random
    import networkx as nx
    rng = random.Random(seed)
    base = nx.barabasi_albert_graph(n, 4, seed=seed)
    to_u = dict(enumerate(rng.sample(range(n + 40), n)))
    d = Path(data_dir) / "labeled_dev"
    d.mkdir(parents=True)
    for name, relabel in (("validation_G1.edgelist", lambda i: i),
                          ("validation_G2.edgelist", lambda i: to_u[i])):
        with open(d / name, "w") as fh:
            for a, b in base.edges():
                if rng.random() > 0.03:
                    fh.write(f"{relabel(a)} {relabel(b)}\n")
    with open(d / "validation_seed_mapping.txt", "w") as fh:
        for i in range(n):
            fh.write(f"{i} {to_u[i]}\n")


def _self_test():
    import shutil
    import subprocess
    import tempfile

    hash_seed = hash_seed_from_env()
    tmp = Path(tempfile.mkdtemp(prefix="e0_arm_a_selftest_"))
    try:
        data = tmp / "data"
        _write_synthetic_pair(data)
        runs = tmp / "runs"

        def go(mode, tag, index):
            rc = main(["--mode", mode, "--out", str(tmp / tag),
                       "--data-dir", str(data), "--run-index", str(index),
                       "--archive", str(runs)])
            assert rc == 0, rc
            # record*.json, not record.json: two runs of one configuration in
            # one UTC minute share a run id, and provenance writes the second
            # as record.<epoch>.json beside the first.
            return max(runs.glob("*/record*.json"),
                       key=lambda p: p.stat().st_mtime_ns)

        # -- integrated, twice: a repeat must share the digest ------------------
        r1 = json.loads(go("integrated", "int_a", 0).read_text())
        r2 = json.loads(go("integrated", "int_b", 5).read_text())
        assert r1["kind"] == "report1_seed_free_integrated", r1["kind"]
        assert r1["status"] == "exploratory" and r1["status_exit"] == "ok"
        assert not r1.get("record_incomplete"), r1["record_incomplete"]
        assert r1["seed"] == hash_seed == int(r1["env"]["hashseed"])
        assert r1["beta"] is None and r1["beta_mode"] is None
        assert r1["dataset"] == DATASET and r1["top_k"] == TOP_K
        assert r1["theta"] == THETA and r1["pagerank_alpha"] == 1
        assert r1["validation_filter"] == VALIDATION_FILTER
        assert len(r1["inputs"]) == 3 and len(r1["outputs"]) == 2
        assert "run 0" in r1["note"] and "run 5" in r2["note"]
        assert r1["config_sha256"] == r2["config_sha256"], \
            "a repeat of one configuration must share its digest"
        assert r1["n_validation_inherited"] < r1["n_truth"], \
            "the synthetic pair no longer separates the two lists"
        m = r1["metrics"]
        assert m["correct"] >= m["correct_inherited"] == m["correct_inherited_file"]
        assert m["extracted"] == m["new_pairs"] + TOP_K
        assert 0 <= r1["seeds_given_correct"] <= TOP_K
        assert m["mapping_set_sha256"] == r2["metrics"]["mapping_set_sha256"]
        print(f"  integrated: A_inh {m['accuracy_inherited']:.4f}  A_full "
              f"{m['accuracy_report']}  correct {m['correct']} "
              f"(inherited list {m['correct_inherited']} of "
              f"{r1['n_validation_inherited']}, truth {r1['n_truth']})  "
              f"seeds {r1['seeds_given_correct']}/{TOP_K}  new {m['new_pairs']}",
              file=sys.stderr)

        # -- pure ---------------------------------------------------------------
        rp = json.loads(go("pure", "pure_a", 6).read_text())
        assert rp["kind"] == "report1_seed_free_rank" and rp["integrated"] is False
        assert rp["status_exit"] == "ok" and "theta" not in rp
        assert rp["top_k"] == rp["metrics"]["extracted"] > TOP_K
        assert [Path(o["path"]).name for o in rp["outputs"]] == [
            "accuracy_top0.txt", "seed_free_final_mapping_0.txt"]
        print(f"  pure: {rp['metrics']['correct']} correct of "
              f"{rp['metrics']['extracted']} ranked "
              f"({rp['metrics']['accuracy_report']}%)", file=sys.stderr)

        # -- the checks must be able to fail -------------------------------------
        acc = parse_accuracy_file(tmp / "int_a" / "accuracy_top50.txt")
        counted = {"correct_inherited": acc["correct"], "correct": acc["correct"],
                   "extracted": 1}
        seeds_ok = {"in_mapping": True,
                    "correct_inherited": round(acc["method_accuracy"]
                                               * r1["n_validation_inherited"] / 100)}
        verify_against_file(acc, counted, r1["n_validation_inherited"], True,
                            seeds_ok)
        for label, bad_counted, bad_seeds in (
                ("count off by one",
                 dict(counted, correct_inherited=acc["correct"] + 1), seeds_ok),
                ("seeds missing from mapping",
                 counted, dict(seeds_ok, in_mapping=False)),
                ("seed count off by one",
                 counted, dict(seeds_ok,
                               correct_inherited=seeds_ok["correct_inherited"] + 1))):
            try:
                verify_against_file(acc, bad_counted, r1["n_validation_inherited"],
                                    True, bad_seeds)
            except AssertionError as exc:
                print(f"  check fires on {label}: {exc}", file=sys.stderr)
            else:
                raise SystemExit(f"self-test: the check did not fire on {label}")

        # -- refusals ------------------------------------------------------------
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
        for label, extra_env in (("unset", {}), ("random", {"PYTHONHASHSEED": "random"})):
            got = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--mode", "pure",
                 "--out", str(tmp / "never"), "--data-dir", str(data),
                 "--archive", str(runs)],
                env={**env, **extra_env}, capture_output=True, text=True)
            assert got.returncode == 2 and "PYTHONHASHSEED" in got.stderr, got
            assert not (tmp / "never").exists()
            print(f"  refuses PYTHONHASHSEED {label}: rc {got.returncode}",
                  file=sys.stderr)
        assert main(["--mode", "pure", "--out", str(tmp / "int_a"),
                     "--data-dir", str(data), "--archive", str(runs)]) == 2, \
            "a used --out must be refused"
        assert main(["--mode", "pure", "--out", str(data / "labeled_dev" / "o"),
                     "--data-dir", str(data), "--archive", str(runs)]) == 2

        # -- a killed run still leaves a record ------------------------------------
        import seed_free
        real_main = seed_free.main

        def terminated():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(30)                 # the handler raises long before this

        before = set(runs.glob("*/record*.json"))
        seed_free.main = terminated
        try:
            run_arm_a("pure", tmp / "killed", data, hash_seed, archive=runs)
        except Killed:
            pass
        else:
            raise SystemExit("self-test: SIGTERM did not raise Killed")
        finally:
            seed_free.main = real_main
        (new,) = set(runs.glob("*/record*.json")) - before
        rk = json.loads(new.read_text())
        assert rk["status_exit"] == "killed" and "Killed" in rk["error"], rk
        print("  SIGTERM mid-run: record written, status_exit killed",
              file=sys.stderr)

        assert len(list(runs.glob("*/record*.json"))) == 4
        print("e0_arm_a_record self-test passed")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
