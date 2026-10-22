#!/usr/bin/env python3
"""Tabulate the Report 1 run records into a CSV, and summarise one axis.

A study here is N runs with N records plus a small script that reads them --
not a bespoke JSON shape that only its own writer understands. This is that
script.

It reads `results/runs/*/record.json` rather than `index.jsonl`, deliberately.
The ledger line is Adv-GD shaped: it carries `chi`, `gamma` and
`anchor_precision` and no accuracy, so it can tell you a Report 1 cell ran but
not what it found. The records hold everything. `index.jsonl` stays the thing
you grep to see that a run exists.

Records are immutable, so label fixes live in `results/runs/corrections.jsonl`
(docs/experiment_logging_policy.md, Rule 7). They are overlaid on each record
in memory, later lines winning; no record file is written.

    python3 collect.py --kind report1_seed_based
    python3 collect.py --kind report1_seed_based --summary beta_nominal theta
    python3 collect.py --out /data/$USER/report1.csv

Python 3.6 compatible, like grid.py: it runs on the head node, outside the
container, with no third-party imports.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ARCHIVE = REPO_ROOT / "results" / "runs"

# Config first, then what was measured. A column order that puts the question
# before the answer, so a row reads left to right.
COLUMNS = [
    "run_id", "kind", "status_exit", "status", "pair",
    "beta_nominal", "beta_generated", "beta_mode", "pair_seed",
    "train_portion", "theta", "split", "seed", "seed_draw",
    "method", "top_k", "integrated",
    "n_truth", "n_train", "n_test", "matchable_nodes",
    "extracted", "new_pairs", "correct", "correct_heldout",
    "accuracy_report", "precision_extracted",
    "final_mapped", "final_correct",
    "seconds", "git", "note",
]
METRIC_KEYS = {"extracted", "new_pairs", "correct", "correct_heldout",
               "accuracy_report", "precision_extracted", "final_mapped",
               "final_correct"}


def load_corrections(archive):
    """{run_id: [(field, old, new), ...]} from `corrections.jsonl`, file order.

    No file means no corrections. A list rather than {field: new} so a
    correction of a correction is checked against the value the earlier line
    set, not against the record.
    """
    path = archive / "corrections.jsonl"
    out = {}
    if not path.is_file():
        return out
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            c = json.loads(line)
            out.setdefault(c["run_id"], []).append(
                (c["field"], c["old"], c["new"]))
        except (ValueError, KeyError, TypeError) as exc:
            print("  skipping bad line {} of {}: {!r}".format(n, path, exc),
                  file=sys.stderr)
    return out


def apply_corrections(rec, corrections):
    """Overlay `corrections` on the loaded dict `rec`; return the fields set.

    A stale `old` is a warning, not a reason to drop the correction.
    """
    fields = []
    for field, old, new in corrections.get(rec.get("run_id"), ()):
        if rec.get(field) != old:
            print("  correction for {} {}: record has {!r}, expected old {!r};"
                  " applying anyway".format(rec.get("run_id"), field,
                                            rec.get(field), old),
                  file=sys.stderr)
        rec[field] = new
        fields.append(field)
    return fields


def rows(archive, kinds=None, include_failed=True):
    corrections = load_corrections(archive)
    for rec_path in sorted(archive.glob("*/record.json")):
        try:
            rec = json.loads(rec_path.read_text())
        except (OSError, ValueError) as exc:
            print("  skipping unreadable {}: {}".format(rec_path, exc),
                  file=sys.stderr)
            continue
        corrected = apply_corrections(rec, corrections)
        if kinds and rec.get("kind") not in kinds:
            continue
        if not include_failed and rec.get("status_exit") != "ok":
            continue
        metrics = rec.get("metrics") or {}
        row = {"_corrected": bool(corrected)}
        for col in COLUMNS:
            if col in METRIC_KEYS:
                row[col] = metrics.get(col)
            elif col == "git":
                g = rec.get("git") or {}
                row[col] = (g.get("head", "") +
                            ("+dirty" if g.get("dirty") else ""))
            else:
                row[col] = rec.get(col)
        # seed_draw is the seed-based name for what the record stores as `seed`;
        # keep both columns populated so a filter on either works.
        if row["seed_draw"] is None and rec.get("kind") == "report1_seed_based":
            row["seed_draw"] = rec.get("seed")
        yield row


def summarise(data, keys, value="accuracy_report"):
    """Group by `keys` and report n, mean, min, max of `value`.

    No standard deviation over fewer than two draws, and no interval of any
    kind presented as one: with the default grid a group is three noise draws,
    which is a range and not a confidence interval.
    """
    groups = {}
    for r in data:
        if r.get(value) is None:
            continue
        k = tuple(r.get(key) for key in keys)
        groups.setdefault(k, []).append(float(r[value]))
    out = []
    for k in sorted(groups, key=lambda t: [(_ is None, _) for _ in t]):
        v = groups[k]
        out.append(list(k) + [len(v), round(sum(v) / len(v), 4),
                              round(min(v), 4), round(max(v), 4)])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Tabulate Report 1 run records.")
    ap.add_argument("--archive", default=str(ARCHIVE),
                    help="Record archive (default results/runs/).")
    ap.add_argument("--kind", nargs="+", default=None,
                    help="Filter by record kind, e.g. report1_seed_based "
                         "report1_seed_free_integrated. Default: all Report 1 "
                         "kinds.")
    ap.add_argument("--all-kinds", action="store_true",
                    help="Include adv_gd and generator records too.")
    ap.add_argument("--ok-only", action="store_true",
                    help="Drop runs whose status_exit is not ok. Off by "
                         "default: a cell that died is data about where it got.")
    ap.add_argument("--out", default=None, help="Write CSV here (default stdout).")
    ap.add_argument("--summary", nargs="+", default=None, metavar="KEY",
                    help="Group by these columns and print n/mean/min/max of "
                         "--value instead of the rows.")
    ap.add_argument("--value", default="accuracy_report",
                    help="Column to summarise (default accuracy_report).")
    args = ap.parse_args(argv)

    kinds = args.kind
    if kinds is None and not args.all_kinds:
        kinds = ["report1_seed_based", "report1_seed_free_rank",
                 "report1_seed_free_integrated"]

    archive = Path(args.archive)
    if not archive.is_dir():
        print("no archive at {}".format(archive), file=sys.stderr)
        return 1
    data = list(rows(archive, kinds, include_failed=not args.ok_only))
    if not data:
        print("no matching records in {}".format(archive), file=sys.stderr)
        return 1

    if args.summary:
        unknown = [k for k in args.summary if k not in COLUMNS]
        if unknown:
            print("unknown summary key(s): {}\navailable: {}".format(
                ", ".join(unknown), ", ".join(COLUMNS)), file=sys.stderr)
            return 2
        table = summarise(data, args.summary, args.value)
        header = list(args.summary) + ["n", "mean", "min", "max"]
        widths = [max(len(str(h)), max((len(str(r[i])) for r in table),
                                       default=0))
                  for i, h in enumerate(header)]
        fmt = "  ".join("{{:>{}}}".format(w) for w in widths)
        print(fmt.format(*header))
        print(fmt.format(*["-" * w for w in widths]))
        for r in table:
            print(fmt.format(*[("" if c is None else c) for c in r]))
        failed = sum(1 for r in data if r["status_exit"] != "ok")
        print("\n{} records, {} not ok, {} groups, {} with corrected fields"
              .format(len(data), failed, len(table),
                      sum(r["_corrected"] for r in data)))
        return 0

    out = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        w = csv.DictWriter(out, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in data:
            w.writerow(r)
    finally:
        if args.out:
            out.close()
            print("wrote {} rows to {} ({} with corrected fields)".format(
                len(data), args.out, sum(r["_corrected"] for r in data)),
                file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
