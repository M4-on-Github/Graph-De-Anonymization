#!/usr/bin/env python3
"""Tabulate run records into a CSV, and summarise one axis.

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
in memory, later lines winning; no record file is written. A row's `corrected`
column names the fields it got that way.

A run found invalid is never deleted: a new record names it in `supersedes`
(Rule 1). Records named that way are left out by default; the naming record is
looked for in the whole archive, even if `--kind` or `--ok-only` drops it from
the output. `--include-superseded` keeps them, with `superseded_by` filled in.

    python3 collect.py                                # every record, as CSV
    python3 collect.py --summary kind                 # mean chi per kind
    python3 collect.py --kind report1_seed_based --summary beta_nominal theta
    python3 collect.py --include-superseded --out /data/$USER/all.csv

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
    # Appended, so a reader that indexes the older columns by position still
    # works. `chi` is the Adv-GD suite's headline number; the Report 1 kinds
    # have `accuracy_report` instead. `superseded_by` and `corrected` are
    # semicolon-joined, empty when there is nothing to say.
    "chi", "superseded_by", "corrected",
]
METRIC_KEYS = {"extracted", "new_pairs", "correct", "correct_heldout",
               "accuracy_report", "precision_extracted", "final_mapped",
               "final_correct", "chi"}
# What `--summary` averages when `--value` is not given: the first of these any
# selected row has. A fixed default matches nothing on one half of the archive.
VALUE_DEFAULTS = ("accuracy_report", "chi")


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
        if field not in fields:
            fields.append(field)
    return fields


def superseded_ids(rec):
    """The run_ids `rec` names in `supersedes`. A str in practice; a list is
    tolerated so a hand-written record cannot crash the whole table."""
    sup = rec.get("supersedes")
    if isinstance(sup, str):
        return [sup] if sup else []
    if isinstance(sup, (list, tuple)):
        return [x for x in sup if isinstance(x, str) and x]
    return []


def rows(archive, kinds=None, include_failed=True, include_superseded=False):
    corrections = load_corrections(archive)
    recs = []
    for rec_path in sorted(archive.glob("*/record.json")):
        try:
            rec = json.loads(rec_path.read_text())
        except (OSError, ValueError) as exc:
            print("  skipping unreadable {}: {}".format(rec_path, exc),
                  file=sys.stderr)
            continue
        recs.append((rec, apply_corrections(rec, corrections)))

    # Both checks below read every record, before --kind or --ok-only drop any,
    # so they hold whatever the filters are.
    seen = set(rec.get("run_id") for rec, _ in recs)
    for run_id in sorted(set(corrections) - seen):
        fields = sorted(set(c[0] for c in corrections[run_id]))
        print("  correction for {} ({}): no record has that run_id in {}"
              .format(run_id, ", ".join(fields), archive), file=sys.stderr)
    # {old run_id: [run_ids that supersede it]}. A chain A <- B <- C needs no
    # walking: A and B each have a superseder, so both are out.
    superseded_by = {}
    for rec, _ in recs:
        for old in superseded_ids(rec):
            superseded_by.setdefault(old, []).append(rec.get("run_id"))

    hidden = 0
    for rec, corrected in recs:
        if kinds and rec.get("kind") not in kinds:
            continue
        if not include_failed and rec.get("status_exit") != "ok":
            continue
        by = [x for x in superseded_by.get(rec.get("run_id"), ()) if x]
        if by and not include_superseded:
            hidden += 1
            continue
        metrics = rec.get("metrics") or {}
        row = {}
        for col in COLUMNS:
            if col in METRIC_KEYS:
                row[col] = metrics.get(col)
            elif col == "git":
                g = rec.get("git") or {}
                row[col] = (g.get("head", "") +
                            ("+dirty" if g.get("dirty") else ""))
            elif col == "superseded_by":
                row[col] = ";".join(by)
            elif col == "corrected":
                row[col] = ";".join(corrected)
            else:
                row[col] = rec.get(col)
        # seed_draw is the seed-based name for what the record stores as `seed`;
        # keep both columns populated so a filter on either works.
        if row["seed_draw"] is None and rec.get("kind") == "report1_seed_based":
            row["seed_draw"] = rec.get("seed")
        yield row
    if hidden:
        print("  left out {} superseded record(s); --include-superseded keeps "
              "them".format(hidden), file=sys.stderr)


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
    ap = argparse.ArgumentParser(description="Tabulate run records.")
    ap.add_argument("--archive", default=str(ARCHIVE),
                    help="Record archive (default results/runs/).")
    ap.add_argument("--kind", nargs="+", default=None,
                    help="Filter by record kind, e.g. report1_seed_based "
                         "report1_seed_free_integrated adv_gd. Default: every "
                         "kind in the archive.")
    ap.add_argument("--include-superseded", action="store_true",
                    help="Keep records that a later record names in "
                         "`supersedes`. Off by default: they were found "
                         "invalid. Their superseded_by column is filled in.")
    ap.add_argument("--ok-only", action="store_true",
                    help="Drop runs whose status_exit is not ok. Off by "
                         "default: a cell that died is data about where it got.")
    ap.add_argument("--out", default=None, help="Write CSV here (default stdout).")
    ap.add_argument("--summary", nargs="+", default=None, metavar="KEY",
                    help="Group by these columns and print n/mean/min/max of "
                         "--value instead of the rows.")
    ap.add_argument("--value", default=None,
                    help="Column to summarise. Default: accuracy_report if any "
                         "selected row has it (Report 1), else chi (Adv-GD "
                         "suite).")
    args = ap.parse_args(argv)

    kinds = args.kind
    archive = Path(args.archive)
    if not archive.is_dir():
        print("no archive at {}".format(archive), file=sys.stderr)
        return 1
    data = list(rows(archive, kinds, include_failed=not args.ok_only,
                     include_superseded=args.include_superseded))
    if not data:
        print("no matching records in {}".format(archive), file=sys.stderr)
        return 1

    if args.summary:
        unknown = [k for k in args.summary if k not in COLUMNS]
        if unknown:
            print("unknown summary key(s): {}\navailable: {}".format(
                ", ".join(unknown), ", ".join(COLUMNS)), file=sys.stderr)
            return 2
        value = args.value
        if value is None:
            value = next((v for v in VALUE_DEFAULTS
                          if any(r[v] is not None for r in data)),
                         VALUE_DEFAULTS[-1])
        if value not in COLUMNS:
            print("unknown --value column: {}\navailable: {}".format(
                value, ", ".join(COLUMNS)), file=sys.stderr)
            return 2
        have = sum(1 for r in data if r[value] is not None)
        if not have:
            usable = [c for c in COLUMNS if (c in METRIC_KEYS or c == "seconds")
                      and any(r[c] is not None for r in data)]
            print("warning: column '{}' is None for all {} selected records, "
                  "so the summary is empty; columns with values here: {}"
                  .format(value, len(data), ", ".join(usable) or "none"),
                  file=sys.stderr)
        table = summarise(data, args.summary, value)
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
        print("\n{} records, {} with {}, {} not ok, {} groups, {} with "
              "corrected fields".format(
                  len(data), have, value, failed, len(table),
                  sum(1 for r in data if r["corrected"])))
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
                len(data), args.out, sum(1 for r in data if r["corrected"])),
                file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
