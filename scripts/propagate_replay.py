#!/usr/bin/env python3
"""Replay Phase-3 `propagate` from stored anchors and compare with the stored mapping hash.

Supported kind: ignition_grid. For each record (results/runs/<run_id>/record.json) the
anchors file and the stored mapping sha256 come from the record's outputs/metrics, the
pair dir from the record's `pair_dir`, and candidates are {} when `candidate_mode` is
"empty" else {a: [u]} per anchor (ignition_grid.py, run_one). The graphs are loaded with
adv_gd.load_pair, and the mapping is written as "<a> <u>\\n" per item, as
ignition_grid.py does, then hashed with provenance.sha256_file.

adv_gd_w_control is NOT supported: its propagate call (w_control.py, _execute) receives
the refined anchors and candidates returned by align_from_seeds, while the only anchors
file on disk is the seed set (seed_anchors_*.txt), so the inputs are not recoverable.

Usage (run in the container, from the repo root):
    python3 scripts/propagate_replay.py --records ID1 ID2 ...
    python3 scripts/propagate_replay.py --sample 5 --kind ignition_grid [--seed 0]
    python3 scripts/propagate_replay.py --sample 5 --old /tmp/.../propagation_old.py
Use PYTHONHASHSEED=0 as the original runs did (not needed for correctness: no set
iteration order reaches the output). Exit status 1 on any mismatch, 2 on setup error.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import random
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDS_DIR = REPO_ROOT / "results" / "runs"


def load_module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_records(kind, records_dir):
    out = []
    for p in sorted(records_dir.glob("*/record.json")):
        try:
            with p.open() as fh:
                if json.load(fh).get("kind") == kind:
                    out.append(p.parent.name)
        except (OSError, ValueError):
            continue
    return out


def repo_path(p):
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def write_mapping(mapping, path):
    # Same writer as ignition_grid.py run_one.
    path.write_text("".join(f"{a} {u}\n" for a, u in mapping.items()))


def timed_propagate(mod, Ga, Gu, anchors, candidates, theta, support):
    buf = io.StringIO()
    t = time.time()
    with contextlib.redirect_stderr(buf):
        m = mod.propagate(Ga, Gu, anchors, candidates, theta=theta,
                          support=support, verbose=True)
    return m, buf.getvalue(), time.time() - t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="+", metavar="RUN_ID")
    ap.add_argument("--sample", type=int, metavar="N",
                    help="random sample of N records of --kind (fixed seed)")
    ap.add_argument("--kind", default="ignition_grid")
    ap.add_argument("--seed", type=int, default=0, help="sample seed (default 0)")
    ap.add_argument("--old", metavar="PATH",
                    help="pre-change propagation.py; also run it and compare times "
                         "and mappings")
    ap.add_argument("--records-dir", default=str(RECORDS_DIR))
    args = ap.parse_args(argv)
    if bool(args.records) == bool(args.sample):
        ap.error("give exactly one of --records or --sample")
    if args.kind != "ignition_grid":
        ap.error("only kind ignition_grid is replayable (see --help)")

    records_dir = Path(args.records_dir)

    # Imports that need the repo and its dependencies happen only now, so
    # --help works without loading anything.
    sys.path.insert(0, str(REPO_ROOT / "src" / "adv_gd"))
    from adv_gd import load_pair                       # noqa: E402
    from provenance import sha256_file                 # noqa: E402
    import propagation as new_mod                      # noqa: E402
    old_mod = load_module_from(args.old, "propagation_old") if args.old else None

    ids = args.records
    if args.sample:
        allr = find_records(args.kind, records_dir)
        ids = random.Random(args.seed).sample(allr, min(args.sample, len(allr)))
    # Group by pair dir so each pair is loaded once.
    recs = []
    for rid in ids:
        rp = records_dir / rid / "record.json"
        if not rp.exists():
            print(f"{rid}: no record.json", file=sys.stderr)
            return 2
        with rp.open() as fh:
            rec = json.load(fh)
        if rec.get("kind") != "ignition_grid":
            print(f"{rid}: kind {rec.get('kind')!r} is not replayable", file=sys.stderr)
            return 2
        recs.append((rid, rec))
    recs.sort(key=lambda x: x[1]["pair_dir"])

    bad = 0
    cache_key, cache = None, None
    tmp = Path(tempfile.mkdtemp(prefix="propagate_replay_"))
    hdr = ["run_id", "stored_sha256", "replay_sha256", "result",
           "mapped_stored", "mapped_replay", "new_s"]
    if old_mod:
        hdr += ["old_s", "old==new", "speedup"]
    print("\t".join(hdr))
    for rid, rec in recs:
        outs = {Path(o["path"]).name: o for o in rec["outputs"]}
        anc = next(o for n, o in outs.items() if n.startswith("anchors_"))
        mp = next(o for n, o in outs.items() if n.startswith("out_mapping_"))
        stored = rec["metrics"].get("mapping_sha256") or mp["sha256"]
        anchors = []
        for line in repo_path(anc["path"]).read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                anchors.append((parts[0], parts[1]))
        if sha256_file(repo_path(anc["path"])) != anc["sha256"]:
            print(f"{rid}: anchors file hash differs from record", file=sys.stderr)
            bad += 1
            continue
        candidates = ({} if rec["candidate_mode"] == "empty"
                      else {a: [u] for a, u in anchors})
        if cache_key != rec["pair_dir"]:
            cache = load_pair(repo_path(rec["pair_dir"]))
            cache_key = rec["pair_dir"]
        Ga, Gu = cache[0], cache[1]
        theta, support = rec["theta"], rec["eccen_support"]

        mapping, _, t_new = timed_propagate(new_mod, Ga, Gu, anchors, candidates,
                                            theta, support)
        out = tmp / f"out_mapping_replay_{rid}.txt"
        write_mapping(mapping, out)
        got = sha256_file(out)
        ok = got == stored
        bad += not ok
        row = [rid, stored[:16], got[:16], "MATCH" if ok else "MISMATCH",
               str(rec["metrics"].get("mapped")), str(len(mapping)), f"{t_new:.1f}"]
        if old_mod:
            m_old, _, t_old = timed_propagate(old_mod, Ga, Gu, anchors, candidates,
                                              theta, support)
            same = list(m_old.items()) == list(mapping.items())
            bad += not same
            row += [f"{t_old:.1f}", "equal" if same else "DIFFERENT",
                    f"{t_old / t_new:.1f}x" if t_new else "n/a"]
        print("\t".join(row), flush=True)
        out.unlink()
    print(f"{len(recs)} records, {bad} problems", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
