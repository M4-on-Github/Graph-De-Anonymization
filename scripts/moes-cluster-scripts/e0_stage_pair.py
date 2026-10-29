#!/usr/bin/env python3
"""Stage Report 1's own graph pair as a flat pair directory, for E0 arm (b).

`centrality_rank_match.py` reads a pair through `adv_gd.load_pair`, which wants
a directory holding G1.edgelist, G2.edgelist, mapping.txt and stats.json.
`data/labeled_dev/` has the first three under other names and no stats.json, and
`report1_cell.py`'s loader is no use here (it wants a `labeled_dev/`
subdirectory and four exact names). This builds the directory:

    runs/e0_labeled_dev/
        G1.edgelist -> data/labeled_dev/validation_G1.edgelist     (symlink)
        G2.edgelist -> data/labeled_dev/validation_G2.edgelist     (symlink)
        mapping.txt -> data/labeled_dev/validation_seed_mapping.txt (symlink)
        stats.json                                                  (written)

The one number that matters in stats.json is `matchable_nodes`. Without it
`propagation.evaluate` divides chi by len(truth), the 27,770 lines of the
mapping file (propagation.py:170); with it, by the 26,948 pairs whose two nodes
exist in the two graphs, which is the denominator of Report 1's "valid" pairs
and of arm (a)'s A_full. It is counted from the data here, and the script
refuses if the count is not the 26,948 that Log 3 fixes. `beta` is null: this
pair has no noise parameter. Nothing in stats.json varies from run to run (no
timestamp), because the file is hashed into every record that reads it.

Idempotent and safe to run from several array tasks at once: an existing link or
stats.json that already says the same thing is left alone, and one that says
something else is an error, never overwritten.

Python 3.6 compatible and standard library only, like grid.py and collect.py, so
it can also be run on the head node (it reads two 7 MB files, a couple of
seconds).

    python3 scripts/moes-cluster-scripts/e0_stage_pair.py
    python3 scripts/moes-cluster-scripts/e0_stage_pair.py --pair-dir /elsewhere/e0_labeled_dev
"""
import argparse
import json
import os
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# Log 3, E0 "New code": the staged names, and what they point at.
LINKS = (
    ("G1.edgelist", "validation_G1.edgelist"),
    ("G2.edgelist", "validation_G2.edgelist"),
    ("mapping.txt", "validation_seed_mapping.txt"),
)
EXPECT_MATCHABLE = 26948


def read_nodes(path):
    """Node labels of an edge list: every token, two per line, as strings."""
    nodes = set()
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            parts = line.split()
            if len(parts) != 2:
                raise SystemExit("%s line %d: expected 'u v', got %r"
                                 % (path, n, line.rstrip("\n")))
            nodes.add(parts[0])
            nodes.add(parts[1])
    return nodes


def read_pairs(path):
    pairs = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            parts = line.split()
            if len(parts) != 2:
                raise SystemExit("%s line %d: expected 'u v', got %r"
                                 % (path, n, line.rstrip("\n")))
            pairs.append((parts[0], parts[1]))
    return pairs


def build_stats(data_dir, expect_matchable=EXPECT_MATCHABLE):
    """The stats.json content, counted from the data. Raises on a wrong count."""
    src = os.path.join(data_dir, "labeled_dev")
    missing = [s for _, s in LINKS if not os.path.isfile(os.path.join(src, s))]
    if missing:
        raise SystemExit("not found under %s: %s" % (src, ", ".join(missing)))
    g1 = read_nodes(os.path.join(src, LINKS[0][1]))
    g2 = read_nodes(os.path.join(src, LINKS[1][1]))
    pairs = read_pairs(os.path.join(src, LINKS[2][1]))
    matchable = sum(1 for u, v in pairs if u in g1 and v in g2)
    if expect_matchable is not None and matchable != expect_matchable:
        raise SystemExit(
            "matchable_nodes counted from %s is %d, but %d was expected (Log 3 "
            "fixes %d for Report 1's own pair). These may not be Report 1's "
            "files; refusing to stage them."
            % (src, matchable, expect_matchable, EXPECT_MATCHABLE))
    return {
        "beta": None,
        "matchable_nodes": matchable,
        "nodes_a": len(g1),
        "nodes_u": len(g2),
        "mapping_pairs": len(pairs),
        "dataset": "labeled_dev",
        "matchable_definition":
            "mapping pairs (u, v) with u a node of G1 and v a node of G2",
        "source": "written by scripts/moes-cluster-scripts/e0_stage_pair.py "
                  "for E0 (Log 3); not generator output. This pair has no "
                  "noise parameter, so beta is null.",
    }


def _link(pair_dir, name, target):
    link = os.path.join(pair_dir, name)
    want = os.path.realpath(target)
    if not os.path.lexists(link):
        try:
            os.symlink(target, link)
            return "linked"
        except OSError:
            if not os.path.lexists(link):      # not a lost race: a real failure
                raise
    if not os.path.islink(link) or os.path.realpath(link) != want:
        raise SystemExit("%s exists and is not a link to %s; refusing to "
                         "overwrite it" % (link, target))
    return "kept"


def _stats(pair_dir, stats):
    path = os.path.join(pair_dir, "stats.json")
    text = json.dumps(stats, indent=2, sort_keys=True) + "\n"
    if os.path.lexists(path):
        with open(path) as fh:
            have = json.load(fh)
        if have != stats:
            raise SystemExit("%s exists and differs from what this script "
                             "writes; refusing to overwrite it" % path)
        return "kept"
    tmp = os.path.join(pair_dir, ".stats.%s.tmp" % uuid.uuid4().hex)
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
    return "written"


def stage(pair_dir, data_dir, expect_matchable=EXPECT_MATCHABLE):
    """Create or verify the pair directory. Returns {name: 'linked'|'kept'|...}."""
    pair_dir = os.path.abspath(pair_dir)
    data_dir = os.path.abspath(data_dir)
    stats = build_stats(data_dir, expect_matchable)
    os.makedirs(pair_dir, exist_ok=True)
    done = {}
    for name, src in LINKS:
        done[name] = _link(pair_dir, name,
                           os.path.join(data_dir, "labeled_dev", src))
    done["stats.json"] = _stats(pair_dir, stats)
    return done, stats


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stage data/labeled_dev as a flat pair directory.")
    ap.add_argument("--pair-dir", default=os.path.join(REPO_ROOT, "runs",
                                                         "e0_labeled_dev"),
                    help="Where to build it (default runs/e0_labeled_dev/).")
    ap.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"),
                    help="Directory holding labeled_dev/ (default data/).")
    ap.add_argument("--expect-matchable-nodes", type=int,
                    default=EXPECT_MATCHABLE,
                    help="Refuse unless the data gives exactly this many "
                         "matchable nodes (default %d, Log 3's value)."
                         % EXPECT_MATCHABLE)
    args = ap.parse_args(argv)

    done, stats = stage(args.pair_dir, args.data_dir,
                        args.expect_matchable_nodes)
    for name in [n for n, _ in LINKS] + ["stats.json"]:
        print("  %-12s %s" % (name, done[name]))
    print("staged %s: matchable_nodes %d (G1 %d nodes, G2 %d nodes, %d mapping "
          "pairs)" % (os.path.abspath(args.pair_dir), stats["matchable_nodes"],
                      stats["nodes_a"], stats["nodes_u"],
                      stats["mapping_pairs"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
