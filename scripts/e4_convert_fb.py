#!/usr/bin/env python3
"""Convert an uploaded Facebook (FB) graph file into the edge list the generator reads.

E4a "Upload check" (experiment_log_03, section E4): when the user uploads the paper's FB
graph, before any run (1) record the file name, format, size and SHA-256; (2) reduce it to
an undirected simple graph -- self-loops and duplicate edges dropped, as
`src/beta_noise_generator/perturbation.py:load_parent` does -- and record nodes, edges,
components and the giant share; (3) compare with Table 1's FB row (8,758 nodes, 405,450
edges) as a FILE-IDENTITY check only, and report any mismatch. This script does all three
and writes the edge list that `sweep.PARENTS["fb"]` points at.

Input formats
  * MATLAB .mat in the fb100 release's format (Traud, Mucha & Porter): a sparse symmetric
    adjacency matrix in variable `A`, plus `local_info`. Read with scipy.io.loadmat
    (MATLAB v4/v5/v6/v7). v7.3 files are HDF5 and need h5py, which the project container
    does not have; the script says so and stops rather than guessing.
  * A plain edge list, optionally gzip-compressed: one edge per line, the two node ids
    separated by whitespace and/or a comma. Blank lines and lines starting with '#' or '%'
    are comments (the same rule as `load_parent`); columns after the second are ignored.

Output
  * the edge list: one "u v" line per undirected edge, sorted, so the same graph always
    gives the same bytes (the generator's draw depends on edge order, and the pair
    records hash this file). Node ids are the file's own tokens; for a .mat they are the
    MATLAB row numbers of `A`, 1-based, i.e. the row numbers of `local_info` too.
    Nodes with no edge do not appear, exactly as `load_parent` would never see them.
  * an identity JSON (default: next to the edge list) and the same report on stdout.

A mismatch with Table 1 is reported, never fatal: the exit status is 0 unless the file
cannot be read or the output cannot be written. Table 1's counts are the source file's
advertised counts, not those of the undirected simple graph (adv_gd_protocol.md:64-68), so
they identify the file; they do not validate a loaded graph.

    python3 scripts/e4_convert_fb.py Yale4.mat                 # -> data/fb/fb.edgelist
    python3 scripts/e4_convert_fb.py Yale4.mat --check-only    # report, write nothing
    python3 scripts/e4_convert_fb.py --self-test

Run it in the project container (numpy and scipy are needed for .mat input).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Must equal sweep.PARENTS["fb"] (src/adv_gd/sweep.py); --self-test checks it.
DEFAULT_OUT = REPO_ROOT / "data" / "fb" / "fb.edgelist"

# Table 1, printed p.749 of docs/SeedFree_Li.etal.pdf (text re-read from the PDF
# 2026-10-28): "FB 8,758 405,450 16 1000 400 5,000". adv_gd_protocol.md:55-59 repeats it.
PAPER_TABLE1_FB = {"nodes": 8758, "edges": 405450}
# The spec's recollection that the Yale file has 8,578 nodes. From memory, unverified;
# the upload settles it (experiment_log_03, "FB: what the paper says").
RECOLLECTED_YALE_NODES = 8578
NEAR_FRACTION = 0.01     # "near" in the verdict: within 1 % on edges and on nodes

SCHEMA = "e4_fb_identity/1"


class ConvertError(Exception):
    """A problem with the input or output that stops the conversion (reported, exit 1)."""


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _tok_key(tok: str):
    """Sort key: integer ids numerically first, then anything else as text.

    The token itself is the last element so that "007" and "7", which are distinct nodes
    to the generator (it keeps ids as strings), still sort in a fixed order.
    """
    try:
        return (0, int(tok), tok)
    except ValueError:
        return (1, 0, tok)


def _pct_diff(ours: int, ref: int) -> str:
    return f"{ours - ref:+,} ({(ours - ref) / ref:+.2%})"


# ---------------------------------------------------------------------------------------
# format detection
# ---------------------------------------------------------------------------------------

def detect_format(path: Path, forced: str = "auto") -> str:
    """'mat', 'edgelist' (plain text) or 'edgelist.gz'. Raises ConvertError otherwise."""
    with open(path, "rb") as fh:
        head = fh.read(4096)
    is_gz = head[:2] == b"\x1f\x8b"
    is_mat_text = head.startswith(b"MATLAB")
    if head[:4] == b"PK\x03\x04":
        raise ConvertError(f"{path.name} is a zip archive; extract the Yale .mat file "
                           "from it and pass that")
    if head.startswith(b"\x89HDF\r\n\x1a\n") or (is_mat_text and b"MATLAB 7.3" in head[:128]):
        raise ConvertError(f"{path.name} is a MATLAB v7.3 (HDF5) file. scipy.io.loadmat "
                           "cannot read it, and h5py is not in the project container. "
                           "Re-save it from MATLAB with save(file, '-v7') (or '-v6'), "
                           "or export an edge list.")
    if forced == "mat":
        return "mat"
    if forced == "edgelist":
        return "edgelist.gz" if is_gz else "edgelist"
    if is_mat_text or path.suffix.lower() == ".mat":
        return "mat"
    if is_gz:
        return "edgelist.gz"
    if b"\x00" in head:
        raise ConvertError(f"{path.name} is binary and not a recognised MAT file; "
                           "pass --format mat or --format edgelist to force a reading")
    return "edgelist"


# ---------------------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------------------

def read_edgelist(path: Path, gz: bool) -> tuple[dict, list[tuple[str, str]]]:
    """Parse a plain edge list.

    Returns (raw counts, the unique undirected pairs as (u, v) token pairs with
    key(u) < key(v), sorted).
    """
    opener = gzip.open if gz else open
    lines = blank = comments = malformed = data = loops = extra = non_int = 0
    first_non_int = None
    first_data: list[str] = []
    keys: dict[str, tuple] = {}
    pairs: set[tuple[str, str]] = set()
    all_ids: set[str] = set()

    def key(tok):
        k = keys.get(tok)
        if k is None:
            k = keys[tok] = _tok_key(tok)
        return k

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            lines += 1
            line = line.strip()
            if not line:
                blank += 1
                continue
            if line[0] in "#%":
                comments += 1
                continue
            parts = [t for t in re.split(r"[,\s]+", line) if t]
            if len(parts) < 2:
                malformed += 1
                continue
            data += 1
            if len(first_data) < 3:
                first_data.append(line[:80])
            if len(parts) > 2:
                extra += 1
            u, v = parts[0], parts[1]
            all_ids.add(u)
            all_ids.add(v)
            if key(u)[0] or key(v)[0]:
                non_int += 1
                if first_non_int is None:
                    first_non_int = [lines, line[:80]]
            if u == v:
                loops += 1
                continue
            pairs.add((u, v) if key(u) < key(v) else (v, u))

    edges = sorted(pairs, key=lambda e: (key(e[0]), key(e[1])))
    raw = {
        "lines": lines, "blank_lines": blank, "comment_lines": comments,
        "malformed_lines": malformed, "edge_lines": data,
        "self_loop_lines": loops, "lines_with_extra_columns": extra,
        "lines_with_non_integer_ids": non_int,
        "first_line_with_non_integer_id": first_non_int,
        "first_edge_lines": first_data,
        "raw_nodes": len(all_ids),
        "raw_edge_entries": data,
        "self_loops": loops,
        "duplicate_or_reciprocal_absorbed": data - loops - len(edges),
    }
    return raw, edges


def read_mat(path: Path, var: str | None) -> tuple[dict, list[tuple[int, int]]]:
    """Read the adjacency matrix of an fb100-style .mat file.

    Returns (raw counts, the unique undirected off-diagonal pairs as (u, v) with u < v,
    the 1-based MATLAB row numbers, sorted).
    """
    try:
        import numpy as np
        from scipy import io as sio, sparse
    except ImportError as exc:                      # not the case in the container
        raise ConvertError(
            f".mat input needs numpy and scipy ({exc}). The project container "
            "(graph_deanon.sif) has both; run this script inside it.") from exc

    try:
        whos = sio.whosmat(str(path), appendmat=False)
    except NotImplementedError as exc:
        raise ConvertError(f"scipy cannot read {path.name}: {exc}") from exc
    except Exception as exc:
        raise ConvertError(f"{path.name} is not a readable MATLAB file: "
                           f"{type(exc).__name__}: {exc}") from exc
    names = {n: (tuple(shape), cls) for n, shape, cls in whos}
    listing = [(n, s, c) for n, (s, c) in names.items()]

    if var is None:
        if "A" in names:
            var = "A"
        else:
            square = [n for n, (shape, cls) in names.items()
                      if len(shape) == 2 and shape[0] == shape[1] > 1
                      and cls not in ("char", "cell", "struct")]
            if len(square) != 1:
                raise ConvertError(
                    f"no variable `A`, and the square numeric variables are "
                    f"{square or 'none'}; variables in the file: {listing}. "
                    "Pass --var NAME.")
            var = square[0]
    if var not in names:
        raise ConvertError(f"no variable {var!r}; variables in the file: {listing}")
    shape, cls = names[var]
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ConvertError(f"variable {var!r} has shape {shape}; an adjacency matrix "
                           "must be square")
    n = int(shape[0])

    # scipy 1.18 warns that loadmat will return sparse arrays instead of sparse matrices
    # from 1.20. Everything below works on either, so the warning is only noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        M = sio.loadmat(str(path), variable_names=[var], appendmat=False)[var]
    coo = (M if sparse.issparse(M) else sparse.coo_matrix(np.asarray(M))).tocoo()
    stored = int(coo.nnz)
    coo.eliminate_zeros()
    nnz = int(coo.nnz)
    row = coo.row.astype(np.int64)
    col = coo.col.astype(np.int64)
    off = row != col
    diag = int((~off).sum())
    lo = np.minimum(row[off], col[off])
    hi = np.maximum(row[off], col[off])
    code, counts = np.unique(lo * n + hi, return_counts=True)
    if (counts > 2).any():                           # cannot happen without duplicates
        raise ConvertError("duplicate entries survived loading; refusing to guess")
    one_way = int((counts == 1).sum())
    both_ways = int((counts == 2).sum())
    local = names.get("local_info")
    raw = {
        "variable": var, "matlab_class": cls, "matrix_shape": [n, n],
        "stored_entries": stored, "explicit_zero_entries": stored - nnz,
        "nonzero_entries": nnz,
        "diagonal_entries": diag,
        "off_diagonal_entries": nnz - diag,
        "values_all_one": bool(np.all(coo.data == 1)),
        "values_not_one": int(np.count_nonzero(coo.data != 1)),
        "pairs_present_both_directions": both_ways,
        "pairs_present_one_direction_only": one_way,
        "symmetric": one_way == 0,
        "variables_in_file": [{"name": nm, "shape": list(s), "class": c}
                              for nm, (s, c) in names.items()],
        "local_info_shape": list(local[0]) if local else None,
        "local_info_rows_match_matrix": (local[0][0] == n) if local else None,
        "raw_nodes": n,
        "raw_edge_entries": nnz,
        "self_loops": diag,
        "duplicate_or_reciprocal_absorbed": both_ways,
    }
    edges = list(zip((code // n + 1).tolist(), (code % n + 1).tolist()))
    return raw, edges


# ---------------------------------------------------------------------------------------
# reduction and components
# ---------------------------------------------------------------------------------------

def components(nodes: list, edges: list) -> dict:
    """Union-find over the reduced graph; every endpoint of `edges` is in `nodes`."""
    index = {x: i for i, x in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for u, v in edges:
        a, b = find(index[u]), find(index[v])
        if a != b:
            parent[a] = b
    sizes: dict[int, int] = {}
    for i in range(len(nodes)):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    ordered = sorted(sizes.values(), reverse=True)
    return {
        "count": len(ordered),
        "giant_nodes": ordered[0] if ordered else 0,
        "giant_share": (ordered[0] / len(nodes)) if nodes else 0.0,
        "second_largest": ordered[1] if len(ordered) > 1 else 0,
    }


# ---------------------------------------------------------------------------------------
# identity check
# ---------------------------------------------------------------------------------------

def identity_check(raw_nodes: int, nodes: int, edges: int) -> dict:
    """Compare with Table 1's FB row and the recollected Yale node count. Never raises."""
    t1n, t1e = PAPER_TABLE1_FB["nodes"], PAPER_TABLE1_FB["edges"]
    rec = RECOLLECTED_YALE_NODES
    rows = []

    def add(label, ours, ref, source):
        rows.append({"quantity": label, "ours": ours, "reference": ref,
                     "source": source, "difference": ours - ref, "exact": ours == ref})

    raw_label = "nodes: matrix dimension / distinct ids"
    red_label = "nodes: with at least one edge (as the generator sees them)"
    add("edges (undirected simple graph)", edges, t1e, "Table 1")
    add(raw_label, raw_nodes, t1n, "Table 1")
    add(red_label, nodes, t1n, "Table 1")
    add(raw_label, raw_nodes, rec, "recollection of the Yale file, unverified")
    add(red_label, nodes, rec, "recollection of the Yale file, unverified")

    edges_exact = edges == t1e
    on_table1 = [lab for lab, x in (("matrix dimension", raw_nodes), ("nodes with an edge", nodes))
                 if x == t1n]
    on_recollection = [lab for lab, x in (("matrix dimension", raw_nodes), ("nodes with an edge", nodes))
                       if x == rec]
    near_edges = abs(edges - t1e) <= NEAR_FRACTION * t1e
    near_nodes = any(abs(x - r) <= NEAR_FRACTION * r
                     for x in (raw_nodes, nodes) for r in (t1n, rec))
    counts = f"{raw_nodes:,} nodes ({nodes:,} with an edge) and {edges:,} edges"
    def which(labels):                    # "the X equals" / "the X and Y both equal"
        return (f"the {labels[0]} equals" if len(labels) == 1
                else f"both the {' and the '.join(labels)} equal")

    if edges_exact and on_table1:
        status = "match"
        verdict = (f"MATCH: {counts}; edges equal Table 1's {t1e:,} and "
                   f"{which(on_table1)} Table 1's {t1n:,}.")
    elif edges_exact and on_recollection:
        status = "match_recollection"
        verdict = (f"MATCH ON EDGES, NODES DIFFER FROM TABLE 1: {counts}; edges equal "
                   f"Table 1's {t1e:,}, {which(on_recollection)} the recollected Yale "
                   f"count {rec:,}, not Table 1's printed {t1n:,} (8,758 against 8,578 "
                   "reads as a transposition in Table 1). Report the node-count "
                   "difference to the user.")
    elif edges_exact:
        status = "edges_only"
        verdict = (f"PARTIAL: {counts}; edges equal Table 1's {t1e:,} but the node count "
                   f"equals neither {t1n:,} nor {rec:,}. Report to the user.")
    elif near_edges and near_nodes:
        status = "near"
        verdict = (f"NEAR, NOT EXACT: {counts}; edges are {_pct_diff(edges, t1e)} against "
                   f"Table 1's {t1e:,}. Could be the right school under another edge "
                   "convention, or a different school. Report to the user before "
                   "running anything.")
    else:
        status = "mismatch"
        verdict = (f"MISMATCH: {counts} against Table 1's {t1n:,} nodes and {t1e:,} edges. "
                   "This does not look like the paper's FB graph (Yale, 2005); it may be "
                   "another school's file. Report to the user before running anything.")
    return {"table1_fb": PAPER_TABLE1_FB, "recollected_yale_nodes": rec,
            "near_fraction": NEAR_FRACTION, "rows": rows,
            "status": status, "verdict": verdict,
            "caveat": "Table 1's counts are the source file's advertised counts, not those "
                      "of the undirected simple graph (adv_gd_protocol.md:64-68): they "
                      "identify the file, they do not validate a loaded graph."}


# ---------------------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------------------

def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(tmp, 0o666 & ~mask)             # mkstemp makes the file 0600
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def convert(src: Path, out: Path | None, fmt: str = "auto", var: str | None = None,
            force: bool = False) -> dict:
    """Read `src`, reduce it, write `out` (unless None) and return the identity dict."""
    src = Path(src)
    if not src.is_file():
        raise ConvertError(f"no such file: {src}")
    kind = detect_format(src, fmt)
    source = {"file_name": src.name, "path": str(src.resolve()), "format": kind,
              "size_bytes": src.stat().st_size, "sha256": sha256_file(src)}

    if kind == "mat":
        raw, edges = read_mat(src, var)
        nodes = sorted({x for e in edges for x in e})
        node_ids = "MATLAB row number of A, 1-based (the row number in local_info)"
    else:
        raw, edges = read_edgelist(src, kind.endswith(".gz"))
        nodes = sorted({x for e in edges for x in e}, key=_tok_key)
        node_ids = "the file's own id tokens"
    if not edges:
        raise ConvertError("the file holds no edge after dropping self-loops")

    raw_nodes = raw["raw_nodes"]
    reduced = {
        "nodes": len(nodes), "edges": len(edges),
        "self_loops_dropped": raw["self_loops"],
        "duplicate_or_reciprocal_absorbed": raw["duplicate_or_reciprocal_absorbed"],
        "nodes_with_no_edge_not_written": raw_nodes - len(nodes),
        "mean_degree": 2 * len(edges) / len(nodes),
        "edges_per_node": len(edges) / len(nodes),
    }
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source, "raw": raw, "reduced": reduced,
        "components": components(nodes, edges),
        "node_ids": node_ids,
        "identity_check": identity_check(raw_nodes, len(nodes), len(edges)),
        "converted": None,
    }

    if out is not None:
        out = Path(out)
        text = "".join(f"{a} {b}\n" for a, b in edges)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Replacing the parent graph silently would orphan every pair record that hashed
        # the old bytes. The same bytes may be rewritten; different bytes need --force.
        if out.exists() and not force and sha256_file(out) != digest:
            raise ConvertError(
                f"{out} exists and differs from what this conversion would write "
                f"(existing sha256 {sha256_file(out)[:16]}..., new {digest[:16]}...). "
                "Pairs generated from it carry its hash in their records; pass --force "
                "to replace it.")
        write_atomic(out, text)
        result["converted"] = {
            "path": str(out.resolve()),
            "format": "edge list, 'u v' per line, undirected simple graph, sorted",
            "lines": len(edges), "size_bytes": out.stat().st_size,
            "sha256": sha256_file(out)}
    return result


# ---------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------

def render(res: dict) -> str:
    s, raw, red, comp, idc = (res["source"], res["raw"], res["reduced"],
                              res["components"], res["identity_check"])
    out: list[str] = []
    w = out.append
    w("FB upload check (E4a)")
    w(f"  file       {s['file_name']}")
    w(f"  format     {s['format']}")
    w(f"  size       {s['size_bytes']:,} bytes")
    w(f"  sha256     {s['sha256']}")
    w("raw")
    if s["format"] == "mat":
        w(f"  variable {raw['variable']!r} ({raw['matlab_class']}), shape "
          f"{raw['matrix_shape'][0]:,} x {raw['matrix_shape'][1]:,}")
        w(f"  nonzero entries {raw['nonzero_entries']:,} (diagonal {raw['diagonal_entries']:,}, "
          f"off-diagonal {raw['off_diagonal_entries']:,}); explicit zeros "
          f"{raw['explicit_zero_entries']:,}; entries not equal to 1: {raw['values_not_one']:,}")
        w(f"  unordered pairs present in both directions "
          f"{raw['pairs_present_both_directions']:,}, in one direction only "
          f"{raw['pairs_present_one_direction_only']:,} (symmetric: {raw['symmetric']})")
        li = raw["local_info_shape"]
        w("  local_info " + ("absent" if li is None else
                             f"shape {li}" + (" (rows match A)" if raw["local_info_rows_match_matrix"]
                                              else " (ROWS DO NOT MATCH A)")))
        w(f"  variables  {[v['name'] for v in raw['variables_in_file']]}")
    else:
        w(f"  lines {raw['lines']:,}: edge lines {raw['edge_lines']:,}, comment "
          f"{raw['comment_lines']:,}, blank {raw['blank_lines']:,}, malformed "
          f"{raw['malformed_lines']:,}")
        first = raw["first_line_with_non_integer_id"]
        w(f"  self-loop lines {raw['self_loop_lines']:,}; lines with extra columns "
          f"{raw['lines_with_extra_columns']:,}; lines with a non-integer id "
          f"{raw['lines_with_non_integer_ids']:,}"
          + (f" (first: line {first[0]}: {first[1]!r})" if first else ""))
        w(f"  first edge lines {raw['first_edge_lines']}")
    w(f"  raw nodes {raw['raw_nodes']:,}; raw edge entries {raw['raw_edge_entries']:,}")
    w("reduced (undirected simple graph, as the generator reads it)")
    w(f"  nodes {red['nodes']:,}   edges {red['edges']:,}   self-loops dropped "
      f"{red['self_loops_dropped']:,}   duplicate/reciprocal absorbed "
      f"{red['duplicate_or_reciprocal_absorbed']:,}   nodes with no edge not written "
      f"{red['nodes_with_no_edge_not_written']:,}")
    w(f"  mean degree {red['mean_degree']:.2f}   |E|/|V| {red['edges_per_node']:.2f} "
      "(the spec: 46.3 on Table 1's counts for FB, 12.7 for HepTh)")
    w(f"  components {comp['count']:,}   giant {comp['giant_nodes']:,} "
      f"({comp['giant_share']:.4f})   second largest {comp['second_largest']:,}")
    w("identity check against the paper's Table 1 FB row (file identity only)")
    for r in idc["rows"]:
        w(f"  {r['quantity']:<58s} ours {r['ours']:>9,}  ref {r['reference']:>9,}  "
          f"diff {r['difference']:+,}  {'exact' if r['exact'] else 'differs'}  [{r['source']}]")
    w(f"  {idc['verdict']}")
    w(f"  ({idc['caveat']})")
    if res["converted"]:
        c = res["converted"]
        w("converted")
        w(f"  {c['path']}")
        w(f"  {c['lines']:,} lines, {c['size_bytes']:,} bytes, sha256 {c['sha256']}")
    else:
        w("converted  nothing written (--check-only)")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------

def _self_test() -> int:
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    def raises(fn, needle):
        try:
            fn()
        except ConvertError as exc:
            return needle in str(exc), str(exc)[:90]
        return False, "no error raised"

    sys.path.insert(0, str(REPO_ROOT / "src" / "beta_noise_generator"))
    from perturbation import load_parent

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ---- plain edge list with every quirk the generator tolerates -------------------
        txt = td / "quirky.txt"
        txt.write_text("# header\n% another comment\n\n1 2\n2 1\n2,3\n3\t4 0.5\n4 4\n"
                       "5 6 7 8\n1 2\nlonely\n10,11\n")
        res = convert(txt, td / "q.edgelist")
        got = (td / "q.edgelist").read_text().split("\n")[:-1]
        check("edge list: dedupe, reciprocal, comma, tab, loop, malformed -> 5 edges, numeric order",
              got == ["1 2", "2 3", "3 4", "5 6", "10 11"], str(got))
        r = res["raw"]
        check("edge list: raw counts (2 comment, 1 blank, 1 malformed, 1 loop, 8 edge lines, 2 absorbed)",
              (r["comment_lines"], r["blank_lines"], r["malformed_lines"], r["self_loop_lines"],
               r["edge_lines"], r["duplicate_or_reciprocal_absorbed"]) == (2, 1, 1, 1, 8, 2), str(r))
        check("edge list: 8 nodes / 5 edges, 3 components, giant 4",
              (res["reduced"]["nodes"], res["reduced"]["edges"], res["components"]["count"],
               res["components"]["giant_nodes"]) == (8, 5, 3, 4), str(res["reduced"]))
        shuf = td / "shuf.txt"
        shuf.write_text("11,10\n6 5\n4 3\n3 2\n2 1\n")
        convert(shuf, td / "s.edgelist")
        check("edge list: the output bytes do not depend on input order or quirks",
              (td / "s.edgelist").read_bytes() == (td / "q.edgelist").read_bytes())
        with gzip.open(td / "quirky.txt.gz", "wt") as fh:
            fh.write(txt.read_text())
        convert(td / "quirky.txt.gz", td / "g.edgelist")
        check("edge list: gzip input gives the same bytes",
              (td / "g.edgelist").read_bytes() == (td / "q.edgelist").read_bytes())
        G = load_parent(td / "q.edgelist", verbose=False)
        check("the generator's load_parent reads the output to the same counts",
              (G.number_of_nodes(), G.number_of_edges()) == (8, 5))
        (td / "other.txt").write_text("1 2\n")
        passed, msg = raises(lambda: convert(td / "other.txt", td / "q.edgelist"), "--force")
        check("refuses to replace a different existing output without --force", passed, msg)
        convert(txt, td / "q.edgelist")
        convert(td / "other.txt", td / "q.edgelist", force=True)
        check("same bytes rewrite freely; --force replaces",
              (td / "q.edgelist").read_text() == "1 2\n")

        # ---- MATLAB ---------------------------------------------------------------------
        try:
            import numpy as np
            from scipy import io as sio, sparse
        except ImportError as exc:
            print(f"  [SKIP] .mat tests need numpy and scipy ({exc})")
            print("e4_convert_fb self-test " + ("passed (mat skipped)" if ok else "FAILED"))
            return 0 if ok else 1

        A = np.zeros((6, 6))                          # node 6 isolated, one self-loop at 2
        for i, j in [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]:
            A[i, j] = A[j, i] = 1
        A[1, 1] = 1
        sio.savemat(td / "sym.mat", {"A": sparse.csc_matrix(A), "local_info": np.ones((6, 7))})
        res = convert(td / "sym.mat", td / "m.edgelist")
        edges = (td / "m.edgelist").read_text().split("\n")[:-1]
        check("mat: symmetric A -> 5 unique edges, 1-based ids",
              edges == ["1 2", "1 3", "2 3", "3 4", "4 5"], str(edges))
        raw = res["raw"]
        check("mat: 11 nonzero = 10 off-diagonal + 1 diagonal; 5 pairs in both directions; symmetric",
              (raw["nonzero_entries"], raw["diagonal_entries"], raw["pairs_present_both_directions"],
               raw["symmetric"]) == (11, 1, 5, True), str(raw))
        check("mat: raw 6 nodes, reduced 5 (node 6 isolated), local_info rows match",
              (raw["raw_nodes"], res["reduced"]["nodes"], raw["local_info_rows_match_matrix"],
               res["reduced"]["nodes_with_no_edge_not_written"]) == (6, 5, True, 1))
        sio.savemat(td / "up.mat", {"A": sparse.csc_matrix(np.triu(A, 1))})
        res_u = convert(td / "up.mat", td / "u.edgelist")
        check("mat: upper triangle only gives the same bytes and is reported asymmetric",
              (td / "u.edgelist").read_bytes() == (td / "m.edgelist").read_bytes()
              and res_u["raw"]["symmetric"] is False
              and res_u["raw"]["pairs_present_one_direction_only"] == 5)
        sio.savemat(td / "named.mat", {"adj": sparse.csc_matrix(A), "other": np.ones((2, 3))})
        check("mat: without `A`, the one square variable is used",
              convert(td / "named.mat", None)["raw"]["variable"] == "adj")
        sio.savemat(td / "two.mat", {"X": sparse.csc_matrix(A), "Y": sparse.csc_matrix(A)})
        passed, msg = raises(lambda: convert(td / "two.mat", None), "--var")
        check("mat: two candidate variables -> clear error naming --var", passed, msg)
        check("mat: --var picks one", convert(td / "two.mat", None, var="Y")["raw"]["variable"] == "Y")
        sio.savemat(td / "rect.mat", {"A": sparse.csc_matrix(np.ones((3, 4)))})
        passed, msg = raises(lambda: convert(td / "rect.mat", None), "square")
        check("mat: non-square -> clear error", passed, msg)
        (td / "v73.mat").write_bytes(b"MATLAB 7.3 MAT-file, Platform: x" + b" " * 100
                                     + b"\x00" * 400 + b"\x89HDF\r\n\x1a\n")
        passed, msg = raises(lambda: convert(td / "v73.mat", None), "h5py")
        check("mat: v7.3 (HDF5) -> clear error naming h5py", passed, msg)
        (td / "z.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 50)
        passed, msg = raises(lambda: convert(td / "z.zip", None), "zip")
        check("zip archive -> clear error", passed, msg)
        (td / "bin.dat").write_bytes(b"\x00\x01\x02\x03" * 50)
        passed, msg = raises(lambda: convert(td / "bin.dat", None), "binary")
        check("unrecognised binary -> clear error", passed, msg)

        # ---- the identity verdicts ------------------------------------------------------
        e, n1, n2 = PAPER_TABLE1_FB["edges"], PAPER_TABLE1_FB["nodes"], RECOLLECTED_YALE_NODES
        check("verdict: Table 1 exactly -> match", identity_check(n1, n1, e)["status"] == "match")
        check("verdict: 8,578 nodes and 405,450 edges -> match_recollection",
              identity_check(n2, n2, e)["status"] == "match_recollection")
        check("verdict: right edges, other node count -> edges_only",
              identity_check(9000, 9000, e)["status"] == "edges_only")
        check("verdict: 0.5 % fewer edges -> near", identity_check(n2, n2, e - 2000)["status"] == "near")
        check("verdict: a 5-edge graph -> mismatch", identity_check(6, 5, 5)["status"] == "mismatch")

        # ---- full-size synthetic file at exactly 8,578 nodes and 405,450 edges ---------
        rng = np.random.default_rng(0)
        N, E = n2, e
        seen = {(min(i, (i + 1) % N), max(i, (i + 1) % N)) for i in range(N)}   # ring: no isolates
        while len(seen) < E:
            a = rng.integers(0, N, 8192).tolist()
            b = rng.integers(0, N, 8192).tolist()
            for x, y in zip(a, b):
                if x != y and len(seen) < E:
                    seen.add((min(x, y), max(x, y)))
        rows = np.array([p[0] for p in seen] + [p[1] for p in seen])
        cols = np.array([p[1] for p in seen] + [p[0] for p in seen])
        big = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
        sio.savemat(td / "yale_like.mat", {"A": big, "local_info": np.zeros((N, 7))})
        res_big = convert(td / "yale_like.mat", td / "yale_like.edgelist")
        check("full-size synthetic: 8,578 nodes / 405,450 edges reproduced and judged accordingly",
              (res_big["reduced"]["nodes"], res_big["reduced"]["edges"],
               res_big["identity_check"]["status"]) == (N, E, "match_recollection"),
              res_big["identity_check"]["verdict"][:70])
        G = load_parent(td / "yale_like.edgelist", verbose=False)
        check("full-size synthetic: load_parent agrees",
              (G.number_of_nodes(), G.number_of_edges()) == (N, E))
        check("every report renders", all(k in render(x) for x in (res, res_big)
                                          for k in ("FB upload check", "identity check")))

        # ---- the default output path is the one sweep.PARENTS["fb"] reads ----------------
        try:
            sys.path.insert(0, str(REPO_ROOT / "src" / "adv_gd"))
            from sweep import PARENTS
            check("DEFAULT_OUT equals sweep.PARENTS['fb']", PARENTS.get("fb") == DEFAULT_OUT,
                  str(PARENTS.get("fb")))
        except ImportError as exc:
            print(f"  [SKIP] the sweep.PARENTS comparison needs the project environment ({exc})")

    print("e4_convert_fb self-test " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert an uploaded FB graph (fb100 .mat or a plain edge list) into "
                    "the generator's edge list and print the E4a identity check.")
    ap.add_argument("input", nargs="?", help="the uploaded file (.mat, edge list, or .gz)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"edge list to write (default {DEFAULT_OUT.relative_to(REPO_ROOT)}, "
                         "the file sweep.PARENTS['fb'] points at)")
    ap.add_argument("--identity-json", type=Path, default=None,
                    help="where to save the identity check (default: <out>.identity.json "
                         "next to the edge list; none with --check-only)")
    ap.add_argument("--format", choices=("auto", "mat", "edgelist"), default="auto")
    ap.add_argument("--var", default=None,
                    help=".mat only: the adjacency matrix variable (default `A`)")
    ap.add_argument("--check-only", action="store_true",
                    help="print the identity check and write no edge list")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing, different --out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.input:
        ap.error("the input file is required (or pass --self-test)")

    out = None if args.check_only else args.out
    try:
        res = convert(Path(args.input), out, fmt=args.format, var=args.var, force=args.force)
    except ConvertError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(render(res))
    idj = args.identity_json
    if idj is None and out is not None:
        idj = out.with_suffix(".identity.json")
    if idj is not None:
        idj.parent.mkdir(parents=True, exist_ok=True)
        idj.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        print(f"identity check saved to {idj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
