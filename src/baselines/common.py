"""Shared loader and scoring glue for the seed-free baselines.

All four method modules (`structural_fingerprint.py`, `isorank.py`,
`centrality_rank_match.py`, `composite_centrality.py`) read the same pair
directories Adv-GD's sweep generates and are scored through the same Phase 3
propagator Adv-GD uses, so a baseline's number sits in the same table as
Adv-GD's own row rather than needing a footnote about how it was computed
differently. Nothing baseline-specific belongs here; put that in the
baseline's own file.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ADV_GD_DIR = REPO_ROOT / "src" / "adv_gd"
sys.path.insert(0, str(ADV_GD_DIR))

from adv_gd import DATASETS, load_pair               # noqa: E402,F401
from alignment import (mutual_anchors, candidate_sets, TOP10,              # noqa: E402,F401
                       evaluate_anchors, csls_matrix, DEFAULTS)
from propagation import propagate, evaluate           # noqa: E402,F401
from provenance import RunRecord, capture_stderr      # noqa: E402,F401

__all__ = ["DATASETS", "load_pair", "mutual_anchors", "candidate_sets",
           "TOP10", "evaluate_anchors", "csls_matrix", "DEFAULTS",
           "propagate", "evaluate", "RunRecord", "capture_stderr",
           "top_degree", "REPO_ROOT"]


def top_degree(G, nodes: list, k: int):
    """Row indices of the k highest-degree nodes, ties broken by id.

    Identical in spirit to `adv_gd.alignment.top_degree` (and deliberately
    duplicated rather than imported: that one is private to the alignment
    module's row-index convention, this one is the public cut point every
    baseline here uses to restrict itself to a tractable node count, same as
    Adv-GD restricts Phase 2 to the top N_adv/N_anch/M degree nodes).
    """
    import numpy as np
    deg = G.degree()
    order = sorted(range(len(nodes)), key=lambda i: (-deg[nodes[i]], nodes[i]))
    return np.array(order[:min(k, len(nodes))], dtype=np.int64)
