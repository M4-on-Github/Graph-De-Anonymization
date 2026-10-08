"""Adv-GD: seed-free graph de-anonymization with adversarial learning.

A reimplementation of Li, Lu, Luo and Cai, CIKM '20, from the paper text --
there is no official code release. Read `adv_gd_protocol.md` in this directory
before using or extending any of it: it records which parts are specified by
the paper, which are our choices, and the two printed equations that cannot be
implemented as written.

Three phases, one module each:
    embedding.py    Phase 1 -- GAE node embeddings
    alignment.py    Phase 2 -- adversarial linear map, CSLS, Procrustes
    propagation.py  Phase 3 -- Algorithms 1 and 2
    adv_gd.py       driver, and the Table 2 comparison

Every module has a `--self-test` that runs without a dataset.
"""

from alignment import align, csls_matrix, evaluate_anchors, procrustes
from embedding import gae_embed, reconstruction_auc
from propagation import eccentricity, evaluate, get_scores, propagate

__all__ = [
    "align",
    "csls_matrix",
    "eccentricity",
    "evaluate",
    "evaluate_anchors",
    "gae_embed",
    "get_scores",
    "procrustes",
    "propagate",
    "reconstruction_auc",
]
