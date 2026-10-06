"""Evaluation-pair generation for seed-free graph de-anonymization.

The generator cuts two independently perturbed copies out of one real parent graph, so
the correct node mapping is known by construction. Protocol and its limits:
`perturbation_protocol.md` in this directory.
"""

from .perturbation import (  # noqa: F401
    check_overlap,
    load_parent,
    make_pair,
    perturb,
    relabel_random,
    resolve_edgelist_path,
    self_test,
    write_pair,
)

__all__ = [
    "check_overlap",
    "load_parent",
    "make_pair",
    "perturb",
    "relabel_random",
    "resolve_edgelist_path",
    "self_test",
    "write_pair",
]
