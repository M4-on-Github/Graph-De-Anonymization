"""Seed-free matching baselines for the Adv-GD suite, none of them learned.

Deliberately kept out of `src/adv_gd/`: everything in that package follows the
paper's own lineage (GAE embeddings, adversarial alignment, CSLS, Procrustes).
The four methods here are the classic `structural_fingerprint` and `isorank`,
`centrality_rank_match` (Report 1's rank-matching, reproduced) and
`composite_centrality` (this repo's own, which borrows CSLS from the Adv-GD
lineage). None of them uses embeddings or adversarial training, so a reader
looking for "what Adv-GD actually does" should never have to step past them,
and a reader looking for "what else was tried" should not have to go digging
inside the Adv-GD package for it.

What they share with `src/adv_gd/` is only the data and the scoring: all of
them read pair directories from the same root (`runs/sweep_nominal/` by
default, `runs/sweep_divergence/` under `--beta-mode divergence`, written by
`src/adv_gd/sweep.py --generate`) and all hand their anchors and
candidate sets to the same Phase-3 propagator
(`src.adv_gd.propagation.propagate`), so a number from any of these files is
comparable to Adv-GD's own row in the same table, not a differently-shaped
result that merely looks similar. `common.py` is where that sharing happens;
nothing baseline-specific belongs in it.
"""
