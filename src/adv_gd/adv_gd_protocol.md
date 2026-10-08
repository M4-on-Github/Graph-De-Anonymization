# Adv-GD: reimplementation spec and provenance

Design note for the Adv-GD reimplementation in this directory.
Source: `docs/SeedFree_Li.etal.pdf` — Li, Lu, Luo, Cai, *Seed Free Graph
De-anonymization with Adversarial Learning*, CIKM '20.

**There is no official implementation.** The paper contains six URLs — its own
DOI, the three dataset pages, and two Statista links — and no code release. The
authors' pages carry none either. Everything here is reconstructed from the text,
so this document records what the paper states, what it omits, and what we chose
in its place. Treat an unmarked choice as ours, not theirs.

## What it does

Three phases. Input is a pair (G^a, G^u) with no seeds; output is a node mapping.

```
Phase 1  embedding    Z^a = GAE(G^a),  Z^u = GAE(G^u)        # independently, d dims
Phase 2  alignment    learn linear W : Z^a -> Z^u adversarially
                      score pairs by CSLS
                      anchors = mutual CSLS nearest neighbours
                                among top-N_anch degree nodes of both graphs
                      refine W by Procrustes on the anchors, repeat
                      candidates = top-10 CSLS matches for each
                                   top-M degree node of G^a
Phase 3  propagation  seed the mapping with the anchors, then spread over
                      matched neighbours (Algorithms 1 and 2)
```

Phases 1–2 replace the seed assumption; Phase 3 is a propagation step of the same
family as the inherited `src/seed_based.py`.

## Where each piece comes from

| Element | Paper location |
|---|---|
| Three-phase framework | §3, Fig. 1, p.746 |
| GAE encoder, two-layer GCN, X = I | §2.2, Eq. 1–3, p.746 |
| Adversarial objective for discriminator and W | §3.1, Eq. 4–5, p.747 |
| Train on top-N_adv degree nodes only | §3.1, p.747 |
| CSLS similarity | §3.1, Eq. 6–7, p.747 (method from MUSE [21]) |
| Anchor selection rule | §3.1, p.747 |
| Procrustes refinement by SVD | §3.1, Eq. 8–9, p.747 |
| Candidate sets: top-10 CSLS per top-M degree node | §3.1, p.747 |
| GetScores | Algorithm 1, p.748 |
| Propagating De-anonymization, two stages on `flag` | Algorithm 2, p.748 |
| Eccentricity threshold theta = 0.5 | §4.2, p.750 |
| Per-dataset D, M, N_adv, N_anch | Table 1, p.749 |
| Model selection by mean cosine of CSLS matches | §3.1, p.747 |

## Hyperparameters the paper supplies

Table 1, p.749. `D` is the embedding dimension, not a degree.

| Dataset | \|V\| | \|E\| | D | N_adv | N_anch | M |
|---|---|---|---|---|---|---|
| FB | 8,758 | 405,450 | 16 | 1,000 | 400 | 5,000 |
| HepTh | 27,770 | 352,807 | 32 | 5,000 | 1,000 | 10,000 |
| Gplus | 107,614 | 13,673,453 | 300 | 20,000 | 5,000 | 50,000 |

Also stated: discriminator by SGD, learning rate 0.1, batch 32; W by Adam,
learning rate 0.01, batch 32; theta = 0.5 for all datasets.

Those \|V\| and \|E\| are the **source files' advertised counts**, not counts of the
undirected simple graphs Adv-GD runs on. HepTh's 352,807 is exactly the number of
edge lines in `cit-HepTh.txt`; collapsed it is 352,285 (see
`src/beta_noise_generator/perturbation_protocol.md`). Do not use Table 1 to
validate a loaded graph beyond confirming the right file was downloaded.

Gplus does not use GAE: the paper reports out-of-memory on a 32 GB GPU and
switches to PyTorch-BigGraph (§4.2). So Gplus is not a test of this code path.

## What the paper leaves unstated — and what this module declares

Every gap is in training detail, and every fill below comes from the work Li
cites for that component. Phase 2 is MUSE applied to graph embeddings instead of
word embeddings, so MUSE's defaults are the defensible reading — but they are
still our choice.

| Choice | Paper | This module | Taken from |
|---|---|---|---|
| Embedding normalization before Phase 2 | unstated | **none**, the literal reading; measured equal to unit-norming, while MUSE's centering is harmful | measured here |
| Adversarial restarts | unstated; model selection itself is specified | **4**, selected by mean cosine - see the reproduction log | measured here; a single-seed run is a coin flip |
| K in CSLS | unstated | 10 | MUSE [21] |
| Discriminator architecture | unstated | 2 hidden layers x 2048, LeakyReLU 0.2, dropout 0.1, label smoothing 0.2 | MUSE [21] |
| Adversarial epochs | unstated | 5 | MUSE [21] |
| Refinement rounds | "again", count unstated | 5 | MUSE [21] |
| Orthogonalization of W during training | unstated | yes, beta = 0.01 | MUSE [21] |
| GAE hidden dim | "two-layer", only output D given | 64 -> D | Kipf & Welling [18] |
| GAE epochs | unstated | **600**, not Kipf's 200 — see the reproduction log | measured here |
| GAE optimizer | unstated | Adam lr 0.01 | Kipf & Welling [18] |
| GAE loss negatives | Eq. 3 has none | full BCE, positive class reweighted | see below |
| Random seeds | unstated | recorded per run |  |

## Errors in the paper as printed

Two equations cannot be implemented literally. These are typesetting faults, not
design decisions, and we implement the standard forms.

- **Eq. 1** gives the normalized adjacency as `D^(-1/2) A D`. That is not
  symmetric and is not a graph Laplacian normalization. The intended form is
  `D^(-1/2) A D^(-1/2)`.
- **Eq. 3** gives the GAE loss as `-(1/N^2) sum_ij A_ij log(A_ij_hat)` — the
  positive term only. Minimizing it is degenerate: drive every reconstructed
  entry to 1 and the loss goes to zero, with no pressure to keep non-edges apart.
  The intended form is binary cross-entropy over both classes, with the positive
  class reweighted because A is sparse.

## Validation targets

Reproduction is checked against HepTh first, because it is the only dataset for
which the paper reports **exact integers** rather than a figure.

**Table 2, p.751 — landmark (anchor) identification on HepTh.** This scores
Phases 1–2 alone, with no propagation, which is what makes it valuable: it
isolates the half of the method that is genuinely new.

| beta | Adv-GD correct/found | CCS14 |
|---|---|---|
| 10% | 753/755 | 25/1000 |
| 20% | 714/729 | 15/1000 |
| 30% | 652/670 | 13/1000 |
| 40% | 585/607 | 12/1000 |
| 50% | 472/512 | 8/1000 |

**End to end,** HepTh chi runs "from 97% to 28%" across beta = 10%–50%
(§4.3, p.750, read off Fig. 3(b)) — a range, not a table, so it is a weaker check.

chi = (correctly de-anonymized nodes) / \|V^a ∩ V^u\|. On our generated HepTh
pairs the denominators are:

| beta | 10% | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|
| \|V^a ∩ V^u\| | 27,444 | 27,108 | 26,657 | 26,138 | 25,423 |

Nothing in the inherited baseline computes chi, so its reported accuracies are
not comparable to any number above.

## What a reproduction here can and cannot claim

- **Can:** that the method as described, with MUSE/GAE defaults for the gaps,
  reaches or misses the published anchor counts on the same parent graph under
  the same perturbation protocol.
- **Cannot:** that any gap was filled the way the authors filled it. A miss is
  evidence about our reconstruction at least as much as about their result.
- **Cannot:** anything about Gplus as a test of this code, since the paper used a
  different embedding method there.
- **Cannot:** robustness to anything but edge deletion. The protocol adds no
  false edges — see the cons in `perturbation_protocol.md`.

## Input

Pairs come from `src/beta_noise_generator/perturbation.py`, written to `runs/`:

```bash
python src/beta_noise_generator/perturbation.py \
    --parent data/cit-HepTH/cit-HepTh.txt --beta 0.1 0.2 0.3 0.4 0.5
```

Ground truth is the identity on V^a ∩ V^u unless `--no-permute` is omitted, in
which case `mapping.txt` carries the permutation. Keep the permutation on: with
identity ground truth any bug that compares raw node IDs scores 100%.

## Sweeping beta across seeds

`src/adv_gd/sweep.py` is the multi-cell driver. One invocation of `adv_gd.py`
runs one pair at one seed; the sweep is the cross product of noise level and
alignment seed, and it exists because every number in the log below rests on a
single noise draw at a single beta.

```bash
# What would run, and in what order. Writes nothing.
python src/adv_gd/sweep.py --plan

# Make the pairs (idempotent: existing pairs are left alone).
python src/adv_gd/sweep.py --generate

# One cell. SLURM array index, or a bash loop.
python src/adv_gd/sweep.py --task-id $SLURM_ARRAY_TASK_ID
```

Four decisions are built into it, each because of something measured.

**The alignment seed and the GAE seed are separate.** Phase 1 is deterministic
given its own seed and is cached as `Z1.npz`/`Z2.npz`, keyed on epochs, dim and
that seed. Phase 2's bimodality is a property of the adversarial
initialization, not of the embedding, so a seed sweep that moved both seeds
together would re-train two GAEs per cell -- roughly twenty minutes of GPU time
each -- to vary something the question is not about. The sweep pins `--gae-seed
0` and varies only the alignment seed, so Phase 1 is paid for once per pair.
This is also why `--gae-seed` exists on `adv_gd.py` at all; it defaults to
`seed`, which is how every result recorded before 2026-10-08 was produced.

**A cell is one process.** `--task-id N` runs exactly cell N of the plan and
exits. There is no in-process loop over cells, because a single long process
that dies at cell 14 of 20 loses the uncompleted cells and tells you nothing
about which; twenty processes that each write a record lose only their own. The
plan order is fixed and printable, so `--task-id` is stable across invocations
as long as the flags are.

**`--beta-mode` chooses what beta means.** The paper does not say whether its
beta is the per-copy deletion rate or the divergence between the two copies,
and the difference is large: deleting independently at beta from each copy
leaves an edge in both with probability (1-beta)^2, so a nominal beta=0.1 makes
the copies differ by about 19%. Generating instead at beta' = 1 - sqrt(1-beta)
makes them differ by beta. Measured on HepTh, that single change moves chi from
0.913 to 0.943 against the paper's ~0.97 (see the noise-model section below),
which is most of our shortfall. `nominal` reproduces our own earlier numbers;
`divergence` is the comparison to make against Table 2. **Only `divergence`
cells are comparable to the paper, and the two modes are not comparable to each
other.**

**The two modes generate into separate roots.** `runs/sweep_nominal/` and
`runs/sweep_divergence/`, because a pair directory is named
`<stem>_b<NN>_s<seed>` with beta rounded to whole percent, and that name
encodes neither the mode nor the sampling fraction. Nominal beta=0.05 and
divergence beta'=0.0513 both name `cit-HepTh_b05_s0`; sharing a root would let
one silently replace the other. This is the pair-naming defect recorded in
`docs/experiment_logging_policy.md`, worked around rather than fixed.

Operational notes for the cluster.

- **Set `PYTHONHASHSEED`.** The sweep refuses to run a cell without it, and
  prints the value into every record. Phase 3 was measured to be hash-order
  independent here, unlike the inherited baseline, but that measurement is only
  meaningful next to the value it was taken under.
- **Phase 1 needs a GPU.** 600 GAE epochs on HepTh's 27k nodes is the bulk of a
  cell's wall clock, and on CPU it is far worse than the ~1.8 h a full
  four-restart run took on an RTX 5070. Generate the pairs and warm the
  embedding caches on a GPU node even if Phase 2 and 3 run elsewhere.
- **Cost.** One end-to-end cell with four restarts took 6,511 s. Five betas by
  four alignment seeds is 20 cells, of which the GAE training is shared within
  each beta.
- **Collate afterwards.** Each cell writes its own record and appends one line
  to `results/runs/index.jsonl`; concurrent appends to one file are not
  guaranteed intact on a network filesystem, so run
  `python src/adv_gd/provenance.py --collate` when the array finishes. It
  rebuilds the ledger from the record directories, which are authoritative.

What the sweep still will not give us: a confidence interval over noise draws.
`--pair-seeds` takes more than one value and the plan multiplies out, but at one
pair seed -- the default, and what 20 cells of budget buys -- the beta axis is
still N=1 per cell and only the alignment axis is replicated.

## Reproduction log

### GAE training length is the dominant unstated hyperparameter

The paper gives no epoch count for Phase 1. Starting from Kipf and Welling's
default of 200 produced anchors far below Table 2, and the shortfall was traced
to Phase 1 rather than Phase 2 by measuring a **supervised ceiling**: hand
Procrustes the complete ground-truth correspondence, which is the best any
linear map could do on a given pair of embedding spaces, and see how far the
anchors get. Anything the unsupervised pipeline achieves is bounded by it.

On HepTh at beta = 0.1, one pair, seed 0:

| GAE epochs | Reconstruction AUC | Ceiling anchors | Ceiling precision | Supervised rank-1 over all 27,444 matchable nodes |
|---|---|---|---|---|
| 200 | 0.988 | 394/478 | 0.824 | 0.103 |
| 600 | 0.996 | 823/848 | 0.971 | 0.420 |

Two things follow.

- **Reconstruction AUC is a poor guide to alignability.** It moved 0.988 ->
  0.996, under one point, while the ceiling precision moved 0.82 -> 0.97 and
  rank-1 accuracy quadrupled. A GAE can reconstruct its own adjacency matrix
  well while still sitting in a gauge no linear map can reconcile with a second
  graph's. Do not tune Phase 1 on AUC.
- **The default was wrong, not the method.** At 200 epochs the two graphs had
  not even converged to the same loss (0.4059 vs a still-falling G2); at 600
  they agree to four decimals (0.3847 / 0.3849). Independently trained
  embeddings have to be comparably converged before a single W can align them.

This matters for anyone reading Table 2 as a reproducibility target: one
parameter that moved the ceiling a long way is one the paper never states.

Two cautions about this table. Both rows are a single pair at a single seed, so
the ceiling figures have no error bar -- though the supervised Procrustes that
produces them involves no adversarial training and is deterministic, so they at
least do not inherit the seed problem described below.

More importantly, the table says nothing about the end-to-end result, and an
earlier version of this log drew a conclusion from it that the evidence does
not support. The unsupervised run at seed 0 read 342/472 at 200 epochs and
7/282 at 600, which looked like longer training breaking the alignment. It was
not: at 600 epochs seeds 2 and 3 both reach 0.971 precision and only seeds 0
and 1 fail. The 200-vs-600 end-to-end comparison was one run on each side of a
bimodal distribution, and it cannot separate training length from the seed
draw. What survives is the ceiling improvement; the claim that 600 epochs hurt
the pipeline does not.

### Why the ceiling is the right diagnostic

The unsupervised result and the supervised ceiling separate the two failure
modes that otherwise look identical:

- unsupervised far below ceiling  -> Phase 2 is at fault (adversarial training,
  CSLS, refinement)
- unsupervised near a low ceiling -> Phase 1 is at fault (the embeddings cannot
  be aligned by any linear map, however good the search)

At 200 epochs our unsupervised run reached 342/472 against a 394/478 ceiling,
87% of what was attainable, which is what ruled Phase 2 out as the cause *at
that seed*. The qualifier matters: with hindsight from the seed study, a single
unsupervised run sitting far below the ceiling is just as likely to be a
failing seed as a broken Phase 2, so the comparison is only diagnostic when the
unsupervised side is itself the best of several restarts. Used that way the
logic holds, and it is why `align` now reports the mean cosine of every
restart.

### Adversarial training is bimodal across seeds, and model selection is what makes it work

This is the largest single effect we have measured, and it governs how every
other number in this log has to be read.

Phase 2 at a fixed seed is bit-reproducible: two runs at seed 0 agree on sum|W|
to all printed digits. Across seeds it is not merely noisy, it is bimodal.
HepTh, beta = 0.1, 600-epoch embeddings, DEFAULTS otherwise:

| Seed | Anchors correct/found | Precision | Mean match cosine |
|---|---|---|---|
| 0 | 7/282 | 0.025 | 0.845 |
| 1 | 0/257 | 0.000 | 0.824 |
| 2 | 832/857 | 0.971 | 0.958 |
| 3 | 829/854 | 0.971 | 0.959 |

Roughly half of all seeds find the right alignment and the rest find nothing;
there is no middle. Reporting a mean over seeds is therefore meaningless -- the
spread across four seeds is 0 to 832 correct -- and **a single-seed result is a
coin flip, not a measurement**. Any number quoted from this pipeline has to say
how many restarts produced it.

What rescues the method is the one line of section 3.1 that is easy to read as
a formality: *use the average cosine similarity of the matching node pairs
derived by CSLS to select models.* It separates the two modes perfectly, with
no ground truth and no validation set, which is the only thing available to a
genuinely seed-free method. Across twelve runs (three preprocessing modes by
four seeds) it selected the best or statistically tied-best seed in every mode:

| Preprocessing | Seeds reaching a good alignment | Best precision | Precision of the mean-cosine pick |
|---|---|---|---|
| `none` -- the paper as written | 2/4 | 0.971 | 0.971 |
| `renorm` -- unit-norm rows | 2/4 | 0.976 | 0.976 |
| `center_renorm` -- MUSE's default | **0/4** | 0.190 | 0.190 |

`n_restarts` is therefore 4 by default and the criterion is applied inside
`align`, rather than left to the caller. With it the pipeline is reliable: the
confirming run rejected seeds 0 and 1 on mean cosine alone (0.845, 0.824),
selected seed 3 (0.959), and finished at chi = 0.913. Every run records the
mean cosine of all four restarts and which seed was selected, in the
`restarts` and `selected_seed` fields of `adv_gd_result.json`, so a result can
always be checked against the spread it was drawn from.

The pattern replicates on an independently generated pair. On
`cit-HepTh_b05_s0` -- a different parent perturbation, 90% edge overlap instead
of 81% -- the four restarts read 0.8296, 0.9557, 0.9554, 0.8405: again two
seeds in the high mode and two in the low, and again cleanly separated, with
seed 1 selected at 859 anchors. Two pairs is not a distribution, but the
split is not an artefact of one noise draw.

One consequence for anyone extending Phase 2: **do not compare settings at one
seed each.** Four restarts per cell is the minimum that distinguishes a real
effect from a seed draw, and that is what the twelve-run grid above costs.

Two readings of that table are worth separating.

- **Row-normalizing the embeddings does not matter**, and an earlier version of
  this log claimed it did. That claim came from three runs at seed 0 only,
  where `none` happened to draw a failing seed and `renorm` a succeeding one.
  Measured across four seeds the two are indistinguishable, so the default is
  the paper's literal reading. This is a good illustration of the hazard the
  bimodality creates: a one-run-per-cell comparison of any Phase 2 setting will
  mostly measure which seeds it drew.
- **Centering is genuinely harmful**, and this survives the correction: 0 of 4
  seeds, best of four 0.190, and tripling the adversarial budget drove it to
  0/246 rather than fixing it. The reason is structural. The GAE decoder
  sigma(Z Z^T) is invariant under an orthogonal transform of Z but not under a
  translation of it, and that invariance is precisely why one linear W can
  align two independently trained spaces. Subtracting a different per-graph
  mean breaks the symmetry the method depends on. MUSE centers word vectors,
  which carry no such constraint; carrying that default across was our error,
  not MUSE's.

### What is and is not controlled

| Source of randomness | Controlled by | Status |
|---|---|---|
| GAE weight init | `--seed` -> `torch.manual_seed` | bit-reproducible at a fixed seed |
| Adversarial init, batch sampling | `--seed` -> `torch.manual_seed` plus a CPU `Generator` | bit-reproducible at a fixed seed; **bimodal across seeds** |
| Restart selection | `n_restarts`, mean cosine | deterministic given the seed range |
| AUC sampling | `seed` argument | diagnostic only, does not affect results |
| Degree ties in `top_degree` | broken by node id | deterministic |
| Phase 3 iteration order | nothing needed | deterministic; see below |
| Pair generation (which edges are deleted) | the generator's own `--seed`, recorded in `stats.json` | one pair per beta so far |

Phase 3 needs checking rather than assuming, because the inherited baseline is
hash-seed sensitive: it iterates Python `set`s of string node ids, so its
accuracy moves about 0.1pp between runs. This reimplementation does not inherit
that. Sets are only membership-tested, never iterated; everything iterated is a
`dict` or a NetworkX adjacency view, both insertion-ordered. Confirmed
empirically -- the full 25,782-pair mapping has the same SHA-256 under
PYTHONHASHSEED 0, 1 and 12345, with chi identical to six decimals.

The remaining uncontrolled dimension is the **pair**: one generated pair per
beta, so none of these figures carry a confidence interval over the noise draw.

### Every run in this log has a record

From 2026-10-07, every Adv-GD run writes an immutable record under
`results/runs/<run_id>/` -- git HEAD plus the uncommitted diff, the
environment down to the torch build and GPU, digests of all four input files,
the fully resolved configuration, the restart spread and the selection rule, the
output mapping's digest, and whether it exited ok, errored or was killed. One
line per run is appended to `results/runs/index.jsonl`. The policy, including
what a record does **not** entitle a reader to claim, is
`docs/experiment_logging_policy.md`.

Figures in this log that predate the policy are marked where they appear, and
only two of them have records at all, both backfilled with a null environment.
The rest -- the seed table, the preprocessing grid, the eccentricity and epoch
studies -- were overwritten as they were produced and now exist only as the
prose here. Treat them as reported, not verifiable, and re-run any one of them
whose exact value a conclusion turns on.

### Where the reproduction stands

HepTh, beta = 0.1, one pair, all three phases, 4 restarts selected by mean
cosine (seed 3 chosen):

| | Ours | Paper |
|---|---|---|
| Anchors correct/found (Table 2) | 830/854 | 753/755 |
| Anchor precision | 0.972 | 0.997 |
| chi -- correct over 27,444 matchable | **0.913** | ~0.97 |
| gamma -- correct over 25,727 mapped | 0.974 | not reported for this cell |

The two-stage structure of Algorithm 2 behaves as described: the candidate-set
restriction grows the mapping conservatively to 8,301 over five sweeps, then
lifting it adds 17,452 more over eight. Precision barely moves between the
anchors (0.972) and the final mapping (0.974), which is the property the
paper's design is after -- propagation does not degrade what alignment handed
it.

This is one cell of Table 2 on our own generated pair, not the paper's pair, so
it is evidence that the method is reproducible as described plus the gaps
above, not a replication of the paper's number. We land about 5 points of chi
short, and the section after next locates that shortfall outside Phases 1-2.
The remaining beta values are still to run.

### The residual gap to the paper is not in anchor quality

With 854 *perfect* anchors handed to Phase 3 -- the true images of the 854
highest-degree nodes, so Phase 2 plays no part -- propagation reaches:

| sigma support | theta | Mapped | Correct | chi | gamma |
|---|---|---|---|---|---|
| all (literal) | 0.5 | 25,782 | 25,125 | 0.9155 | 0.9745 |
| all (literal) | 1.0 | 25,777 | 25,145 | 0.9162 | 0.9755 |
| nonzero | 0.5 | 24,321 | 23,992 | 0.8742 | 0.9865 |

Our full pipeline, with 829/854 anchors, reaches chi = 0.913. Perfect anchors
buy 0.002. So propagation is almost indifferent to the 25 wrong anchors, and
**the roughly 5 point gap to the paper's ~0.97 is not an anchor-quality
problem** -- it sits in Phase 3 or in the noise model, not in Phases 1-2. The
eccentricity ambiguity does not explain it either: the literal reading is
already the better one, and theta barely matters.

The most likely remaining explanation is the perturbation protocol rather than
the algorithm. We delete edges independently from each copy at beta, so an edge
survives in both with probability (1-beta)^2 and about 19% of edges differ
between the two graphs at beta = 0.1. If the paper's beta instead describes the
total divergence between the two copies, their task at a nominal beta = 0.1 is
materially easier than ours. This is the interpretation question already
flagged in `src/beta_noise_generator/perturbation_protocol.md`.

### Most of the residual gap is the noise model, not the algorithm

Tested directly. Generating a pair at beta' = 1 - sqrt(0.9) = 0.0513, which
makes the two copies diverge by the same 10% that the paper's beta = 0.1 would
describe under the total-divergence reading, and running the identical pipeline:

| Pair | Edge overlap | Anchors | Anchor precision | chi | gamma |
|---|---|---|---|---|---|
| `cit-HepTh_b10_s0`, beta = 0.1 independent | 0.81 | 829/854 | 0.971 | 0.913 | 0.974 |
| `cit-HepTh_b05_s0`, beta' = 0.0513 | 0.90 | 846/859 | 0.985 | **0.943** | 0.990 |
| Paper, HepTh, beta = 0.1 | unstated | 753/755 | 0.997 | ~0.97 | not reported |

Run ids `backfill_cit-HepTh_b10_s0_3edcf4b5` and
`backfill_cit-HepTh_b05_s0_db3e0a51` in `results/runs/index.jsonl`; both are
salvaged pre-policy results, so their environment is not recorded.

Three points, in decreasing confidence.

- **The interpretation accounts for about half the shortfall.** chi moves from
  0.913 to 0.943 against a target of ~0.97, on the same code, the same
  hyperparameters and the same parent graph. Nothing but the noise model
  changed, so the comparison is clean in the one respect that matters.
- **It does not account for all of it.** Roughly 3 points remain. Anchor
  precision also rises (0.971 to 0.985) but stays below the paper's 0.997, and
  the anchor counts differ in kind rather than degree: we select 859 anchors and
  get 846 right, they select 755 and get 753. Finding more, less pure anchors is
  a different operating point, not a worse version of the same one, and the
  `n_anch` / `m` interaction is a candidate for where that comes from.
- **This is one pair against one pair.** Each row is a single noise draw, so a
  3-point residual is not separable from draw-to-draw variation, which we have
  never measured. The honest reading is "most of the gap is the noise model, the
  remainder is unresolved", not "the remainder is real".

The bimodality applies here too: the beta' run selected seed 1 at mean cosine
0.9557 from restarts of 0.8296 / 0.9557 / 0.9554 / 0.8405.

One process note, because it bears on how the figures in this log should be
trusted. That run was reported mid-session as having been killed partway through
Phase 3 for low system memory. It was not: the kill notice came from the tooling
around the run, the process itself survived and finished 6,511 seconds later,
and the result sat complete on disk for some time while being described as lost.
It was recovered only when the logging policy's backfill step read every
`adv_gd_result.json` in `runs/`. Both halves of that are the argument for
`docs/experiment_logging_policy.md`: a progress log is not evidence of what a
run did, and a result that is not archived at the moment it appears is
indistinguishable from one that never happened.



### A phi/psi inversion that the self-test could not see

Phase 3's first run mapped nothing at all beyond the anchors. The cause was
ours, not the paper's: the reverse-direction mapping psi was rebuilt by
unpacking `psi.items()` into the wrong variable order, which turned it into a
second copy of phi. Every reverse `get_scores` then looked up G^u neighbours in
a dict keyed by G^a ids, found nothing, and the symmetry check rejected all
26,748 unmapped nodes.

Worth recording because of *why* it survived: the self-test propagated a graph
against itself, where the G^a and G^u id spaces coincide and a mapping keyed by
the wrong one still resolves. The test now relabels the two copies with
disjoint id prefixes, which fails on the old code (50/800, anchors only) and
passes on the fixed code (800/800). Any test for a two-sided matching algorithm
needs the two sides to be distinguishable, or it cannot see this whole class of
bug.