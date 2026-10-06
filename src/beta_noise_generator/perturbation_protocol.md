# Graph-pair generator: the Li et al. perturbation protocol

Design note and provenance for `perturbation.py` in this directory.
Source: `docs/SeedFree_Li.etal.pdf` — Li, Lu, Luo, Cai, *Seed Free Graph
De-anonymization with Adversarial Learning*, CIKM '20.

## What it does

Produces an evaluation pair (G^a, G^u) plus exact ground truth from **one** real
graph. You never collect two graphs; you cut two damaged copies from the same
parent, so the correct node mapping is known by construction.

```
load parent G(V, E)              # undirected, simple
G^a = delete each edge w.p. β    # independent draw
G^u = delete each edge w.p. β    # second, independent draw
drop degree-0 nodes from each    # nodes are never deleted directly
ground truth = identity on V^a ∩ V^u
score χ = correct / |V^a ∩ V^u|
```

## Using it

```bash
# Protocol checks on a synthetic graph -- no dataset needed, run this first
python src/beta_noise_generator/perturbation.py --self-test

# One parent -> five betas x five repeats, written under data/generated/
python src/beta_noise_generator/perturbation.py --parent data/raw/cit-HepTh.txt --repeats 5

# Google+ is 30M lines; sample nodes rather than loading all of it
python src/beta_noise_generator/perturbation.py --parent gplus_combined.txt     --node-sample 0.12 --beta 0.1 0.3
```

Each run writes `G1.edgelist`, `G2.edgelist`, `mapping.txt` and `stats.json`
into `data/generated/<parent>_b<beta>_s<seed>/`.

**Caveat for the inherited baseline.** `src/seed_based.py` and `src/seed_free.py`
gate loading on the literal substring `labeled_dev` appearing in the *directory*
path, and load nothing — silently — if it does not. Pass `--compat-name` to
suffix generated directories so those scripts accept them.

## Where each piece comes from

| Element | Paper location |
|---|---|
| Parent graphs with known ground truth (FB, HepTh, Gplus) + source URLs | §4.1, Table 1, p.749 |
| Remove β of edges independently, twice, to get G^a and G^u | §4.1 p.749; restated §4.3 p.750 |
| β swept 10%–50% | §4.1 p.749, §4.3 p.750 |
| Expected edge overlap = (1−β)² | §4.1, p.749 |
| Nodes not deleted directly; degree-0 nodes removed, so \|V^a ∩ V^u\| ≠ \|V\| | §4.3, p.750 |
| Deletion-only chosen over add+delete, to avoid picking a link predictor | §4.3, p.750 |
| χ = n / \|V^a ∩ V^u\| | p.751 |
| Repeat 5×, report the average | §4, p.749 |
| G^a, G^u assumed undirected | §2.1, p.746 |

## Parameters

β is the only one, and the paper supplies it. Everything else is a declared
choice, not a gap — see "Left open" below.

## Built-in self-checks

The protocol can be validated, not merely written. An edge survives both draws
with probability (1−β)², which gives two cheap assertions:

| Quantity | Converges to |
|---|---|
| \|E^a ∩ E^u\| / \|E\| | (1−β)² |
| Jaccard(E^a, E^u) | (1−β)²/(1−β²) = (1−β)/(1+β) |
| \|E^a\| / \|E\| | 1−β |

`check_overlap()` asserts the first, at four binomial standard deviations floored
at 0.01, and `make_pair()` records all three in `stats.json`. This is how
`data/labeled_dev` was identified as HepTh under this protocol: measured 0.9682
against 0.9685 predicted from its *effective* β, not its nominal one.

A third, weaker check: **parent |E| against Table 1.** Gplus is 107,614 nodes /
13,673,453 edges, and `gplus_combined.txt` has exactly 107,614 nodes, so our
collapse of it can be checked against their edge count. Treat as approximate —
HepTh's listed 352,807 looks like SNAP's raw directed count, not a collapsed
undirected one.

## Pros

- **Ground truth is free and exact.** No labeling, no second data source, no
  partial truth. This is the whole reason the protocol is standard.
- **One knob.** Difficulty is dialled by β alone, so results are a clean curve.
- **Reproducible and comparable.** Same protocol as the prior work Li cites, so
  numbers are at least nominally comparable across papers.
- **Self-validating** via (1−β)², which catches most implementation bugs.
- **Cheap.** O(|E|) per graph; no model, no training, no tuning.

## Cons

These bound what a result on this protocol can claim.

- **Deletion only — no false edges.** Real auxiliary graphs contain edges the
  published graph lacks. Li acknowledges add+delete is the general case (§4.3)
  and drops it only to avoid committing to a link-prediction model. A matcher
  tuned here never faces a spurious edge.
- **Symmetric noise.** G^a and G^u are statistically exchangeable draws from one
  parent. In reality the anonymized release and the crawled auxiliary differ in
  kind — different collection, different coverage, different bias — not just by
  an independent coin flip.
- **Degree sequence is perturbed, but recoverably.** Deletion thins degrees
  roughly proportionally, so degree survives as a signal. Degree-preserving
  schemes (edge switching, as in SecGraph/CCS14) would defeat degree-based
  candidate generation outright. Robustness shown here does not transfer.
- **Node overlap is not controlled, it is a side effect.** |V^a ∩ V^u| falls out
  of isolated-node removal, so at low β nearly every node survives in both.
  Narayanan–Shimatikov instead parameterize node overlap directly. Consequence:
  edges get hard as β rises while the node set stays nearly complete — a
  mismatch that flatters recall.
- **Identity ground truth leaks silently.** Since φ is the identity, any bug
  that compares raw node IDs scores 100% and looks like success. Permuting IDs
  costs nothing and removes the failure mode.
- **Structure only.** No attribute, profile or temporal noise.

## Left open by the paper — and what this module declares

| Choice | Paper | This module |
|---|---|---|
| Bernoulli per edge vs exactly ⌊β\|E\|⌋ removals | "randomly remove β fraction", ambiguous | `--mode bernoulli` (default), `exact` available |
| RNG seeds | unstated | `--seed`; G^a and G^u draw from disjoint substreams, recorded in `stats.json` |
| Permute node IDs | no, ground truth is the identity | yes by default; `--no-permute` restores the paper's behaviour |
| Directed → undirected collapse | unstated; §2.1 just assumes undirected | collapse first, always — see below |
| Repeats | 5, averaged | `--repeats`, default 1 |

## The trap, with evidence

The collapse-ordering choice is not academic — it already damaged
`data/labeled_dev`.

That pair is stored as a **directed** line list. Of its surviving undirected
edges, 84.2% still carry both directions. The line count is ~90% of the
symmetric parent, so roughly 10% of *lines* were removed — but an undirected
edge only disappears when *both* of its lines do, so the effective deletion rate
`nx.read_edgelist` sees is **2.2% for G1 and 1.0% for G2**.

A nominally "10% noise" dataset is really a ~1–2% noise dataset, which is a large
part of why the baseline reaches 92%.

**Rule: collapse to an undirected simple graph first, then perturb.**
Perturbing a symmetric edge list and collapsing afterwards squares your
intended β, because an undirected edge only dies when both of its lines do.
`load_parent()` collapses on read, and `--self-test` pins the arithmetic: a
nominal β = 0.1 applied the wrong way round measures 0.0100 effective, exactly
β². That is the 10%-becomes-1% failure, reproduced on demand.
