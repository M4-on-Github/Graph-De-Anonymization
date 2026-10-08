# Moe's cluster scripts — Report 1 on cit-HepTh

Reimplementing `docs/Project_Report_1_Seed_Based_Seed_Free_.pdf` on generated
HepTh noise pairs, as a SLURM job array on `pleiades`.

The report measures three things on **one** graph pair with no noise parameter:

| Report 1 | Its number | Where it goes here |
|---|---|---|
| Seed-based propagation (Algorithm 1), 20% seeds, θ=0.5 | 92.13% | `report1_seed_based` records |
| Accuracy vs θ (Fig 1a) and vs training % (Fig 1b) | — | the two slices of the seed-based grid |
| Six seed-free rankings scored alone (Table 2) | 0.01–0.83% | `report1_seed_free_rank` records |
| Top-k of a ranking fed back as seeds (Fig 4) | 88.88% at top-50 PageRank | `report1_seed_free_integrated` records |

What this adds is the axis the report does not have: **noise**. Every cell runs
on a pair cut from the HepTh parent at a known β, with exact ground truth, so
"accuracy" becomes a surface over (β, seeds) instead of a single number.

## Prerequisites

```bash
# Once. Needs 64G -- a 4G interactive session is OOM-killed mid-build.
sbatch --partition=pleiades --cpus-per-task=8 --mem=64G --time=1:30:00 \
       --job-name=build_gda --output=/data/$USER/Graph-De-Anonymization/logs/build_gda_%j.out \
       scripts/moes-cluster-scripts/build_container.sh
```

There is no python newer than 3.6 on `pleiades` outside a container, and
`networkx 3.7` needs 3.11+. The parent graph is downloaded by step 0 if absent;
only `cit-HepTh.txt.gz` is fetched, never the ~29,000-file abstracts archive.

## Running it

```bash
cd ~/Graph-De-Anonymization                       # submit from the repo root
bash scripts/moes-cluster-scripts/submit_report1.sh --dry-run   # see the plan
bash scripts/moes-cluster-scripts/submit_report1.sh             # submit it
```

Three chained stages: generate the pairs (one job), then the seed-based array,
then the seed-free array, the arrays gated on `afterok` of the generation so
they cannot start on pairs that do not exist. Defaults: 18 pairs, 144
seed-based cells, 30 seed-free cells writing 150 records, at most 24 tasks at a
time.

Narrow it with any `grid.py` flag, which `submit_report1.sh` passes straight
through:

```bash
# one pair, the report's anchor cell only -- the first thing to run
bash .../submit_report1.sh --betas 0.0 --pair-seeds 0 \
     --train-portions 0.20 --thetas 0.5 --only seed_based

# the theta sweep at one noise level
bash .../submit_report1.sh --betas 0.2 --pair-seeds 0 --train-portions 0.20
```

## Reading the results

```bash
python3 scripts/moes-cluster-scripts/collect.py --kind report1_seed_based \
        --summary beta_nominal theta
python3 scripts/moes-cluster-scripts/collect.py \
        --out /data/$USER/Graph-De-Anonymization/report1.csv
```

`collect.py` reads `results/runs/*/record.json`, not `index.jsonl`: the ledger
line is Adv-GD shaped (`chi`, `gamma`, `anchor_precision`) and carries no
accuracy. Records are tracked in git; the bytes they describe are not. See
`docs/experiment_logging_policy.md`.

## Where things land

Nothing is written outside two directories, both named after the repo:

| Path | What | In git? |
|---|---|---|
| `~/Graph-De-Anonymization/results/runs/<run_id>/` | run records, one per cell | **yes** |
| `~/Graph-De-Anonymization/runs/report1_<mode>/<pair>/` | pair edge lists, ground truth, output mappings | no |
| `/data/$USER/Graph-De-Anonymization/graph_deanon.sif` | the container image | no |
| `/data/$USER/Graph-De-Anonymization/logs/` | SLURM logs and the frozen manifests | no |

`GDA_DATA_DIR` overrides the NAS root and `GDA_SIF` the image path. The 500 GB
`/home` quota is why the image lives on `/data` (1.9 TB) rather than beside the
code — see the cluster guide.

## The files

| File | What it is |
|---|---|
| `grid.py` | the cells, and nothing else. Owns the cross product so an array's size and a task's config cannot disagree. Python 3.6 compatible on purpose: the head node runs it. |
| `report1_cell.py` | runs one cell and writes its record. Calls the baseline's algorithms; owns the split, the scoring and the provenance. |
| `collect.py` | records → CSV, or a grouped summary. Also 3.6 compatible. |
| `common.sh` | `REPO`/`SIF`/log paths and `in_container`. Sourced, not run. |
| `container_report1.def`, `build_container.sh` | the image. One image for both halves of the repo: Report 1 is CPU, Adv-GD adds `--nv`. |
| `00_generate_pairs_job.sh` | step 0: fetch the parent if needed, cut every pair in one process. |
| `10_seed_based_job.sh`, `20_seed_free_job.sh` | one array task = one cell. |
| `submit_report1.sh` | freezes the manifests and submits the three stages. |

## Three axis conventions, each a choice and not the paper's

**β is a divergence, not a deletion rate.** `--beta-mode divergence` (the
default) generates each pair at `1-sqrt(1-β)` per copy, so the two released
graphs differ by β. `nominal` deletes at β from each copy and they differ by
~2β. The two modes write into **separate roots** — `runs/report1_divergence/`
and `runs/report1_nominal/` — because pair directory names round β to whole
percent and nominal 0.05 collides with divergence 0.10. Same argument as
`src/adv_gd/sweep.py`.

**β=0 is in the default sweep as a control, not as padding.** A pair at β=0 is
the parent graph twice over, relabelled. It is the cell that says whether the
reimplementation reproduces the report at all; anything well below ~90% there
is a bug, not a finding. Run it first.

**The seed set is drawn, not sliced.** The generator writes ground truth sorted
by parent arXiv id, so taking the first 20% of `mapping.txt` — which is what
`src/seed_based.py` does — is a chronologically structured seed set rather than
an arbitrary one. The default is a shuffle under an explicit seed;
`--seed-draws -1` restores the prefix.

## What `report1_cell.py` changes, and why

The algorithms are the baseline's, imported and called: `seed_based.SeedFreeD`
is the report's Algorithm 1 and `seed_free.determine_seed_pairs` is its ranking
dispatch. Three things around them are not reused.

1. **The split.** `src/seed_based.py` hardcodes `train_portion = 0.20` and
   `ecce_threshold = 0.5` inside `main()`, *below* the argument parser — and
   both of the report's figures vary exactly those two numbers. `--method_name`,
   `--integrate_with_Seed_Based` and `--number_of_top_index_pairs` are parsed
   there and never read. A sweep has to own the split.
2. **The membership test.** The baseline keeps a ground-truth pair when
   `u in G1 and u in G2`: the same label on both sides. On the inherited
   `labeled_dev` pair, whose graphs share an id space, that is merely
   redundant. On a generated pair it is **wrong** — the generator relabels each
   side independently, so `u in G2` tests a G1 label against a permuted G2 and
   silently drops nearly all of the ground truth. Here it is
   `u in G1 and v in G2`. If you ever point the baseline's own `main()` at a
   generated pair, expect a near-empty validation list and an accuracy that
   means nothing.
3. **The scoring container.** `get_seed_pairs` does `tup in validation_pairs`
   against a *list*; at the full ranking that is 27k linear scans of a
   27k-element list. A `set` is passed instead. Membership is the only
   operation performed on it.

Everything else — score function, eccentricity test, iteration to a fixed
point, the accuracy definitions — is called rather than reimplemented, so that
a difference from the report is a difference in the data and not in the code.

## Costs and failure modes worth knowing before you shorten a walltime

- **A seed-free cell is much slower than a seed-based one.** `k-truss` and
  `k-core` decompose the graph by repeated subgraph extraction, and *every*
  cell also scores the full ranking before any top-k is integrated. Defaults:
  12h for a seed-based cell, 24h for a seed-free one. Trim both after the first
  batch reports its `seconds`.
- **Seed-free noise draws are expensive and probably not worth it.** The
  rankings are deterministic, so a second draw buys an interval over the noise
  and costs another k-truss decomposition. `submit_report1.sh` therefore runs
  the seed-free arrays at `--sf-pair-seeds 0` by default while the seed-based
  arrays get all three.
- **`nx.pagerank(G, alpha=1)`** — the baseline's call — is power iteration at
  α=1 and can raise `PowerIterationFailedConvergence`;
  `nx.eigenvector_centrality` defaults to `max_iter=100` and can do the same. A
  cell that raises still writes a record, with `status_exit: "error"` and the
  traceback, so these show up in `collect.py` as rows rather than as silence.
- **`--mem`**: 16G per seed-based cell, 48G per seed-free cell. The container
  build needs 64G (see above).

## Not covered

- **node2vec**, in the report's Table 2 at 0.01%, has no branch in
  `src/seed_free.py`. It is excluded rather than reimplemented.
- **A GPU array for `src/adv_gd/`.** `sweep.py --task-id` is already written for
  one; `10_seed_based_job.sh` is the template — add `--gpus=1`,
  `--constraint=RTX6000ADA`, and set `GDA_NV=1` so `in_container` passes `--nv`.
- **A held-out pair.** Every number here is measured on the same pairs the
  settings were chosen on. The `status` field (`exploratory` by default) makes
  that visible; it does not fix it.
