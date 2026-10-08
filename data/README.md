# Datasets

Four graph-pair sets, inherited from Privacy Aware Computing Project 1 (Dec 2024).
**The thing that distinguishes them is how much ground truth you get**, not their size.

| Directory | G1 / G2 edges | Ground truth | What it is for |
|---|---|---|---|
| `labeled_dev/` | 634,148 | **full** — 27,770 pairs | The only set the scripts read. Full truth, so you can measure accuracy. |
| `task_seed_based/` | 290,527 | **500 seed pairs only** | Graded task input: you are given 500 seeds and must produce the rest. No truth to score against locally. |
| `task_seed_free/` | 729,810 | **none** | Graded task input for the seed-free task. Nothing to score against locally. |
| `small_ca_grqc/` | 14,496 | **full** — 5,242 pairs | Small labeled pair (arXiv ca-GrQc). Fast enough to iterate on. Not wired into the scripts (see below). |

Every edge list is whitespace-separated `u v`, undirected, read with `nx.read_edgelist`.
Every mapping file is one `u v` ground-truth pair per line.

## Why only `labeled_dev/` works today

Both scripts gate graph loading on a **literal substring of the directory path**:

```python
if 'labeled_dev' in str(file_path):     # src/seed_based.py, src/seed_free.py
```

and then pick files out of that directory by substring too — `'G1.edgelist'`,
`'G2.edgelist'`, `'mapping'`. So:

- Pointing a script at `task_seed_based/` or `task_seed_free/` loads **nothing**, silently,
  and then fails on the `None` graphs. It is not a supported mode.
- `small_ca_grqc/` additionally fails the filename test: its files are
  `ca-GrQc_Graph1_edgelist.txt`, which does not contain the substring `G1.edgelist`.
  Using it means replacing the discovery logic, not renaming the folder.

Before this restructure the guard string was `'validation_Dataset'`. If you rename these
directories again, **grep for the directory name in `src/` first** — a rename alone will
break loading without raising anything at the point of failure.

## Naming history

These directories were renamed from the inherited layout. Old → new:

| Before | Now |
|---|---|
| `Project_1/Graph1/validation_Dataset/` | `data/labeled_dev/` |
| `Project_1/Graph1/Seed_based/` | `data/task_seed_based/` |
| `Project_1/Graph1/Seed_free/` | `data/task_seed_free/` |
| `Project_1/smallValidGraphPair/` | `data/small_ca_grqc/` |

The individual filenames inside were deliberately **not** changed, because the discovery
logic matches on them.

## Generated pairs

Nothing generated is written under `data/` — this directory holds source datasets
only. `src/beta_noise_generator/perturbation.py` writes pairs cut from a single
parent graph, with exact ground truth and a known β, into `runs/` at the repo root.
That directory is gitignored — regenerate from the parent and the seed recorded in
`stats.json` rather than committing it.

The baseline scripts build their dataset path as `os.path.join(--path,
'labeled_dev')`, so a generated pair loads only if it sits in a directory of
exactly that name; pass `--compat-layout` to the generator to write that nesting.

## Not in this repo

`gplus_combined.txt/` (Google+, 1.3 GB, 107,614 nodes) sits at the repo root and is
gitignored. It is a single directed edge list, not a pair, so using it for matching means
generating the pair yourself by perturbation.

`data/cit-HepTH/` (arXiv HEP-Th citations, 27,770 nodes, 352,807 directed edges) is
present on disk and gitignored. It is the parent graph for every Adv-GD run, chosen
because the Li et al. paper reports HepTh in its Table 2. Download it from
<https://snap.stanford.edu/data/cit-HepTh.html>; only `cit-HepTh.txt` is used.

Two cautions. The archive also unpacks `cit-HepTh-abstracts/`, roughly 29,000 tiny `.abs`
files -- do not glob or `du` the directory, it is slow enough to look like a hang. And the
edge list is directed and contains both directions for some pairs; the loader reads it as
undirected, which is what the paper's node counts imply.

Like Google+, it is a single graph, not a pair: `src/adv_gd/sweep.py --generate` cuts the
pairs from it.
