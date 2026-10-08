# Experiment logging and provenance policy

Scope: every experimental run in this repository — Adv-GD runs, perturbation-pair
generation, baseline runs, and any sweep or study built out of them.

This policy exists because of a specific failure we created. Over roughly two days of
Adv-GD work we executed on the order of twenty-five scored runs: a four-seed study, a
three-by-four preprocessing grid, two eccentricity settings, two GAE epoch counts, a
supervised ceiling study, and several end-to-end runs. **Exactly two of those runs still
exist as data.** Every other one was overwritten, because the driver wrote its result to
`<pair>/adv_gd_result.json` — one slot per pair, rewritten on every invocation. What
survives of the rest is prose in `src/adv_gd/adv_gd_protocol.md` and a set of
block-buffered `.log` files that had to be read by eye. The numbers in that spec are
currently **assertions, not records.** No reviewer, teammate, or future version of us can
check them against anything.

Two conclusions follow, and they shape everything below.

1. A result that is not archived at the moment it is produced is lost, not deferred.
   Re-running is not a substitute: the environment, the code, and the random draw have all
   moved on.
2. The expensive part of an experiment is the compute; the record costs milliseconds.
   There is no run small enough to be worth not logging.

## What provenance does and does not buy

Stating this first, because the main risk of a thorough logging policy is that it makes
results feel more trustworthy than they are.

**It buys auditability.** Given a record you can say exactly what code, what inputs, what
seeds, and what environment produced a number, and you can detect when a later run differs
in any of those. That is the whole of the benefit.

**It does not buy reproducibility.** Phase 1 trains on CUDA. A fixed seed is
bit-reproducible *on this machine with this torch build* — we measured that, two runs at
seed 0 agreeing on `sum|W|` to all printed digits — and that measurement does not transfer
to another GPU, another driver, or another torch version. The record cannot make a run
portable. What it does is make the mismatch **visible** instead of silent, which is the
achievable goal.

**It does not make a single run into evidence.** A fully documented run of a bimodal
procedure is still a coin flip (see the seed section of `src/adv_gd/adv_gd_protocol.md`:
four seeds spanning 0 to 832 correct anchors, with no middle). Rich metadata on one run is
if anything a hazard, because it looks like rigour. The schema therefore forces the sample
size and the selection rule into every record, so that a number can never be quoted
without them.

**It does not check correctness.** Hashing an output mapping proves determinism, not that
the mapping is right. Two runs agreeing byte-for-byte on a wrong answer is a thing that
happens.

## Three classes of thing, three destinations

| Class | Example | Regenerable? | Where | Tracked in git? |
|---|---|---|---|---|
| **Record** — what was run, by what code, on what input, with what result | `record.json`, the ledger line, a dirty-tree diff | **No.** The environment that produced it is gone. | `results/runs/` | **Yes** |
| **Bulk artifact** — the bytes a run consumed or produced | pair edge lists, GAE embeddings (`Z*.npz`), output node mappings | Yes, from a record plus a seed | `runs/` | No |
| **Narrative** — what we concluded and why | `adv_gd_protocol.md`, `perturbation_protocol.md` | No | `src/<module>/` | Yes |

The split between the first two is the load-bearing decision. `.gitignore` previously
excluded all of `results/` except the 2024 archive, on the rule that generated output is
regenerable. **That rule is wrong for records and right for bytes.** A 2 KB JSON record
describing a run on a torch nightly from March 2026 can never be regenerated; a 300 KB
output mapping can be, from the record. So records are now tracked and bulk stays ignored,
with each artifact's SHA-256 carried in the record. That gives the property we actually
need: if a regenerated artifact hashes differently from the one a conclusion was drawn
from, we find out.

Consequence to accept deliberately: the hash in a tracked record may point at bytes nobody
still has. That is a worse outcome than keeping everything and a much better one than
keeping nothing, and it is the only version that stays inside a repository that is not a
data store. Where an artifact is both small and genuinely irreplaceable, copy it *into* the
record directory rather than hashing it in place — which is why each run record embeds the
generator's `stats.json` verbatim instead of referencing it.

## Layout

```
results/runs/
  index.jsonl                     append-only ledger, one line per run, tracked
  <run_id>/
    record.json                   the full record, tracked
    diff.patch                    present only when the tree was dirty, tracked
    stderr.log                    the run's console output, tracked when small
runs/<pair>/
  G1.edgelist G2.edgelist         pair input, hashed into the record
  mapping.txt stats.json          ground truth and generator provenance
  Z1.npz Z2.npz                   Phase 1 cache, stamped with epochs/dim/seed
  out_mapping_<run_id>.txt        Phase 3 output, hashed into the record
  adv_gd_result.json              last run on this pair; a convenience, not the record
```

`run_id` is `<UTC timestamp>_<pair>_<8 hex>`, for example
`20261007T1430Z_cit-HepTh_b10_s0_3f9a1c02`. The timestamp sorts; the hash is over the code
state plus the full configuration, so two runs that differ in any input differ in id, and a
genuine repeat of an identical configuration is visible as the same trailing hash at a
different time.

The hash covers **everything in the record except the volatile and the measured** rather
than a list of known field names. The first version used an allow-list and so could not see
fields a later caller added: when the perturbation generator was wired in, two generations
differing in `beta` and `node_sample` produced the *same* id. Excluded from the hash are
timestamps, environment, argv, note and status, and anything the run measured -- an id
identifies the question asked, so a repeat that produced a different answer must still be
recognisable as a repeat.

## Record schema

Every field is there to answer a question somebody will actually ask. Grouped by question.

**What code?** `git.head`, `git.branch`, `git.dirty`, `git.diff_sha256`. A bare commit hash
is a lie in a repository where the work is uncommitted — all of the Adv-GD module is, at the
time of writing. When the tree is dirty the uncommitted diff is written next to the record
as `diff.patch` and hashed, so `head` plus `diff.patch` reconstructs the exact source.
`git.dirty: true` with no diff stored is not an acceptable record.

**What environment?** `env.python`, `env.platform`, `env.host`, `env.packages` (torch,
numpy, networkx, scipy), `env.cuda` (availability, device name, build), `env.hashseed`.
Torch here is a dated nightly (`2.12.0.dev20260329+cu128`); "torch 2.12" would not identify
it. `hashseed` is recorded because the inherited baseline is hash-order sensitive and this
reimplementation was measured not to be — a claim that needs the value it was measured
under.

**What input?** `pair`, `inputs[]` with a path, byte size and SHA-256 for each of
`G1.edgelist`, `G2.edgelist`, `mapping.txt`, plus `pair_stats` holding the generator's
`stats.json` inline. The pair *name* is not an identifier: regenerate a pair with a changed
generator and the name is unchanged while every number computed from it has moved.

**What configuration?** `argv` exactly as invoked, `params` (the resolved dataset
parameters), `config` (the resolved Phase 2 hyperparameters, including defaults the caller
did not pass), `gae_epochs`, `theta`, `eccen_support`, `seed`. Resolved values, not the
flags — a default that changes later must not silently rewrite the meaning of an old record.

**How large was the sample, and what was selected?** `n_restarts`, `restarts[]` (the mean
cosine of each), `selected_seed`, `selection: "mean_match_cosine"`. This group is
mandatory. A number drawn from a selection over restarts is not the same kind of claim as a
number from a single run, and the record must not permit the distinction to be lost.

**Exploratory or confirmatory?** `status`, one of `exploratory` or `confirmatory`, plus a
free-text `note`. Set it before the run, not after. This is the cheapest available guard
against reporting a cell found by searching as though it had been predicted; it is weaker
than pre-registration and better than nothing. A sweep is exploratory. A re-run of one
setting to confirm a specific prior expectation is confirmatory. If in doubt it is
exploratory.

**What came out?** `metrics` (AUC per graph, anchors found/correct/precision, mapped,
correct, chi, gamma, the paper's Table 2 cell for comparison), `refine_history`,
`outputs[]` with a path, size and SHA-256 per produced artifact, and `seconds`. The output
node mapping is written and hashed. Previously it was discarded: chi was reported and the
25,782 pairs it was computed from were thrown away, which made the determinism check we ran
impossible to repeat.

**Did it finish?** `status_exit`, one of `ok`, `error`, `killed`, or `verify_failed`, with
`error` carrying the traceback. `verify_failed` is for a run that executed correctly and
whose *output* missed a criterion stated in advance -- a generated pair whose measured edge
overlap falls outside tolerance of (1-beta)^2. That is neither success nor a crash, and
collapsing it into either loses the most interesting thing the generator can report. A killed run is data — the beta-interpretation run was killed for low system
memory partway through Phase 3, and that it got as far as it did is worth knowing. Records
are written for failures too.

## Rules

1. **Records are append-only and immutable.** Never edit or delete a `record.json`. A run
   found to be invalid gets a new record whose `note` explains it and whose
   `supersedes` names the old `run_id`. The old record stays. We have already lost one
   session's experiments to overwriting; the fix is structural, not discipline.
2. **One record per run, written even on failure.** Write it in a `finally`.
3. **Every reported number cites a `run_id`.** Tables in `adv_gd_protocol.md` and in any
   report name the runs behind them. A number with no run id is a claim, and should be
   labelled as one.
4. **Studies are compositions, not new formats.** A sweep is N runs with N records plus a
   small script that reads `index.jsonl`. It does not get a bespoke JSON schema —
   `seed_study_phase2.json` and `norm_seed_grid.json` were both one-off shapes that only
   the script that wrote them could read.
5. **Logs are not evidence.** Background shell logs here are block-buffered: `betatest.log`
   showed epoch 120 of 600 while the run had in fact completed two phases, and we
   misreported progress from it once. Trust `record.json` and artifact hashes on disk.
   `stderr.log` is archived as a courtesy for reading, not as a source of truth. Note
   that it is buffered in memory and written only when the run finishes, so it tells you
   nothing while a run is in progress -- check the artifacts in `runs/` for that. The
   same session that wrote this policy twice misread a run's state from a progress log,
   once reporting a finished run as killed.
6. **Never hand-edit a bulk artifact.** The two `cit-HepTh_b10_s0` embedding caches were
   retroactively stamped with `epochs=600 dim=32 seed=0` to match a new cache-validation
   check. That was defensible — the values were known with certainty from the run that
   produced them — and it is exactly the kind of act that must not become routine, because
   the stamp now asserts something no record witnesses. Re-generate instead.

## What this does not cover, and what to watch for

- **One pair per beta.** Every Adv-GD figure we have rests on a single noise draw. Logging
  does not create a confidence interval over the pair, and no amount of recorded metadata
  changes that `N = 1` on the dimension most likely to matter.
- **Hyperparameter search is not yet separated from evaluation.** Preprocessing mode and
  epoch count were chosen by looking at scored outcomes on the same pair the result is
  reported on. The `status` field makes that visible; it does not fix it. A held-out pair
  for final numbers is the real fix.
- **A pair directory name does not identify the pair.** The generator names pairs
  `<stem>_b<NN>_s<seed>`, which encodes neither the sampling fraction nor the mode, so two
  different configurations write to one directory and the second replaces the first. This
  bit during this feature's own testing: a 5% node sample and a 0.4% one both landed in
  `cit-HepTh_b10_s0`, leaving the first record's output digests pointing at bytes that no
  longer existed. Renaming the scheme would invalidate every path in the protocol docs and
  the embedding caches, so the overwrite is reported rather than prevented -- on stderr at
  the time, and as `overwrote_existing` in the record of the run that did it. **Generate
  into a separate `--out` root when varying anything the name does not capture.**
- **Selection by mean cosine is itself a modelling choice** that the paper specifies and we
  validated against ground truth on one dataset. Recording it per run is what will let a
  later disagreement be traced.

## Citation

The critical-appraisal framing of this policy follows: Kassis, T., Agarwal, V., He, Y.,
Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural
Knowledge for Research Agents.* arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
