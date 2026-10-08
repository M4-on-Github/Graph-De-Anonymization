"""Adv-GD driver: run the phases on a generated pair and score the result.

Reads a pair directory written by `src/beta_noise_generator/perturbation.py`
(G1.edgelist, G2.edgelist, mapping.txt, stats.json), runs Phase 1 and Phase 2,
and reports anchor quality in the form of the paper's Table 2.

`--phases 12` stops after alignment, which is the fast loop: Table 2 is an
anchor-quality table and needs nothing further. `--phases 123` adds
propagation and reports chi and gamma.

Embeddings are cached as Z1.npz / Z2.npz inside the pair directory, because
Phase 1 costs about two minutes per graph and Phase 2 is where the parameters
actually get swept.

CLI:
    python src/adv_gd/adv_gd.py --pair runs/cit-HepTh_b10_s0 --dataset hepth
    python src/adv_gd/adv_gd.py --pair runs/cit-HepTh_b10_s0 --dataset hepth --refresh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alignment import DEFAULTS, align, evaluate_anchors   # noqa: E402
from embedding import gae_embed, reconstruction_auc       # noqa: E402
from propagation import evaluate, propagate               # noqa: E402
from provenance import RunRecord, capture_stderr          # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# The paper gives no epoch count for Phase 1. Kipf and Welling's default of 200
# leaves the two graphs at different losses and a much lower alignment ceiling
# (0.82 against 0.97) -- see the reproduction log in `adv_gd_protocol.md`. It is
# declared here rather than left to `gae_embed`'s default so that a run record
# always states it.
GAE_EPOCHS = 600

# Table 1, p.749. D is the embedding dimension, not a degree.
DATASETS = {
    "fb":    {"dim": 16,  "n_adv": 1000,  "n_anch": 400,  "m": 5000},
    "hepth": {"dim": 32,  "n_adv": 5000,  "n_anch": 1000, "m": 10000},
    "gplus": {"dim": 300, "n_adv": 20000, "n_anch": 5000, "m": 50000},
}

# Table 2, p.751 -- anchors reported as correct/found. The only exact integers
# the paper gives, which is why they are the primary reproduction target.
PAPER_TABLE2 = {
    "fb":    {10: (90, 90),    20: (77, 77),    30: (73, 74),
              40: (69, 70),    50: (65, 67)},
    "hepth": {10: (753, 755),  20: (714, 729),  30: (652, 670),
              40: (585, 607),  50: (472, 512)},
    "gplus": {10: (4921, 4956), 20: (4912, 4918), 30: (4899, 4906),
              40: (4882, 4889), 50: (4788, 4815)},
}


def load_pair(pair: Path) -> tuple[nx.Graph, nx.Graph, dict, dict]:
    """Load the two graphs, the ground-truth mapping, and the generator stats."""
    Ga = nx.read_edgelist(pair / "G1.edgelist", nodetype=str)
    Gu = nx.read_edgelist(pair / "G2.edgelist", nodetype=str)
    truth = {}
    with (pair / "mapping.txt").open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                truth[parts[0]] = parts[1]
    stats = json.loads((pair / "stats.json").read_text())
    return Ga, Gu, truth, stats


def embed_cached(G: nx.Graph, path: Path, dim: int, seed: int, refresh: bool,  # noqa: E501
                 epochs: int = GAE_EPOCHS,
                 verbose: bool = True) -> tuple[np.ndarray, list]:
    """Embed G, reusing a cached .npz when it was built the same way.

    Both the node set and the settings are checked, not just the file's
    existence. A cache written for a different graph would silently misalign
    every row, and one written at a different epoch count or dimension is a
    different embedding entirely -- epoch count moves the alignment ceiling
    from 0.82 to 0.97, so quietly reusing the wrong one would invalidate the
    run without any sign of it. Older caches carry no settings, so they are
    rejected rather than trusted.
    """
    if path.exists() and not refresh:
        blob = np.load(path, allow_pickle=True)
        nodes = [str(x) for x in blob["nodes"]]
        cached = {k: int(blob[k]) for k in ("epochs", "dim", "seed")
                  if k in blob}
        want = {"epochs": epochs, "dim": dim, "seed": seed}
        if set(nodes) != set(G.nodes()):
            print(f"  {path.name} is for a different graph; re-embedding",
                  file=sys.stderr)
        elif cached != want:
            print(f"  {path.name} was built as {cached or 'unrecorded'}, "
                  f"need {want}; re-embedding", file=sys.stderr)
        else:
            if verbose:
                print(f"  reusing {path.name} ({len(nodes):,} nodes, "
                      f"{epochs} epochs)", file=sys.stderr)
            return blob["Z"], nodes
    Z, nodes = gae_embed(G, dim=dim, epochs=epochs, seed=seed, verbose=verbose)
    # Write-then-rename. Several cluster jobs can share a pair directory, and a
    # direct savez to the final path lets one job read another's half-written
    # file. os.replace is atomic within a filesystem.
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, Z=Z, nodes=np.array(nodes, dtype=object),
                        epochs=epochs, dim=dim, seed=seed)
    os.replace(tmp, path)
    return Z, nodes


def run(pair: Path, dataset: str, seed: int = 0, refresh: bool = False,
        cfg: dict | None = None, phases: str = "123", theta: float = 0.5,
        eccen_support: str = "all", gae_epochs: int = GAE_EPOCHS,
        status: str = "exploratory", note: str = "", record: bool = True,
        gae_seed: int | None = None, verbose: bool = True) -> dict:
    """Run the phases on one pair.

    `seed` drives Phase 2. `gae_seed` drives Phase 1 and defaults to `seed`,
    which is how every result recorded before 2026-10-08 was produced. Pin it
    for a seed sweep: the embedding cache is keyed on its seed, so leaving them
    coupled makes every cell re-train both GAEs -- about twenty wasted minutes
    per cell -- to answer a question that is entirely about Phase 2's
    adversarial initialization.
    """
    params = DATASETS[dataset]
    gae_seed = seed if gae_seed is None else gae_seed
    Ga, Gu, truth, stats = load_pair(pair)
    beta_pct = int(round(stats["beta"] * 100))

    # The record is built before any work, because everything identifying the
    # run is known now, and finished in a `finally`, because a run that fails or
    # is killed partway is still data about where it got to. See
    # `docs/experiment_logging_policy.md`.
    resolved = {**DEFAULTS, **(cfg or {})}
    rec = RunRecord("adv_gd", pair.name, status, note) if record else None
    if rec:
        rec.set(seed=seed, gae_seed=gae_seed, beta=stats["beta"],
                dataset=dataset, phases=phases,
                params=params, config=resolved, gae_epochs=gae_epochs,
                theta=theta, eccen_support=eccen_support,
                n_restarts=resolved["n_restarts"],
                selection="mean_match_cosine",
                matchable_nodes=stats.get("matchable_nodes"),
                pair_stats=stats, pair_dir=str(pair))
        rec.add_inputs(pair / "G1.edgelist", pair / "G2.edgelist",
                       pair / "mapping.txt", pair / "stats.json")
        rec.seal_id()

    print(f"pair {pair.name}  beta={stats['beta']}  "
          f"G1 {Ga.number_of_nodes():,}n/{Ga.number_of_edges():,}e  "
          f"G2 {Gu.number_of_nodes():,}n/{Gu.number_of_edges():,}e",
          file=sys.stderr)

    t0 = time.time()
    with capture_stderr() as console:
        try:
            out = _run_phases(pair, params, Ga, Gu, truth, stats, beta_pct,
                              dataset, seed, refresh, cfg, phases, theta,
                              eccen_support, gae_epochs, t0,
                              rec.run_id if rec else "unrecorded", gae_seed,
                              verbose)
            exit_status, error = "ok", None
        except BaseException as exc:      # KeyboardInterrupt included: a killed
            import traceback               # run is recorded, not discarded.
            exit_status, error = "error", traceback.format_exc()
            out = {"failed_in": f"{type(exc).__name__}: {exc}"}
            raise
        finally:
            if rec:
                if "3" in phases and "mapping_path" in out:
                    rec.add_outputs(Path(out.pop("mapping_path")))
                rec.set(selected_seed=out.get("selected_seed"),
                        restarts=out.get("restarts"),
                        refine_history=out.get("refine_history"),
                        config=out.get("config", resolved))
                rec.metric(**{k: out.get(k) for k in
                              ("auc_g1", "auc_g2", "anchors_found",
                               "anchors_correct", "anchor_precision",
                               "paper_correct", "paper_found", "mapped",
                               "correct", "chi", "gamma")})
                rec.finish(exit_status, error, console.getvalue(), verbose)
                out["run_id"] = rec.run_id
    (pair / "adv_gd_result.json").write_text(json.dumps(out, indent=2))
    return out


def _run_phases(pair: Path, params: dict, Ga, Gu, truth: dict, stats: dict,
                beta_pct: int, dataset: str, seed: int, refresh: bool,
                cfg: dict | None, phases: str, theta: float,
                eccen_support: str, gae_epochs: int, t0: float,
                run_id: str, gae_seed: int, verbose: bool) -> dict:
    """The three phases. Split out of `run` so the record can wrap it."""
    print("phase 1: GAE embeddings", file=sys.stderr)
    Za, nodes_a = embed_cached(Ga, pair / "Z1.npz", params["dim"], gae_seed,
                               refresh, gae_epochs, verbose)
    Zu, nodes_u = embed_cached(Gu, pair / "Z2.npz", params["dim"], gae_seed,
                               refresh, gae_epochs, verbose)
    auc_a = reconstruction_auc(Ga, Za, nodes_a)
    auc_u = reconstruction_auc(Gu, Zu, nodes_u)
    print(f"  reconstruction AUC  G1 {auc_a:.4f}  G2 {auc_u:.4f}", file=sys.stderr)

    print("phase 2: adversarial alignment", file=sys.stderr)
    res = align(Za, nodes_a, Zu, nodes_u, Ga, Gu,
                n_adv=params["n_adv"], n_anch=params["n_anch"], m=params["m"],
                cfg=cfg, seed=seed, verbose=verbose)

    score = evaluate_anchors(res["anchors"], truth)
    paper = PAPER_TABLE2.get(dataset, {}).get(beta_pct)

    out = {
        "pair": pair.name,
        "dataset": dataset,
        "beta": stats["beta"],
        "seed": seed,
        "matchable_nodes": stats.get("matchable_nodes"),
        "gae_epochs": gae_epochs,
        "auc_g1": auc_a,
        "auc_g2": auc_u,
        "anchors_found": score["found"],
        "anchors_correct": score["correct"],
        "anchor_precision": score["precision"],
        "paper_correct": paper[0] if paper else None,
        "paper_found": paper[1] if paper else None,
        "refine_history": res["history"],
        "selected_seed": res["selected_seed"],
        "restarts": res["restarts"],
        "config": {k: v for k, v in res["config"].items()},
        "seconds": round(time.time() - t0, 1),
    }

    if "3" in phases:
        print("phase 3: propagation", file=sys.stderr)
        mapping = propagate(Ga, Gu, res["anchors"], res["candidates"],
                            theta=theta, support=eccen_support, verbose=verbose)
        final = evaluate(mapping, truth, stats.get("matchable_nodes"))
        # The output mapping is the run's actual result. It used to be discarded
        # once chi had been computed, which made the determinism check we ran on
        # it impossible to repeat.
        mpath = pair / f"out_mapping_{run_id}.txt"
        mpath.write_text("".join(f"{u} {v}\n" for u, v in mapping.items()))
        out.update({
            "mapping_path": str(mpath),
            "mapped": final["mapped"],
            "correct": final["correct"],
            "chi": final["chi"],
            "gamma": final["gamma"],
            "theta": theta,
            "eccen_support": eccen_support,
        })

    print("", file=sys.stderr)
    print(f"  anchors (ours)   {score['correct']}/{score['found']}  "
          f"precision {score['precision']:.4f}", file=sys.stderr)
    if paper:
        print(f"  anchors (paper)  {paper[0]}/{paper[1]}  "
              f"precision {paper[0] / paper[1]:.4f}", file=sys.stderr)
    if "3" in phases:
        print(f"  de-anonymized    {out['correct']:,}/{out['mapped']:,} mapped  "
              f"chi {out['chi']:.4f}  gamma {out['gamma']:.4f}", file=sys.stderr)
    print(f"  {time.time() - t0:.0f}s total", file=sys.stderr)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run Adv-GD on a generated pair.")
    ap.add_argument("--pair", type=str, required=True,
                    help="Pair directory from the beta noise generator.")
    ap.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS),
                    help="Which Table 1 parameter row to use.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", action="store_true",
                    help="Recompute embeddings even if a cache exists.")
    ap.add_argument("--adv-epochs", type=int, default=None)
    ap.add_argument("--adv-steps", type=int, default=None)
    ap.add_argument("--refine-rounds", type=int, default=None)
    ap.add_argument("--csls-k", type=int, default=None)
    ap.add_argument("--status", choices=("exploratory", "confirmatory"),
                    default="exploratory",
                    help="Declare this before the run, not after. A sweep is "
                         "exploratory; only a run confirming a stated prior "
                         "expectation is confirmatory.")
    ap.add_argument("--note", type=str, default="",
                    help="Why this run was made. Goes in the record.")
    ap.add_argument("--no-record", action="store_true",
                    help="Skip the run record. For debugging only -- an "
                         "unrecorded run cannot be reported.")
    ap.add_argument("--gae-seed", type=int, default=None,
                    help="Phase 1 seed. Defaults to --seed. Pin it across a "
                         "seed sweep so the embedding cache is reused and the "
                         "sweep varies only Phase 2.")
    ap.add_argument("--gae-epochs", type=int, default=GAE_EPOCHS,
                    help="Phase 1 training epochs. The paper states none.")
    ap.add_argument("--n-restarts", type=int, default=None,
                    help="Adversarial runs to select among by mean cosine.")
    ap.add_argument("--phases", type=str, default="123",
                    choices=("12", "123"),
                    help="Stop after alignment, or run propagation too.")
    ap.add_argument("--theta", type=float, default=0.5,
                    help="Eccentricity threshold; the paper uses 0.5 throughout.")
    ap.add_argument("--eccen-support", type=str, default="all",
                    choices=("all", "nonzero"),
                    help="Whether sigma covers the score vector's zeros.")
    ap.add_argument("--normalize", type=str, default=None,
                    choices=("none", "renorm", "center_renorm"),
                    help="Embedding preprocessing before Phase 2.")
    args = ap.parse_args(argv)

    cfg = {}
    for flag, key in (("adv_epochs", "adv_epochs"), ("adv_steps", "adv_steps"),
                      ("refine_rounds", "refine_rounds"), ("csls_k", "csls_k"),
                      ("normalize", "normalize"), ("n_restarts", "n_restarts")):
        val = getattr(args, flag)
        if val is not None:
            cfg[key] = val

    run(Path(args.pair), args.dataset, seed=args.seed, refresh=args.refresh,
        cfg=cfg or None, phases=args.phases, theta=args.theta,
        eccen_support=args.eccen_support, gae_epochs=args.gae_epochs,
        status=args.status, note=args.note, record=not args.no_record,
        gae_seed=args.gae_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
