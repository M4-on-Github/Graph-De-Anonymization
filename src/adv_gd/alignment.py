"""Phase 2 of Adv-GD: adversarial alignment of two embedding spaces.

The two graphs are embedded independently, so their spaces are unrelated: node
v in G^a and its true counterpart in G^u land in arbitrary, different places.
This phase learns a single linear map W taking one space onto the other, with
no seed pairs, by training W to fool a discriminator that tries to tell mapped
source vectors from target vectors.

Paper provenance (docs/SeedFree_Li.etal.pdf, section 3.1, p.747):
    equations 4-5   discriminator and W objectives
    equations 6-7   CSLS similarity
    equations 8-9   Procrustes refinement, solved by SVD
    train on the top N_adv degree nodes of each graph only
    anchors  = mutual CSLS nearest neighbours among top N_anch degree nodes
    candidates = top 10 CSLS matches for each top M degree node of G^a
    model selection = mean cosine of the CSLS-matched pairs

The paper fixes the structure but not the training schedule. Everything in
DEFAULTS below is ours, taken from MUSE (reference [21]), which is where the
paper takes CSLS from and which Phase 2 otherwise follows closely. See
`adv_gd_protocol.md`.

Library use:
    from alignment import align
    res = align(Za, nodes_a, Zu, nodes_u, Ga, Gu, n_adv=5000, n_anch=1000, m=10000)

    # The same refinement and candidate sets from a given seed set (node-id
    # pairs) in place of adversarial training; returns what `align` returns.
    from alignment import align_from_seeds
    res = align_from_seeds(Za, nodes_a, Zu, nodes_u, Ga, Gu, n_anch=1000,
                           m=10000, seed_pairs=pairs)

CLI:
    python src/adv_gd/alignment.py --self-test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

# Unstated by the paper; MUSE defaults. Changing these changes results, so they
# are reported in the run record rather than buried here.
DEFAULTS = {
    "csls_k": 10,            # neighbours in the CSLS local-scaling term
    "disc_hidden": 2048,     # discriminator width
    "disc_layers": 2,        # hidden layers
    "disc_dropout": 0.1,
    "disc_input_dropout": 0.1,
    "disc_leaky": 0.2,
    "label_smooth": 0.2,
    "adv_epochs": 5,
    "adv_steps": 2000,       # discriminator/W update pairs per epoch
    "refine_rounds": 5,
    "ortho_beta": 0.01,      # strength of the orthogonalization pull on W
    "normalize": "none",     # embedding preprocessing; see _preprocess
    "n_restarts": 4,         # independent adversarial runs; see _search
    "skip_adversarial": False,  # ablation: see the comment in `_search`
}

# Paper-specified, section 3.1 and 4.2.
DISC_LR = 0.1               # SGD
MAP_LR = 0.01               # Adam
BATCH = 32
TOP10 = 10                  # candidate-set size per node


# --------------------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------------------

def _unit(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _preprocess(X: torch.Tensor, mode: str) -> torch.Tensor:
    """Put an embedding matrix on the scale the adversarial phase expects.

    The paper says nothing about this, and leaving it out is defensible on a
    literal reading -- but it is not harmless. GAE embedding norms grow as
    training goes on (mean 1.83, max 8.9 after 600 epochs on HepTh), and the
    discriminator can separate the two spaces on scale alone, which gives W no
    useful gradient. MUSE, which this phase otherwise follows, centers and
    renormalizes first. Measured effect is in the reproduction log of
    `adv_gd_protocol.md`.

    "none"           leave the embeddings as the GAE produced them -- default
    "renorm"         unit-norm each row
    "center_renorm"  subtract the mean row, then unit-norm (MUSE's default)

    Measured over four seeds each, `none` and `renorm` are indistinguishable:
    both reach a good alignment on 2 of 4 seeds, best precision 0.971 against
    0.976. So the default is the paper's literal reading, which costs nothing.

    Centering, which is MUSE's default, is a different matter -- 0 of 4 seeds,
    best precision 0.190 -- and the reason is structural rather than a tuning
    accident. The GAE decoder sigma(Z Z^T) is invariant under an orthogonal
    transform of Z but not under a translation of it, and that invariance is
    the whole reason a single linear W can align two independently trained
    spaces. Subtracting a per-graph mean shifts the two spaces by different
    vectors and breaks the symmetry the method relies on. MUSE centers word
    vectors, which carry no such constraint; carrying that default across was
    our error, not MUSE's.
    """
    if mode == "none":
        return X
    if mode == "center_renorm":
        X = X - X.mean(dim=0, keepdim=True)
    elif mode != "renorm":
        raise ValueError(f"unknown normalize mode {mode!r}")
    return _unit(X)


def csls_matrix(src: torch.Tensor, tgt: torch.Tensor, k: int,
                chunk: int = 2048) -> torch.Tensor:
    """CSLS(i, j) = 2 cos(src_i, tgt_j) - r_t(src_i) - r_s(tgt_j).

    r_t(i) is the mean cosine from src_i to its k nearest targets, and r_s(j)
    the mean cosine from tgt_j to its k nearest sources. Subtracting both
    penalizes hubs -- vectors that sit near everything -- which plain nearest
    neighbour search in high dimensions otherwise favours heavily (paper
    section 3.1, after equation 5).
    """
    src, tgt = _unit(src), _unit(tgt)

    r_t = torch.empty(src.shape[0], device=src.device)
    for lo in range(0, src.shape[0], chunk):
        sims = src[lo:lo + chunk] @ tgt.t()
        r_t[lo:lo + chunk] = sims.topk(min(k, sims.shape[1]), dim=1).values.mean(1)

    r_s = torch.empty(tgt.shape[0], device=tgt.device)
    for lo in range(0, tgt.shape[0], chunk):
        sims = tgt[lo:lo + chunk] @ src.t()
        r_s[lo:lo + chunk] = sims.topk(min(k, sims.shape[1]), dim=1).values.mean(1)

    out = torch.empty(src.shape[0], tgt.shape[0], device=src.device)
    for lo in range(0, src.shape[0], chunk):
        out[lo:lo + chunk] = (2 * (src[lo:lo + chunk] @ tgt.t())
                              - r_t[lo:lo + chunk, None] - r_s[None, :])
    return out


# --------------------------------------------------------------------------------------
# Discriminator
# --------------------------------------------------------------------------------------

class Discriminator(nn.Module):
    """Predicts the probability that a vector came from the mapped source space.

    Architecture is not given in the paper; this is MUSE's.
    """

    def __init__(self, dim: int, hidden: int, layers: int, dropout: float,
                 input_dropout: float, leaky: float):
        super().__init__()
        mods: list[nn.Module] = [nn.Dropout(input_dropout)]
        width = dim
        for _ in range(layers):
            mods += [nn.Linear(width, hidden), nn.LeakyReLU(leaky), nn.Dropout(dropout)]
            width = hidden
        mods += [nn.Linear(width, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def _orthogonalize(W: torch.Tensor, beta: float) -> None:
    """Pull W back towards orthogonality: W <- (1+b) W - b (W W^T) W.

    The refinement step (equation 9) assumes an orthogonal W, so letting the
    adversarial phase drift far from orthogonal makes the two stages
    inconsistent. MUSE applies this after every update.
    """
    with torch.no_grad():
        W.copy_((1 + beta) * W - beta * (W @ W.t()) @ W)


# --------------------------------------------------------------------------------------
# Adversarial training
# --------------------------------------------------------------------------------------

def train_mapping(Za: torch.Tensor, Zu: torch.Tensor, cfg: dict, seed: int = 0,
                  verbose: bool = True) -> torch.Tensor:
    """Learn W mapping the Za space onto the Zu space, without seeds.

    Za and Zu are already restricted to the top N_adv degree nodes of each
    graph: the paper trains on those only, on the grounds that high-degree
    nodes keep more of their structure under edge deletion.
    """
    dev = Za.device
    dim = Za.shape[1]
    torch.manual_seed(seed)

    W = torch.eye(dim, device=dev, requires_grad=True)
    disc = Discriminator(dim, cfg["disc_hidden"], cfg["disc_layers"],
                         cfg["disc_dropout"], cfg["disc_input_dropout"],
                         cfg["disc_leaky"]).to(dev)

    opt_d = torch.optim.SGD(disc.parameters(), lr=DISC_LR)
    opt_w = torch.optim.Adam([W], lr=MAP_LR)
    bce = nn.BCELoss()
    smooth = cfg["label_smooth"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()

    for epoch in range(1, cfg["adv_epochs"] + 1):
        d_running = w_running = 0.0
        for _ in range(cfg["adv_steps"]):
            ia = torch.randint(0, Za.shape[0], (BATCH,), generator=g).to(dev)
            iu = torch.randint(0, Zu.shape[0], (BATCH,), generator=g).to(dev)

            # Discriminator: source-mapped -> 1, target -> 0 (paper equation 4).
            with torch.no_grad():
                mapped = Za[ia] @ W.t()
            x = torch.cat([mapped, Zu[iu]])
            y = torch.cat([torch.full((BATCH,), 1 - smooth, device=dev),
                           torch.full((BATCH,), smooth, device=dev)])
            opt_d.zero_grad()
            loss_d = bce(disc(x), y)
            loss_d.backward()
            opt_d.step()

            # W: flip the labels, so W is rewarded for fooling the
            # discriminator (paper equation 5).
            x = torch.cat([Za[ia] @ W.t(), Zu[iu]])
            y = torch.cat([torch.full((BATCH,), smooth, device=dev),
                           torch.full((BATCH,), 1 - smooth, device=dev)])
            opt_w.zero_grad()
            loss_w = bce(disc(x), y)
            loss_w.backward()
            opt_w.step()
            _orthogonalize(W.data, cfg["ortho_beta"])

            d_running += float(loss_d)
            w_running += float(loss_w)

        if verbose:
            n = cfg["adv_steps"]
            print(f"  adv epoch {epoch}/{cfg['adv_epochs']}  "
                  f"D {d_running / n:.4f}  W {w_running / n:.4f}  "
                  f"{time.time() - t0:.0f}s", file=sys.stderr)

    return W.detach()


# --------------------------------------------------------------------------------------
# Anchors, refinement, candidates
# --------------------------------------------------------------------------------------

def mutual_anchors(scores: torch.Tensor) -> list[tuple[int, int]]:
    """Pairs that are each other's best match under CSLS.

    Paper condition (2) for an anchor. Mutuality is what makes these usable as
    pseudo-seeds: a one-sided nearest neighbour is far more often wrong.
    """
    best_fwd = scores.argmax(dim=1)
    best_bwd = scores.argmax(dim=0)
    return [(i, int(j)) for i, j in enumerate(best_fwd.tolist())
            if int(best_bwd[j]) == i]


def procrustes(Za_anch: torch.Tensor, Zu_anch: torch.Tensor) -> torch.Tensor:
    """W* = U V^T with U S V^T = SVD(Zu_anch^T Za_anch)  (paper equation 9).

    The orthogonal Procrustes solution to min ||W Za - Zu||_F. Constraining W
    to be orthogonal is what makes this a closed form rather than a second
    optimization.
    """
    U, _, Vt = torch.linalg.svd(Zu_anch.t() @ Za_anch)
    return U @ Vt


def candidate_sets(scores: torch.Tensor, top: int = TOP10) -> torch.Tensor:
    """Top-`top` CSLS matches in G^u for each row (paper section 3.1, end)."""
    return scores.topk(min(top, scores.shape[1]), dim=1).indices


def mean_match_cosine(src: torch.Tensor, tgt: torch.Tensor,
                      pairs: list[tuple[int, int]]) -> float:
    """Model-selection measure: mean cosine over the matched pairs.

    Adv-GD is seed-free, so there is no validation set. The paper selects
    hyperparameters by maximizing this instead (section 3.1).
    """
    if not pairs:
        return float("nan")
    i = torch.tensor([p[0] for p in pairs], device=src.device)
    j = torch.tensor([p[1] for p in pairs], device=tgt.device)
    return float((_unit(src)[i] * _unit(tgt)[j]).sum(1).mean())


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def top_degree(G: nx.Graph, nodes: list, k: int) -> np.ndarray:
    """Row indices of the k highest-degree nodes, ties broken by id for determinism."""
    deg = G.degree()
    order = sorted(range(len(nodes)), key=lambda i: (-deg[nodes[i]], nodes[i]))
    return np.array(order[:min(k, len(nodes))], dtype=np.int64)


def _refine(Za_t: torch.Tensor, Zu_t: torch.Tensor, anch_a, anch_u,
            W: torch.Tensor, cfg: dict, verbose: bool = True,
            on_round=None) -> tuple:
    """The refinement rounds: from a starting W to its final W, anchors, history.

    Each round matches the anchor pool by CSLS under the current W, keeps the
    mutual nearest neighbours, and refits W to them by Procrustes (paper
    equations 8-9). Separate from `_search` so that a W that did not come from
    adversarial training -- `align_from_seeds` starts from a Procrustes fit to
    given seed pairs -- is refined by exactly the code that refines the
    adversarial W, and not by a copy of it.

    `on_round(rnd, pairs)`, if given, is called after each round's matching
    with that round's mutual pairs as (row in anch_a, row in anch_u) indices.
    It only observes; the default None leaves the loop as it was.
    """
    history, pairs = [], []
    for rnd in range(cfg["refine_rounds"] + 1):
        s = csls_matrix(Za_t[anch_a] @ W.t(), Zu_t[anch_u], cfg["csls_k"])
        pairs = mutual_anchors(s)
        cos = mean_match_cosine(Za_t[anch_a] @ W.t(), Zu_t[anch_u], pairs)
        history.append({"round": rnd, "anchors": len(pairs), "mean_cosine": cos})
        if on_round is not None:
            on_round(rnd, pairs)
        if verbose:
            print(f"    refine {rnd}/{cfg['refine_rounds']}  "
                  f"anchors {len(pairs):,}  mean cosine {cos:.4f}",
                  file=sys.stderr)
        if rnd == cfg["refine_rounds"] or not pairs:
            break
        W = procrustes(Za_t[anch_a][[p[0] for p in pairs]],
                       Zu_t[anch_u][[p[1] for p in pairs]])
    return W, pairs, history


def _search(Za_t: torch.Tensor, Zu_t: torch.Tensor, adv_a, adv_u, anch_a,
            anch_u, cfg: dict, seed: int, verbose: bool = True,
            label: str = "") -> tuple:
    """One adversarial run plus its refinement rounds: W, its anchors, history.

    Separate from `align` because it has to be repeated. Adversarial training
    is bimodal here -- see `n_restarts` -- so a single run is a coin flip and
    the caller picks among several by mean cosine.
    """
    if verbose and label:
        print(f"  {label}  (seed {seed})", file=sys.stderr)
    if cfg.get("skip_adversarial"):
        # Ablation: isolates whether the adversarial step (equations 4-5) earns
        # its keep, or whether unsupervised CSLS + Procrustes bootstrapping from
        # a neutral starting point gets there on its own. W starts at the
        # identity rather than trained -- there is no other seed-free way to
        # propose a *first* W without training something, and identity is the
        # only choice that injects no information. If the two embedding spaces
        # happen to already be close to co-rotated (GAE has no reason to prefer
        # any particular rotation, so in general they will not be), the refine
        # loop below can still walk itself to a good W from there. What was
        # measured is in the Overleaf experiment logs (Log 1's ablation
        # section; Log 2), not in `adv_gd_protocol.md`.
        dim = Za_t.shape[1]
        W = torch.eye(dim, device=Za_t.device)
    else:
        W = train_mapping(Za_t[adv_a], Zu_t[adv_u], cfg, seed=seed, verbose=verbose)

    return _refine(Za_t, Zu_t, anch_a, anch_u, W, cfg, verbose)


def _prepare(Za: np.ndarray, Zu: np.ndarray, cfg: dict,
             device: str | None) -> tuple:
    """Both embedding matrices as float32 tensors on the device, preprocessed."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Za_t = _preprocess(torch.as_tensor(Za, dtype=torch.float32, device=dev),
                       cfg["normalize"])
    Zu_t = _preprocess(torch.as_tensor(Zu, dtype=torch.float32, device=dev),
                       cfg["normalize"])
    return Za_t, Zu_t


def _anchors_and_candidates(Za_t: torch.Tensor, Zu_t: torch.Tensor,
                            nodes_a: list, nodes_u: list, Ga: nx.Graph,
                            anch_a, anch_u, W: torch.Tensor, pairs: list,
                            m: int, cfg: dict) -> tuple:
    """Anchors as node ids, and the candidate sets, for a final W and its pairs.

    Shared by `align` and `align_from_seeds`, so that both hand `propagate`
    anchors and candidates built by one piece of code.
    """
    anchors = [(nodes_a[anch_a[i]], nodes_u[anch_u[j]]) for i, j in pairs]

    # Candidate sets: top 10 CSLS matches in all of G^u for each of the top M
    # degree nodes of G^a. M exceeds N_adv in Table 1, so this deliberately
    # covers nodes the adversarial phase never trained on.
    cand_a = top_degree(Ga, nodes_a, m)
    s_cand = csls_matrix(Za_t[cand_a] @ W.t(), Zu_t, cfg["csls_k"])
    top = candidate_sets(s_cand, TOP10).cpu().numpy()
    candidates = {nodes_a[cand_a[i]]: [nodes_u[j] for j in top[i]]
                  for i in range(len(cand_a))}
    return anchors, candidates


def align(Za: np.ndarray, nodes_a: list, Zu: np.ndarray, nodes_u: list,
          Ga: nx.Graph, Gu: nx.Graph, n_adv: int, n_anch: int, m: int,
          cfg: dict | None = None, seed: int = 0, device: str | None = None,
          verbose: bool = True) -> dict:
    """Run Phase 2 end to end and return W, anchors, candidate sets, and stats.

    Anchors and candidates are returned as node ids, not row indices, so the
    caller never has to know the embedding row order.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    if cfg["skip_adversarial"] and cfg["n_restarts"] > 1:
        # No training means no randomness: every restart would start from the
        # same identity W and refine identically. Restarts exist to pick a mode
        # out of training's bimodality (see `_search`'s docstring); there is
        # nothing to pick among here.
        cfg["n_restarts"] = 1
    Za_t, Zu_t = _prepare(Za, Zu, cfg, device)

    adv_a = top_degree(Ga, nodes_a, n_adv)
    adv_u = top_degree(Gu, nodes_u, n_adv)
    if verbose:
        print(f"adversarial training on top {len(adv_a):,} / {len(adv_u):,} "
              f"degree nodes", file=sys.stderr)

    # Anchors come from the top N_anch degree nodes of each graph (condition 1),
    # which for every dataset in Table 1 is a subset of the N_adv used above.
    anch_a = top_degree(Ga, nodes_a, n_anch)
    anch_u = top_degree(Gu, nodes_u, n_anch)

    runs = []
    for r in range(cfg["n_restarts"]):
        W, pairs, history = _search(Za_t, Zu_t, adv_a, adv_u, anch_a, anch_u,
                                    cfg, seed=seed + r, verbose=verbose,
                                    label=f"restart {r + 1}/{cfg['n_restarts']}")
        runs.append({"seed": seed + r, "W": W, "pairs": pairs,
                     "history": history,
                     "mean_cosine": history[-1]["mean_cosine"]})

    # Model selection, section 3.1: the mean cosine of the CSLS-matched pairs.
    # This is the only selection signal available -- Adv-GD is seed-free, so
    # there is no ground truth and no validation set to choose on. It is also
    # doing real work: adversarial training reaches a good alignment on roughly
    # half of all seeds and near-zero on the rest, and this criterion separates
    # the two cleanly. See the reproduction log in `adv_gd_protocol.md`.
    best = max(runs, key=lambda r: r["mean_cosine"])
    W, pairs = best["W"], best["pairs"]
    if verbose and cfg["n_restarts"] > 1:
        spread = ", ".join(f"{r['mean_cosine']:.4f}" for r in runs)
        print(f"  mean cosine across restarts: {spread}", file=sys.stderr)
        print(f"  selected seed {best['seed']} "
              f"({best['mean_cosine']:.4f}, {len(pairs):,} anchors)",
              file=sys.stderr)

    anchors, candidates = _anchors_and_candidates(
        Za_t, Zu_t, nodes_a, nodes_u, Ga, anch_a, anch_u, W, pairs, m, cfg)

    return {
        "W": W.cpu().numpy(),
        "anchors": anchors,
        "candidates": candidates,
        "history": best["history"],
        "selected_seed": best["seed"],
        "restarts": [{"seed": r["seed"], "anchors": len(r["pairs"]),
                      "mean_cosine": r["mean_cosine"]} for r in runs],
        "config": cfg,
    }


def align_from_seeds(Za: np.ndarray, nodes_a: list, Zu: np.ndarray,
                     nodes_u: list, Ga: nx.Graph, Gu: nx.Graph, n_anch: int,
                     m: int, seed_pairs: list, cfg: dict | None = None,
                     device: str | None = None, verbose: bool = True,
                     on_round=None) -> dict:
    """Phase 2 from a given seed set, in place of adversarial training.

    W_0 is the orthogonal Procrustes fit (equation 9) to `seed_pairs`, a list
    of (node id in G^a, node id in G^u). From there every step is the one
    `align` runs after its adversarial W: the CSLS / mutual-anchor / Procrustes
    refinement rounds (`_refine`), then the anchor ids and the top-10 candidate
    sets for the top M nodes (`_anchors_and_candidates`). It returns the dict
    `align` returns, so `propagate` and `evaluate_anchors` take it unchanged,
    plus `W0` (the fit before refinement) and `seed_pairs` (as used).

    Nothing is trained and nothing is random here, so there is one run and no
    selection among restarts: `selected_seed` is None and `restarts` holds the
    single run. `config` reports skip_adversarial and n_restarts=1, as `align`
    reports them for the identity-start ablation. N_adv plays no part and is
    not a parameter.

    This function does not know where the seed pairs came from, and the ones
    that name a node with no embedding row are an error, not skipped: a silent
    drop would change the fit without a trace. `on_round(rnd, pairs)`, if
    given, receives each refinement round's mutual anchors as node-id pairs
    (observation only, e.g. to score them against a ground truth the caller
    holds).
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    cfg["skip_adversarial"] = True
    cfg["n_restarts"] = 1
    Za_t, Zu_t = _prepare(Za, Zu, cfg, device)

    seeds = [(a, u) for a, u in seed_pairs]
    if not seeds:
        raise ValueError("align_from_seeds needs at least one seed pair; with "
                         "none, Procrustes returns the identity and this would "
                         "silently be the identity-start ablation")
    row_a = {n: i for i, n in enumerate(nodes_a)}
    row_u = {n: i for i, n in enumerate(nodes_u)}
    missing = [p for p in seeds if p[0] not in row_a or p[1] not in row_u]
    if missing:
        raise ValueError(f"{len(missing):,} of {len(seeds):,} seed pairs name a "
                         f"node with no embedding row, e.g. {missing[0]}")
    ia = np.array([row_a[a] for a, _ in seeds], dtype=np.int64)
    iu = np.array([row_u[u] for _, u in seeds], dtype=np.int64)
    W0 = procrustes(Za_t[ia], Zu_t[iu])
    if verbose:
        print(f"W_0: Procrustes fit to {len(seeds):,} seed pairs",
              file=sys.stderr)

    anch_a = top_degree(Ga, nodes_a, n_anch)
    anch_u = top_degree(Gu, nodes_u, n_anch)
    observe = None
    if on_round is not None:
        def observe(rnd, pairs):
            on_round(rnd, [(nodes_a[anch_a[i]], nodes_u[anch_u[j]])
                           for i, j in pairs])
    W, pairs, history = _refine(Za_t, Zu_t, anch_a, anch_u, W0, cfg, verbose,
                                on_round=observe)
    anchors, candidates = _anchors_and_candidates(
        Za_t, Zu_t, nodes_a, nodes_u, Ga, anch_a, anch_u, W, pairs, m, cfg)

    return {
        "W": W.cpu().numpy(),
        "anchors": anchors,
        "candidates": candidates,
        "history": history,
        "selected_seed": None,
        "restarts": [{"seed": None, "anchors": len(pairs),
                      "mean_cosine": history[-1]["mean_cosine"]}],
        "config": cfg,
        "W0": W0.cpu().numpy(),
        "seed_pairs": seeds,
    }


def evaluate_anchors(anchors: list[tuple[str, str]], truth: dict) -> dict:
    """Score anchors against ground truth, in the paper's Table 2 form.

    Reported as correct/found: the denominator is how many anchors the method
    proposed, not how many it could have.
    """
    correct = sum(1 for a, u in anchors if truth.get(a) == u)
    return {"found": len(anchors), "correct": correct,
            "precision": correct / len(anchors) if anchors else float("nan")}


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------

def self_test(verbose: bool = True) -> bool:
    """Checks on synthetic data where the correct alignment is known.

    A random orthogonal rotation of a point cloud is exactly the problem W is
    meant to undo, so Procrustes and CSLS must solve it near-perfectly. If they
    do not, nothing on real graphs will work.
    """
    ok = True
    rng = np.random.default_rng(0)
    n, d = 400, 16
    Z = rng.normal(size=(n, d)).astype(np.float32)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Zr = (Z @ Q).astype(np.float32)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.as_tensor(Z, device=dev)
    B = torch.as_tensor(Zr, device=dev)

    # Procrustes given the true correspondence must recover the rotation.
    W = procrustes(A, B)
    err = float((A @ W.t() - B).norm() / B.norm())
    if err > 1e-4:
        print(f"  [FAIL] Procrustes relative error {err:.2e}", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] Procrustes recovers the rotation (rel. error {err:.2e})",
              file=sys.stderr)

    # W is orthogonal, so W W^T must be the identity.
    off = float((W @ W.t() - torch.eye(d, device=dev)).abs().max())
    if off > 1e-4:
        print(f"  [FAIL] W not orthogonal, max |WW^T - I| = {off:.2e}", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] W is orthogonal (max |WW^T - I| = {off:.2e})", file=sys.stderr)

    # Under the recovered map, every point's mutual CSLS match is itself.
    s = csls_matrix(A @ W.t(), B, DEFAULTS["csls_k"])
    pairs = mutual_anchors(s)
    hits = sum(1 for i, j in pairs if i == j)
    if hits < 0.99 * n:
        print(f"  [FAIL] CSLS mutual matches {hits}/{n}", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] CSLS mutual nearest neighbours {hits}/{n} correct",
              file=sys.stderr)

    # Candidate sets must contain the true match.
    cands = candidate_sets(s, TOP10).cpu().numpy()
    in_top = sum(1 for i in range(n) if i in cands[i])
    if in_top < 0.99 * n:
        print(f"  [FAIL] true match in top-10 for only {in_top}/{n}", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] true match in top-10 for {in_top}/{n}", file=sys.stderr)

    # An unrelated space must NOT align -- guards against a scoring bug that
    # would make everything look matched.
    Zbad = rng.normal(size=(n, d)).astype(np.float32)
    s_bad = csls_matrix(A, torch.as_tensor(Zbad, device=dev), DEFAULTS["csls_k"])
    bad_hits = sum(1 for i, j in mutual_anchors(s_bad) if i == j)
    if bad_hits > 0.05 * n:
        print(f"  [FAIL] {bad_hits}/{n} spurious matches on unrelated data",
              file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] unrelated spaces do not align ({bad_hits}/{n} matches)",
              file=sys.stderr)

    # The ablation path: identity-started refinement must run to completion
    # and return a plain, scoreable result, even though it is not expected to
    # recover an arbitrary rotation. (That is a finding to report, not a bug.)
    cfg_noadv = {**DEFAULTS, "skip_adversarial": True, "refine_rounds": 5,
                 "csls_k": DEFAULTS["csls_k"]}
    W0, pairs0, hist0 = _search(A, B, np.arange(n), np.arange(n),
                                np.arange(n), np.arange(n), cfg_noadv,
                                seed=0, verbose=False)
    if not isinstance(W0, torch.Tensor) or W0.shape != (d, d):
        print("  [FAIL] skip_adversarial did not return a usable W",
              file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] skip_adversarial ran to completion "
              f"({len(pairs0)} anchors, identity-started)", file=sys.stderr)

    # The seeded entry point: a handful of correct pairs must fix the rotation
    # exactly, and the refinement and candidate steps must then run on it as
    # they do after adversarial training. Z and Zr = Z Q are related by the W
    # with Z W^T = Zr, that is W = Q^T.
    ids_a = [f"a{i:04d}" for i in range(n)]
    ids_u = [f"u{i:04d}" for i in range(n)]
    G_a = nx.relabel_nodes(nx.gnm_random_graph(n, 4 * n, seed=3),
                           dict(enumerate(ids_a)))
    G_u = nx.relabel_nodes(G_a, dict(zip(ids_a, ids_u)))
    seeds = list(zip(ids_a[:3 * d // 2], ids_u[:3 * d // 2]))
    res = align_from_seeds(Z, ids_a, Zr, ids_u, G_a, G_u, n_anch=n, m=n,
                           seed_pairs=seeds, device=dev.type, verbose=False)
    err = float(np.linalg.norm(res["W0"] - Q.T))
    hits = sum(1 for a, u in res["anchors"] if a[1:] == u[1:])
    ok_cand = sum(1 for a, c in res["candidates"].items()
                  if "u" + a[1:] in c)
    if err > 1e-4 or hits < 0.99 * n or ok_cand < 0.99 * n:
        print(f"  [FAIL] align_from_seeds: |W0 - Q^T|_F {err:.2e}, "
              f"{hits}/{n} anchors, {ok_cand}/{n} candidate sets",
              file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] align_from_seeds recovers the rotation from "
              f"{len(seeds)} pairs (|W0 - Q^T|_F {err:.2e}, {hits}/{n} anchors "
              f"correct, true match in the top-10 for {ok_cand}/{n})",
              file=sys.stderr)

    print("all checks passed" if ok else "CHECKS FAILED", file=sys.stderr)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Adv-GD Phase 2: adversarial alignment.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 0 if self_test() else 1
    ap.error("nothing to do; Phase 2 runs from the driver, or use --self-test")


if __name__ == "__main__":
    raise SystemExit(main())
