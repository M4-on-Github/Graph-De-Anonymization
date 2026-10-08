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
    W = train_mapping(Za_t[adv_a], Zu_t[adv_u], cfg, seed=seed, verbose=verbose)

    history, pairs = [], []
    for rnd in range(cfg["refine_rounds"] + 1):
        s = csls_matrix(Za_t[anch_a] @ W.t(), Zu_t[anch_u], cfg["csls_k"])
        pairs = mutual_anchors(s)
        cos = mean_match_cosine(Za_t[anch_a] @ W.t(), Zu_t[anch_u], pairs)
        history.append({"round": rnd, "anchors": len(pairs), "mean_cosine": cos})
        if verbose:
            print(f"    refine {rnd}/{cfg['refine_rounds']}  "
                  f"anchors {len(pairs):,}  mean cosine {cos:.4f}",
                  file=sys.stderr)
        if rnd == cfg["refine_rounds"] or not pairs:
            break
        W = procrustes(Za_t[anch_a][[p[0] for p in pairs]],
                       Zu_t[anch_u][[p[1] for p in pairs]])
    return W, pairs, history


def align(Za: np.ndarray, nodes_a: list, Zu: np.ndarray, nodes_u: list,
          Ga: nx.Graph, Gu: nx.Graph, n_adv: int, n_anch: int, m: int,
          cfg: dict | None = None, seed: int = 0, device: str | None = None,
          verbose: bool = True) -> dict:
    """Run Phase 2 end to end and return W, anchors, candidate sets, and stats.

    Anchors and candidates are returned as node ids, not row indices, so the
    caller never has to know the embedding row order.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Za_t = _preprocess(torch.as_tensor(Za, dtype=torch.float32, device=dev),
                       cfg["normalize"])
    Zu_t = _preprocess(torch.as_tensor(Zu, dtype=torch.float32, device=dev),
                       cfg["normalize"])

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

    anchors = [(nodes_a[anch_a[i]], nodes_u[anch_u[j]]) for i, j in pairs]

    # Candidate sets: top 10 CSLS matches in all of G^u for each of the top M
    # degree nodes of G^a. M exceeds N_adv in Table 1, so this deliberately
    # covers nodes the adversarial phase never trained on.
    cand_a = top_degree(Ga, nodes_a, m)
    s_cand = csls_matrix(Za_t[cand_a] @ W.t(), Zu_t, cfg["csls_k"])
    top = candidate_sets(s_cand, TOP10).cpu().numpy()
    candidates = {nodes_a[cand_a[i]]: [nodes_u[j] for j in top[i]]
                  for i in range(len(cand_a))}

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
