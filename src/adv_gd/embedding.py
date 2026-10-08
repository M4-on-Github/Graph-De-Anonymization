"""Phase 1 of Adv-GD: graph autoencoder embeddings.

Learns a d-dimensional representation of each node from topology alone, by
training a two-layer GCN encoder to reconstruct the adjacency matrix.

Paper provenance (docs/SeedFree_Li.etal.pdf, section 2.2, equations 1-3):
    encoder   GCN(X, A) = A_hat ReLU(A_hat X W0) W1
    decoder   A_recon   = sigmoid(Z Z^T)
    X is the identity -- no node features are used
    A has its diagonal set to 1 before normalization

Two equations in the paper cannot be implemented as printed; see
`adv_gd_protocol.md` for the full argument. In short:

  * Equation 1 writes the normalized adjacency as D^-1/2 A D, which is not
    symmetric. We use the standard D^-1/2 A D^-1/2.
  * Equation 3 writes the loss with the positive term only, which is degenerate
    -- driving every reconstructed entry to 1 sends it to zero. We use full
    binary cross-entropy with the positive class reweighted, as GAE does.

Unstated by the paper and chosen here from Kipf and Welling's GAE defaults:
hidden width, epoch count, and optimizer. All are CLI flags.

Library use:
    from embedding import gae_embed
    Z, nodes = gae_embed(G, dim=32, seed=0)

CLI:
    python src/adv_gd/embedding.py --self-test
    python src/adv_gd/embedding.py --graph runs/cit-HepTh_b10_s0/G1.edgelist --dim 32
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

REPO_ROOT = Path(__file__).resolve().parents[2]

# Rows of the decoder computed at once. The full Z Z^T for HepTh is 27,769^2
# floats (3.1 GB) before gradients, which does not fit in 8.5 GB of VRAM. The
# loss is a sum over rows, so chunking changes nothing but the peak memory.
DEFAULT_CHUNK = 2048


# --------------------------------------------------------------------------------------
# Graph -> tensors
# --------------------------------------------------------------------------------------

def normalized_adjacency(G: nx.Graph, nodes: list, device) -> torch.Tensor:
    """Build A_hat = D^-1/2 (A + I) D^-1/2 as a sparse tensor.

    The paper sets the diagonal of A to 1 ("we assume each node is connected to
    itself") and only then forms D, so the self-loop is included in the degree.
    """
    index = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    eu = [index[u] for u, v in G.edges()]
    ev = [index[v] for u, v in G.edges()]
    rows = eu + ev + list(range(n))     # both directions, then the diagonal
    cols = ev + eu + list(range(n))

    rows_t = torch.tensor(rows, dtype=torch.long)
    cols_t = torch.tensor(cols, dtype=torch.long)
    vals = torch.ones(len(rows), dtype=torch.float32)

    deg = torch.zeros(n, dtype=torch.float32).scatter_add_(0, rows_t, vals)
    dinv = deg.pow(-0.5)
    vals = dinv[rows_t] * vals * dinv[cols_t]

    return torch.sparse_coo_tensor(
        torch.stack([rows_t, cols_t]), vals, (n, n)
    ).coalesce().to(device)


def target_block_indices(G: nx.Graph, nodes: list, index: dict,
                         blocks: list, device) -> list:
    """Per-row-block coordinates of the 1-entries of (A + I), computed once.

    The reconstruction target is the same every epoch, so walking the graph in
    Python each time dominates the runtime. Here the coordinates are built once
    and each epoch only scatters them into a zero block on the device.
    """
    n = len(nodes)
    eu = np.fromiter((index[u] for u, v in G.edges()), dtype=np.int64,
                     count=G.number_of_edges())
    ev = np.fromiter((index[v] for u, v in G.edges()), dtype=np.int64,
                     count=G.number_of_edges())
    diag = np.arange(n, dtype=np.int64)
    rows = np.concatenate([eu, ev, diag])
    cols = np.concatenate([ev, eu, diag])

    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]

    out = []
    for lo, hi in blocks:
        a, b = np.searchsorted(rows, [lo, hi])
        out.append((
            torch.from_numpy(rows[a:b] - lo).to(device),
            torch.from_numpy(cols[a:b]).to(device),
        ))
    return out


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

class GAEEncoder(nn.Module):
    """Two-layer GCN with X = I, so the first layer is a plain weight lookup.

    With an identity feature matrix, A_hat X W0 reduces to A_hat W0, which saves
    materializing an n x n identity. W0 is therefore n x hidden.
    """

    def __init__(self, n: int, hidden: int, dim: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.empty(n, hidden))
        self.w1 = nn.Parameter(torch.empty(hidden, dim))
        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w1)

    def forward(self, a_hat: torch.Tensor) -> torch.Tensor:
        h = torch.relu(torch.sparse.mm(a_hat, self.w0))
        return torch.sparse.mm(a_hat, h @ self.w1)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def gae_embed(G: nx.Graph, dim: int = 32, hidden: int = 64, epochs: int = 200,
              lr: float = 0.01, seed: int = 0, chunk: int = DEFAULT_CHUNK,
              device: str | None = None, verbose: bool = True
              ) -> tuple[np.ndarray, list]:
    """Train a GAE on G and return (Z, nodes) with Z[i] the embedding of nodes[i].

    Node order is sorted so that two calls on the same graph line up, and so the
    caller can map embeddings back to ids without a side channel.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    nodes = sorted(G.nodes())
    index = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)

    a_hat = normalized_adjacency(G, nodes, dev)
    model = GAEEncoder(n, hidden, dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Class reweighting, as in GAE: positives are rare, so without this the model
    # predicts all-zero. n_pos counts the unit diagonal too, matching the target.
    n_pos = 2 * G.number_of_edges() + n
    n_tot = n * n
    pos_weight = torch.tensor((n_tot - n_pos) / n_pos, device=dev)
    norm = n_tot / (2.0 * (n_tot - n_pos))
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")

    blocks = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    coords = target_block_indices(G, nodes, index, blocks, dev)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        Z = model(a_hat)
        total = 0.0
        for k, (lo, hi) in enumerate(blocks):
            r, c = coords[k]
            target = torch.zeros(hi - lo, n, device=dev)
            target[r, c] = 1.0
            logits = Z[lo:hi] @ Z.t()
            loss = norm * lossfn(logits, target) / n_tot
            # Z is shared by every block, so its graph must survive until the last.
            loss.backward(retain_graph=(k < len(blocks) - 1))
            total += float(loss.detach())
            del logits, target
        opt.step()
        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}/{epochs}  loss {total:.4f}  "
                  f"{time.time() - t0:.0f}s", file=sys.stderr)

    with torch.no_grad():
        Z = model(a_hat).cpu().numpy()
    if verbose:
        print(f"embedded {n:,} nodes -> {dim}d in {time.time() - t0:.0f}s on {dev}",
              file=sys.stderr)
    return Z, nodes


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------

def reconstruction_auc(G: nx.Graph, Z: np.ndarray, nodes: list,
                       n_sample: int = 20000, seed: int = 0) -> float:
    """AUC of z_i . z_j at separating real edges from sampled non-edges.

    This is the only honest check on an unsupervised embedding: if the decoder
    cannot tell an edge from a non-edge, the embedding carries no topology and
    everything downstream is noise. Chance is 0.5.
    """
    rng = np.random.default_rng(seed)
    index = {u: i for i, u in enumerate(nodes)}
    edges = list(G.edges())
    pick = rng.choice(len(edges), size=min(n_sample, len(edges)), replace=False)
    pos = np.array([[index[edges[i][0]], index[edges[i][1]]] for i in pick])

    neg = []
    while len(neg) < len(pos):
        i, j = rng.integers(0, len(nodes), 2)
        if i != j and not G.has_edge(nodes[i], nodes[j]):
            neg.append([i, j])
    neg = np.array(neg)

    def score(pairs):
        return np.sum(Z[pairs[:, 0]] * Z[pairs[:, 1]], axis=1)

    sp, sn = score(pos), score(neg)
    # AUC by rank, which avoids a scipy or sklearn dependency.
    allv = np.concatenate([sp, sn])
    ranks = allv.argsort().argsort().astype(float) + 1
    r_pos = ranks[:len(sp)].sum()
    return float((r_pos - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn)))


def self_test(verbose: bool = True) -> bool:
    """Train on a small synthetic graph and assert the embedding is informative."""
    ok = True
    G = nx.gnm_random_graph(1500, 12000, seed=7)
    print(f"self-test on gnm_random_graph(1500, 12000): "
          f"{G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges",
          file=sys.stderr)

    Z, nodes = gae_embed(G, dim=16, hidden=32, epochs=120, seed=0, verbose=verbose)

    if Z.shape != (1500, 16):
        print(f"  [FAIL] shape {Z.shape}, expected (1500, 16)", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] embedding shape {Z.shape}", file=sys.stderr)

    if not np.isfinite(Z).all():
        print("  [FAIL] embedding contains nan or inf", file=sys.stderr)
        ok = False
    else:
        print("  [PASS] all finite", file=sys.stderr)

    auc = reconstruction_auc(G, Z, nodes, seed=0)
    if auc < 0.80:
        print(f"  [FAIL] reconstruction AUC {auc:.4f} -- embedding is not "
              f"capturing topology", file=sys.stderr)
        ok = False
    else:
        print(f"  [PASS] reconstruction AUC {auc:.4f} (chance 0.5)", file=sys.stderr)

    # Determinism: same seed must give the same embedding, or nothing downstream
    # is comparable between runs.
    Z2, _ = gae_embed(G, dim=16, hidden=32, epochs=120, seed=0, verbose=False)
    if not np.allclose(Z, Z2, atol=1e-5):
        print(f"  [FAIL] not reproducible at a fixed seed "
              f"(max delta {np.abs(Z - Z2).max():.2e})", file=sys.stderr)
        ok = False
    else:
        print("  [PASS] reproducible at a fixed seed", file=sys.stderr)

    print("all checks passed" if ok else "CHECKS FAILED", file=sys.stderr)
    return ok


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Adv-GD Phase 1: GAE node embeddings.")
    ap.add_argument("--graph", type=str, help="Edge list to embed.")
    ap.add_argument("--dim", type=int, default=32, help="Embedding dimension D.")
    ap.add_argument("--hidden", type=int, default=64, help="GCN hidden width.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu.")
    ap.add_argument("--out", type=str, default=None, help="Write Z and ids to this .npz.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1
    if not args.graph:
        ap.error("--graph is required unless --self-test is given")

    G = nx.read_edgelist(args.graph, nodetype=str)
    print(f"loaded {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges",
          file=sys.stderr)

    Z, nodes = gae_embed(G, dim=args.dim, hidden=args.hidden, epochs=args.epochs,
                         lr=args.lr, seed=args.seed, chunk=args.chunk,
                         device=args.device)
    print(f"reconstruction AUC {reconstruction_auc(G, Z, nodes):.4f}", file=sys.stderr)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, Z=Z, nodes=np.array(nodes, dtype=object))
        print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
