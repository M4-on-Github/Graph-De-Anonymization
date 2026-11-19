import argparse
from itertools import islice
from pathlib import Path
from random import Random
import sys
from time import perf_counter

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

if not __package__:
    sys.path.insert(0, str(REPO_ROOT))

from src.G2TA import extract_features as features
from src.G2TA.probabilistic_record_linkage import (
    cosine_similarity_matrix,
    resolve_links,
    score_candidate_pairs,
)


#AI-GENERATED-FUNCTION
def load_hepth_sample(path: Path, node_count: int) -> nx.DiGraph:
    """Sample a weakly connected neighborhood and retain citation direction."""
    graph = nx.read_edgelist(path, create_using=nx.DiGraph)
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    if not graph:
        raise ValueError("The input edge list has no nodes")
    start = max(graph, key=graph.degree.__getitem__)
    traversal = nx.bfs_edges(graph.to_undirected(as_view=True), start, sort_neighbors=sorted)
    nodes = [start, *(target for _, target in islice(traversal, node_count - 1))]
    return graph.subgraph(nodes).copy()


#AI-GENERATED-FUNCTION
def make_shuffled_graph(
    graph: nx.Graph | nx.DiGraph, rng: Random, edge_noise: float,
    mode: str = "individual",
) -> tuple[nx.Graph | nx.DiGraph, dict[str, str]]:
    """Shuffle row order and IDs, and optionally replace edges."""
    nodes = list(graph)
    shuffled_nodes = nodes.copy()
    rng.shuffle(shuffled_nodes)
    truth = {node: f"anon_{index}" for index, node in enumerate(shuffled_nodes)}
    shuffled = type(graph)()
    shuffled.add_nodes_from(truth.values())
    shuffled.add_edges_from((truth[source], truth[target]) for source, target in graph.edges)

    replacements = round(edge_noise * shuffled.number_of_edges())
    if mode == "density":
        dyads = sum(1 for a, b in shuffled.edges if a < b or not shuffled.has_edge(b, a))
        replacements = 2 * round(edge_noise * dyads)
    if not replacements:
        return shuffled, truth

    paired = mode == "reciprocal"
    unordered = paired or not shuffled.is_directed()
    if paired:
        groups = []
        for a, b in shuffled.edges:
            if not shuffled.has_edge(b, a):
                groups.append(((a, b),))
            elif a < b:
                groups.append(((a, b), (b, a)))
    else:
        groups = [((a, b),) for a, b in shuffled.edges]

    # Sample absent edges or dyads before deleting anything. This prevents re-adding
    # a deleted edge and keeps reciprocal groups intact at their new endpoints.
    nodes = list(shuffled)
    capacity = len(nodes) * (len(nodes) - 1) // (2 if unordered else 1)
    available = capacity - len(groups)
    rng.shuffle(groups)
    removed = []
    for group in groups:
        if len(removed) == available or replacements == 0:
            break
        if len(group) <= replacements:
            removed.append(group)
            replacements -= len(group)

    added = []
    seen = set()
    attempts = 0
    while len(added) < len(removed) and attempts < max(100, 20 * len(removed)):
        attempts += 1
        a, b = rng.sample(nodes, 2)
        key = tuple(sorted((a, b))) if unordered else (a, b)
        if key not in seen and not shuffled.has_edge(a, b) and (
            not unordered or not shuffled.has_edge(b, a)
        ):
            seen.add(key)
            added.append((a, b))

    if len(added) < len(removed):
        remaining = [
            (a, b) for a in nodes for b in nodes
            if a != b and (not unordered or a < b)
            and (a, b) not in seen and not shuffled.has_edge(a, b)
            and (not unordered or not shuffled.has_edge(b, a))
        ]
        samples = rng.sample(remaining, len(removed) - len(added))
        for a, b in samples:
            added.append((b, a) if paired and rng.randrange(2) else (a, b))

    for group, (a, b) in zip(removed, added):
        shuffled.remove_edges_from(group)
        shuffled.add_edge(a, b)
        if len(group) == 2:
            shuffled.add_edge(b, a)
    return shuffled, truth


#AI-GENERATED-FUNCTION
def build_feature_table(graph: nx.Graph | nx.DiGraph) -> tuple[list[str], np.ndarray]:
    """Return row IDs and baseline structural features in a fixed column order."""
    nodes = list(graph)
    columns = (
        features.extract_degree(graph),
        features.extract_clustering(graph),
        features.extract_PageRank(graph),
        features.extract_kcore(graph),
        nx.eigenvector_centrality(graph, max_iter=5000),
        features.extract_n_hop_degree_bin(graph, 1),
        features.extract_n_hop_degree_bin(graph, 2),
    )
    return nodes, np.column_stack([[column[node] for node in nodes] for column in columns])


#AI-GENERATED-FUNCTION
def main() -> None:
    parser = argparse.ArgumentParser(description="Run probabilistic record linkage on HepTH")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/cit-HepTH/cit-HepTh.txt")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=6767)
    parser.add_argument("--edge-noise", type=float, default=0.0)
    parser.add_argument("--graph", choices=("directed", "undirected"), default="directed")
    parser.add_argument("--noise-mode", choices=("individual", "reciprocal", "density"), default="individual")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--bandwidth", type=float, default=0.02)
    parser.add_argument("--min-gap", type=float, default=0.1)
    args = parser.parse_args()
    if args.nodes < 3 or not 0 < args.calibration_fraction < 1:
        parser.error("Use at least three nodes and a calibration fraction between zero and one")
    if not 0 <= args.edge_noise <= 1:
        parser.error("Edge noise must be between zero and one")
    if args.graph == "undirected" and args.noise_mode != "individual":
        parser.error("Reciprocal and density modes are for directed graphs")

    rng = Random(args.seed)
    started = perf_counter()
    original = load_hepth_sample(args.data, args.nodes)
    if len(original) < 3:
        parser.error("The sampled component must contain at least three nodes")
    if args.graph == "undirected":
        original = original.to_undirected()
    shuffled, truth = make_shuffled_graph(original, rng, args.edge_noise, args.noise_mode)
    replaced = sum(not shuffled.has_edge(truth[a], truth[b]) for a, b in original.edges)
    pair_ready = perf_counter()
    left_ids, left_table = build_feature_table(original)
    right_ids, right_table = build_feature_table(shuffled)
    features_ready = perf_counter()
    right_index = {node: index for index, node in enumerate(right_ids)}

    calibration_size = min(len(left_ids) - 1, max(2, round(len(left_ids) * args.calibration_fraction)))
    known_rows = Random(args.seed).sample(range(len(left_ids)), calibration_size)
    known_pairs = [(row, right_index[truth[left_ids[row]]]) for row in known_rows]
    # Use the same column scaling on both graphs, fitted on calibration rows.
    scale = left_table[known_rows].std(axis=0)
    scale[scale == 0] = 1.0
    similarities = cosine_similarity_matrix(left_table / scale, right_table / scale)
    similarities_ready = perf_counter()
    left, right, scores = score_candidate_pairs(similarities, known_pairs, args.bandwidth, args.seed)
    scores_ready = perf_counter()
    links = resolve_links(scores, args.min_gap)

    correct = sum(truth[left_ids[left[row]]] == right_ids[right[col]] for row, col in links)
    precision = correct / len(links) if links else 0.0
    recall = correct / len(left)
    f1 = 2 * precision * recall / (precision + recall) if correct else 0.0
    finished = perf_counter()
    print(f"Graph {args.graph}, noise mode {args.noise_mode}, seed {args.seed}")
    print(f"Nodes {len(original)}, edges {original.number_of_edges()}, calibration pairs {calibration_size}")
    print(f"Requested noise {args.edge_noise:g}, replaced edges {replaced}, graph density {nx.density(original):.6f}")
    print(f"Held-out nodes {len(left)}, predicted links {len(links)}, correct links {correct}")
    print(f"Precision {precision:.3f}, recall {recall:.3f}, F1 {f1:.3f}, coverage {len(links) / len(left):.3f}")
    # AI-generated stage timings use wall-clock seconds.
    print(f"Cosine pairs {similarities.size:,}, held-out candidate pairs {scores.size:,}")
    print(f"Load and prepare graph pair {pair_ready - started:.3f}s")
    print(f"Extract features {features_ready - pair_ready:.3f}s")
    print(f"Scale features and compute cosine scores {similarities_ready - features_ready:.3f}s")
    print(f"Calibrate and score candidates {scores_ready - similarities_ready:.3f}s")
    print(f"Resolve and evaluate links {finished - scores_ready:.3f}s")
    print(f"Total {finished - started:.3f}s")


if __name__ == "__main__":
    main()
