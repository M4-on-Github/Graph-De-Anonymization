import networkx as nx
from collections import Counter
from random import Random
from math import log2, sqrt
from statistics import fmean, pstdev

def extract_degree(graph: nx.Graph | nx.DiGraph) -> dict[str, int]:
    """Return each node's degree, summing in- and out-degree for directed graphs."""
    return dict(graph.degree)

def extract_clustering(graph: nx.Graph | nx.DiGraph) -> dict[str, float]:
    """Return local clustering using NetworkX's graph-specific coefficient."""
    return nx.clustering(graph)

def extract_PageRank(graph: nx.Graph | nx.DiGraph) -> dict[str, float]:
    """Return PageRank, following outgoing edges in directed graphs."""
    return nx.pagerank(graph)

def extract_kcore(graph: nx.Graph | nx.DiGraph) -> dict[str, int]:
    """Return core numbers using total degree for directed graphs without self-loops."""
    return nx.core_number(graph)

def extract_eigenvector(graph: nx.Graph | nx.DiGraph) -> dict[str, float]:
    """Return eigenvector centrality, using incoming edges in directed graphs."""
    return nx.eigenvector_centrality(graph)

def degree_bin(degree: int) -> int:
    return degree.bit_length()

def extract_n_hop_degree_bin(graph: nx.Graph | nx.DiGraph, n_hop: int) -> dict[str, int]:
    """Count same-bin nodes exactly n_hop edges away, following outgoing edges if directed."""
    if n_hop < 1:
        raise ValueError("n_hop must be at least 1")

    bins = {node: degree_bin(degree) for node, degree in extract_degree(graph).items()}
    counts = {}
    for node in graph:
        distances = nx.single_source_shortest_path_length(graph, node, cutoff=n_hop)
        counts[node] = sum(distance == n_hop and bins[other] == bins[node]for other, distance in distances.items())
    return counts

def extract_degree_n_grams(graph: nx.Graph | nx.DiGraph, walks_per_node: int, walk_steps: int, seed: int = 6767,
                           *, sequence_lengths: dict[str, int] | None = None
                           )-> dict[str,Counter[tuple[int,int,int]]]:
    """Count degree trigrams from random walks starting at each node."""
    degree = extract_degree(graph)
    # For a directed graph, graph[node] contains only outgoing neighbors.
    neighbors = {node: tuple(graph[node]) for node in graph}
    rng = Random(seed)
    n_grams_by_node = dict()
    for start in graph:
        counts = Counter()
        total_labels = 0
        for _ in range(walks_per_node):
            current = start
            # Include the starting node's degree before taking any steps.
            labels = [degree[current]]
            for _ in range(walk_steps):
                if not neighbors[current]:
                    break
                current = rng.choice(neighbors[current])
                labels.append(degree[current])
            # Slide a window of three over this walk. Short walks yield no windows.
            counts.update(zip(labels,labels[1:],labels[2:]))
            # AI-generated addition: pool walk lengths for approximate edit distance.
            total_labels += len(labels)
        n_grams_by_node[start] = counts
        if sequence_lengths is not None:
            sequence_lengths[start] = total_labels
    return n_grams_by_node

def extract_n_gram_entropy(n_grams_by_node: dict[str,Counter[tuple[int,int,int]]]
                           )-> dict[str,float|None]:
    """Return Shannon entropy in bits per node, or None if no trigrams were observed."""
    entropy_by_node: dict[str, float | None] = {}
    for node, counts in n_grams_by_node.items():
        total = sum(counts.values())
        if total==0:
            entropy_by_node[node]=None
            continue
        entropy = 0.0
        for count in counts.values():
            # Shannon entropy: Convert each trigram count to a probability and accumulate -p * log2(p).
            probability = count / total
            entropy -= probability * log2(probability)
        entropy_by_node[node] = entropy
    return entropy_by_node


#AI-GENERATED-FUNCTION
def extract_nk_series(
    graph: nx.Graph | nx.DiGraph,
) -> dict[str, tuple[int, Counter[int], Counter[int]]]:
    """Return nK0 degree and the raw-degree histograms nK1 and nK2."""
    degrees = extract_degree(graph)
    result = {}
    for node in graph:
        one_hop = Counter()
        two_hop = Counter()
        distances = nx.single_source_shortest_path_length(graph, node, cutoff=2)
        for other, distance in distances.items():
            if distance == 1:
                one_hop[degrees[other]] += 1
            elif distance == 2:
                two_hop[degrees[other]] += 1
        result[node] = degrees[node], one_hop, two_hop
    return result


#AI-GENERATED-FUNCTION
def extract_role_similarity(
    graph: nx.Graph | nx.DiGraph,
    n_grams_by_node: dict[str, Counter[tuple[int, int, int]]],
    n_hop: int,
) -> dict[str, float | None]:
    """Average Jaccard similarities to neighbors at exactly n_hop."""
    if n_hop < 1:
        raise ValueError("n_hop must be at least 1")

    grams = {node: set(counts) for node, counts in n_grams_by_node.items()}
    result = {}
    for node in graph:
        source = grams[node]
        similarities = []
        if source:
            distances = nx.single_source_shortest_path_length(graph, node, cutoff=n_hop)
            for other, distance in distances.items():
                target = grams[other]
                if distance == n_hop and target:
                    similarities.append(len(source & target) / len(source | target))
        result[node] = fmean(similarities) if similarities else None
    return result


#AI-GENERATED-FUNCTION
def extract_role_similarity_combined(
    one_hop: dict[str, float | None],
    two_hop: dict[str, float | None],
    one_hop_weight: float = 0.5,
) -> dict[str, float | None]:
    """Blend hop scores and use the available score when one hop is missing."""
    if not 0 <= one_hop_weight <= 1:
        raise ValueError("one_hop_weight must be between 0 and 1")

    combined = {}
    for node, first in one_hop.items():
        second = two_hop[node]
        if first is None:
            combined[node] = second
        elif second is None:
            combined[node] = first
        else:
            combined[node] = one_hop_weight * first + (1 - one_hop_weight) * second
    return combined


#AI-GENERATED-FUNCTION
def extract_role_combined_zscore(
    combined: dict[str, float | None],
) -> dict[str, float | None]:
    """Standardize observed scores using the graph-wide population variance."""
    observed = [score for score in combined.values() if score is not None]
    if not observed:
        return dict(combined)

    mean = fmean(observed)
    deviation = pstdev(observed)
    result = {}
    for node, score in combined.items():
        if score is None:
            result[node] = None
        else:
            result[node] = (score - mean) / deviation if deviation else 0.0
    return result


#AI-GENERATED-FUNCTION
def extract_structurally_similar_neighbors(
    graph: nx.Graph | nx.DiGraph,
    n_grams_by_node: dict[str, Counter[tuple[int, int, int]]],
    sequence_lengths: dict[str, int],
    max_edit_distance: float,
) -> dict[str, int | None]:
    """Count distinct neighbors within two hops meeting the approximate NED cutoff."""
    if not 0 <= max_edit_distance <= 1:
        raise ValueError("max_edit_distance must be between 0 and 1")

    grams = {node: set(counts) for node, counts in n_grams_by_node.items()}
    result = {}
    for node in graph:
        source = grams[node]
        if not source:
            result[node] = None
            continue

        count = 0
        distances = nx.single_source_shortest_path_length(graph, node, cutoff=2)
        for other, distance in distances.items():
            target = grams[other]
            if distance == 0 or not target:
                continue

            jaccard_distance = 1 - len(source & target) / len(source | target)
            first, second = sequence_lengths[node], sequence_lengths[other]
            if min(first, second) <= 0:
                raise ValueError("sequence lengths must be positive for nonempty trigram sets")
            alpha = min(first, second) / max(first, second)
            edit_distance = (1 + alpha * (jaccard_distance - 1)) / (2 - jaccard_distance)
            if edit_distance <= max_edit_distance:
                count += 1
        result[node] = count
    return result


#AI-GENERATED-FUNCTION
def extract_attribute_similarity(
    graph: nx.Graph | nx.DiGraph,
    attributes: dict[str, list[float] | tuple[float, ...]],
    n_hop: int,
    threshold: float = 0.9,
) -> dict[str, tuple[float | None, int | None]]:
    """Return mean cosine and count above threshold at the given hop distance."""
    if n_hop < 1:
        raise ValueError("n_hop must be at least 1")
    if len({len(vector) for vector in attributes.values()}) > 1:
        raise ValueError("attribute vectors must have the same dimension")

    normalized = {}
    for node, vector in attributes.items():
        norm = sqrt(sum(value * value for value in vector))
        if norm:
            normalized[node] = tuple(value / norm for value in vector)

    result = {}
    for node in graph:
        source = normalized.get(node)
        if source is None:
            result[node] = None, None
            continue

        similarities = []
        distances = nx.single_source_shortest_path_length(graph, node, cutoff=n_hop)
        for other, distance in distances.items():
            if distance == n_hop and other in normalized:
                cosine = sum(a * b for a, b in zip(source, normalized[other]))
                similarities.append(max(-1.0, min(1.0, cosine)))
        mean = fmean(similarities) if similarities else None
        result[node] = mean, sum(score > threshold for score in similarities)
    return result


#AI-GENERATED-FUNCTION
def extract_community(
    graph: nx.Graph | nx.DiGraph, seed: int = 6767
) -> dict[str, int]:
    """Assign a community ID using Louvain modularity on the input graph."""
    if graph.number_of_edges() == 0:
        return {node: community_id for community_id, node in enumerate(graph)}

    communities = nx.community.louvain_communities(graph, weight=None, seed=seed)
    return {
        node: community_id
        for community_id, members in enumerate(communities)
        for node in members
    }
