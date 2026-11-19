import numpy as np
from scipy.special import logsumexp


#AI-GENERATED-FUNCTION
def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compare rows with matching feature columns. Zero rows have similarity zero."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Both inputs must be matrices with the same feature columns")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Feature values must be finite")

    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_unit = np.divide(left, left_norm, out=np.zeros_like(left), where=left_norm != 0)
    right_unit = np.divide(right, right_norm, out=np.zeros_like(right), where=right_norm != 0)
    return np.clip(left_unit @ right_unit.T, -1.0, 1.0)


#AI-GENERATED-FUNCTION
def _log_gaussian_density(
    scores: np.ndarray, samples: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Evaluate a fixed-bandwidth Gaussian KDE in log space, in batches."""
    flat_scores = scores.ravel()
    density = np.empty_like(flat_scores, dtype=float)
    normalizer = np.log(len(samples)) + np.log(bandwidth * np.sqrt(2 * np.pi))
    for start in range(0, len(flat_scores), 2048):
        batch = slice(start, start + 2048)
        residuals = (flat_scores[batch, None] - samples[None, :]) / bandwidth
        density[batch] = logsumexp(-0.5 * residuals ** 2, axis=1) - normalizer
    return density.reshape(scores.shape)


#AI-GENERATED-FUNCTION
def score_candidate_pairs(
    similarities: np.ndarray,
    calibration_pairs: list[tuple[int, int]],
    bandwidth: float = 0.02,
    seed: int = 6767,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return held-out row indices, column indices, and log likelihood ratios."""
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite")
    if len(calibration_pairs) < 2:
        raise ValueError("At least two calibration pairs are needed")

    left_known, right_known = map(np.asarray, zip(*calibration_pairs))
    if len(set(left_known)) != len(left_known) or len(set(right_known)) != len(right_known):
        raise ValueError("Calibration pairs must be one-to-one")
    calibration_scores = similarities[np.ix_(left_known, right_known)]
    match_samples = calibration_scores.diagonal()
    nonmatch_samples = calibration_scores[~np.eye(len(left_known), dtype=bool)]
    if len(nonmatch_samples) > 512:
        rng = np.random.default_rng(seed)
        nonmatch_samples = rng.choice(nonmatch_samples, 512, replace=False)

    left = np.setdiff1d(np.arange(similarities.shape[0]), left_known)
    right = np.setdiff1d(np.arange(similarities.shape[1]), right_known)
    candidates = similarities[np.ix_(left, right)]
    scores = (
        _log_gaussian_density(candidates, match_samples, bandwidth)
        - _log_gaussian_density(candidates, nonmatch_samples, bandwidth)
    )
    return left, right, scores


#AI-GENERATED-FUNCTION
def resolve_links(scores: np.ndarray, min_gap: float = 0.1) -> list[tuple[int, int]]:
    """Accept positive, unambiguous row winners in descending confidence-gap order."""
    if not np.isfinite(min_gap) or min_gap < 0:
        raise ValueError("min_gap must be nonnegative and finite")
    if not scores.size:
        return []

    winners = scores.argmax(axis=1)
    best = scores[np.arange(len(scores)), winners]
    if scores.shape[1] > 1:
        gaps = best - np.partition(scores, -2, axis=1)[:, -2]
    else:
        gaps = np.full(len(scores), np.inf)

    links = []
    used_columns = set()
    for row in np.argsort(-gaps, kind="stable"):
        column = int(winners[row])
        if gaps[row] > 0 and gaps[row] >= min_gap and best[row] > 0 and column not in used_columns:
            links.append((int(row), column))
            used_columns.add(column)
    return links
