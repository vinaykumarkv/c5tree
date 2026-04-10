"""
Splitting criteria for C5.0.

C5.0 uses *gain ratio* (information gain / split information) rather than
raw information gain, which corrects the bias of ID3/C4.5 toward features
with many distinct values.

Missing values are handled via fractional instance weighting: an instance
with a missing feature value is split fractionally across all branches,
weighted by the proportion of known instances going to each branch.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy from raw counts (handles zeros safely)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _weighted_entropy(left_counts: np.ndarray, right_counts: np.ndarray) -> float:
    """Weighted average entropy of two branches."""
    n_left = left_counts.sum()
    n_right = right_counts.sum()
    total = n_left + n_right
    if total == 0:
        return 0.0
    return (n_left / total) * _entropy(left_counts) + \
           (n_right / total) * _entropy(right_counts)


# ---------------------------------------------------------------------------
# Gain ratio for a continuous feature
# ---------------------------------------------------------------------------

def best_continuous_split(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_classes: int,
) -> Tuple[Optional[float], float]:
    """
    Find the best threshold for a continuous feature using gain ratio.

    Parameters
    ----------
    x        : 1-D array of feature values (may contain NaN for missing).
    y        : 1-D array of integer class labels.
    weights  : 1-D array of instance weights (fractional for missing handling).
    n_classes: Number of target classes.

    Returns
    -------
    best_threshold : float or None
    best_gain_ratio: float  (0.0 if no valid split found)
    """
    mask = ~np.isnan(x)
    x_obs, y_obs, w_obs = x[mask], y[mask], weights[mask]

    if x_obs.size < 2:
        return None, 0.0

    # Total weighted entropy before split
    total_counts = np.bincount(y_obs, weights=w_obs, minlength=n_classes).astype(float)
    total_w = w_obs.sum()
    base_entropy = _entropy(total_counts)

    # Sort by feature value
    order = np.argsort(x_obs)
    x_sorted, y_sorted, w_sorted = x_obs[order], y_obs[order], w_obs[order]

    # Candidate thresholds: midpoints between consecutive distinct values
    thresholds = []
    for i in range(len(x_sorted) - 1):
        if x_sorted[i] < x_sorted[i + 1]:
            thresholds.append((x_sorted[i] + x_sorted[i + 1]) / 2.0)

    if not thresholds:
        return None, 0.0

    best_gain_ratio = -np.inf
    best_threshold = None

    # Incremental left/right counts
    left_counts = np.zeros(n_classes)
    right_counts = total_counts.copy()

    ptr = 0  # pointer into sorted arrays

    for threshold in thresholds:
        # Advance pointer past all values <= threshold
        while ptr < len(x_sorted) and x_sorted[ptr] <= threshold:
            left_counts[y_sorted[ptr]] += w_sorted[ptr]
            right_counts[y_sorted[ptr]] -= w_sorted[ptr]
            ptr += 1

        n_left = left_counts.sum()
        n_right = right_counts.sum()
        if n_left == 0 or n_right == 0:
            continue

        # Information gain
        after_entropy = _weighted_entropy(left_counts, right_counts)
        info_gain = base_entropy - after_entropy

        # Split information (penalises very uneven splits)
        p_left = n_left / total_w
        p_right = n_right / total_w
        split_info = -p_left * np.log2(p_left) - p_right * np.log2(p_right)

        if split_info == 0:
            continue

        gain_ratio = info_gain / split_info

        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_threshold = threshold

    return best_threshold, max(best_gain_ratio, 0.0)


# ---------------------------------------------------------------------------
# Gain ratio for a categorical feature
# ---------------------------------------------------------------------------

def best_categorical_split(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_classes: int,
) -> Tuple[List, float]:
    """
    Evaluate gain ratio for a multi-way categorical split.

    Parameters
    ----------
    x        : 1-D array of categorical values (may contain NaN).
    y        : 1-D array of integer class labels.
    weights  : 1-D array of instance weights.
    n_classes: Number of target classes.

    Returns
    -------
    categories   : list of unique categories seen (determines branches).
    gain_ratio   : float
    """
    mask = ~_is_missing_categorical(x)
    x_obs, y_obs, w_obs = x[mask], y[mask], weights[mask]

    if x_obs.size == 0:
        return [], 0.0

    categories = np.unique(x_obs).tolist()
    if len(categories) <= 1:
        return categories, 0.0

    total_counts = np.bincount(y_obs.astype(int), weights=w_obs, minlength=n_classes).astype(float)
    base_entropy = _entropy(total_counts)
    total_w = w_obs.sum()

    # Per-category entropy
    weighted_child_entropy = 0.0
    split_info = 0.0

    for cat in categories:
        cat_mask = x_obs == cat
        cat_y = y_obs[cat_mask]
        cat_w = w_obs[cat_mask]
        cat_w_sum = cat_w.sum()
        if cat_w_sum == 0:
            continue
        cat_counts = np.bincount(cat_y.astype(int), weights=cat_w, minlength=n_classes).astype(float)
        p = cat_w_sum / total_w
        weighted_child_entropy += p * _entropy(cat_counts)
        split_info -= p * np.log2(p)

    info_gain = base_entropy - weighted_child_entropy

    if split_info == 0:
        return categories, 0.0

    gain_ratio = max(info_gain / split_info, 0.0)
    return categories, gain_ratio


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _is_missing_categorical(x: np.ndarray) -> np.ndarray:
    """Return boolean mask of missing values for object/categorical arrays."""
    try:
        return np.isnan(x.astype(float))
    except (ValueError, TypeError):
        return np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in x])
