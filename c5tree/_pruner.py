"""
Pessimistic Error Pruning (PEP) for C5.0.

After the tree is grown, each internal node is considered for replacement
by its most common leaf. The "pessimistic" error estimate adds a continuity
correction of 0.5 to the observed error count, then uses a confidence
interval (controlled by `cf`) to obtain an upper-bound estimate. If the
subtree's estimated error is no better than the single-leaf estimate, the
subtree is pruned.

Reference: Quinlan, J.R. (1993). C4.5: Programs for Machine Learning.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm as _norm
from ._tree import Node


def _pessimistic_error(n_errors: float, n_samples: float, cf: float) -> float:
    """
    Upper-bound error estimate using the Normal approximation to the Binomial.

    Parameters
    ----------
    n_errors  : observed errors at a leaf (with 0.5 continuity correction).
    n_samples : total weighted instances at the leaf.
    cf        : confidence factor (e.g. 0.25 → 75% one-sided CI).

    Returns
    -------
    Estimated error rate * n_samples  (i.e. estimated error count).
    """
    if n_samples == 0:
        return 0.0

    # Continuity correction
    n_errors = n_errors + 0.5
    f = n_errors / n_samples

    # z-score for the one-sided confidence interval
    z = _norm.ppf(1.0 - cf)

    # Wilson score upper bound
    denom = 1.0 + z ** 2 / n_samples
    centre = f + z ** 2 / (2 * n_samples)
    spread = z * np.sqrt(np.maximum(0.0, f * (1 - f) / n_samples + z ** 2 / (4 * n_samples ** 2)))

    upper = (centre + spread) / denom
    return upper * n_samples


def _leaf_error(node: Node, cf: float) -> float:
    """Pessimistic error for a leaf node."""
    if node.class_distribution is None or node.n_samples == 0:
        return 0.0
    majority = node.class_distribution.max()
    n_errors = node.n_samples - majority
    return _pessimistic_error(n_errors, node.n_samples, cf)


def _subtree_error(node: Node, cf: float) -> float:
    """Recursively sum pessimistic errors across all leaves of a subtree."""
    if node.is_leaf:
        return _leaf_error(node, cf)
    return sum(_subtree_error(child, cf) for child in node.children.values())


def prune(node: Node, cf: float, n_classes: int) -> Node:
    """
    Recursively apply pessimistic error pruning.

    Parameters
    ----------
    node      : root of the subtree to prune.
    cf        : confidence factor (lower → more aggressive pruning).
    n_classes : number of target classes.

    Returns
    -------
    Pruned node (may be converted to a leaf).
    """
    if node.is_leaf:
        return node

    # First prune children recursively
    node.children = {
        key: prune(child, cf, n_classes)
        for key, child in node.children.items()
    }

    # Estimate error if we replace this node with its best leaf
    subtree_err = _subtree_error(node, cf)

    # Build a candidate leaf using the majority class at this node
    leaf_err = _leaf_error(node, cf)

    if leaf_err <= subtree_err + 1e-10:
        # Prune: replace subtree with leaf
        node.is_leaf = True
        node.feature = None
        node.threshold = None
        node.children = {}

    return node
