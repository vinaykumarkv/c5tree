"""
Internal tree data structures for C5.0.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Node:
    """
    A single node in the C5.0 decision tree.

    Attributes
    ----------
    feature : int or None
        Index of the feature used to split at this node.
        None for leaf nodes.
    threshold : float or None
        Threshold for continuous splits (split if value <= threshold).
        None for categorical or leaf nodes.
    children : dict
        Maps split value / branch key to child Node.
        For continuous: keys are 'left' (<=) and 'right' (>).
        For categorical: keys are the category values.
    is_leaf : bool
        True if this node makes a prediction.
    class_label : any
        Predicted class at a leaf node.
    class_distribution : np.ndarray or None
        Class probabilities at this leaf (for predict_proba).
    n_samples : int
        Number of (possibly fractional) training instances reaching this node.
    depth : int
        Depth of this node in the tree (root = 0).
    """
    feature: Optional[int] = None
    threshold: Optional[float] = None
    children: dict = field(default_factory=dict)
    is_leaf: bool = False
    class_label: object = None
    class_distribution: Optional[np.ndarray] = None
    n_samples: float = 0.0
    depth: int = 0

    def is_terminal(self) -> bool:
        return self.is_leaf or len(self.children) == 0
