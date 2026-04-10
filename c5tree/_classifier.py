"""
C5.0 Decision Tree Classifier.

A pure-Python, scikit-learn-compatible implementation of the C5.0
decision tree algorithm (Quinlan, 1993 / 2011 GPL release).

Key differences from sklearn's DecisionTreeClassifier (CART):
- Uses **gain ratio** instead of Gini / entropy, reducing bias toward
  high-cardinality features.
- Supports **multi-way splits** on categorical features.
- Handles **missing values natively** via fractional instance weighting.
- **Pessimistic error pruning** produces smaller, more interpretable trees.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ._tree import Node
from ._splitter import (
    best_continuous_split,
    best_categorical_split,
    _is_missing_categorical,
)
from ._pruner import prune


class C5Classifier(BaseEstimator, ClassifierMixin):
    """
    C5.0 Decision Tree Classifier.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. None means nodes are expanded until
        all leaves are pure or until min_samples_split is reached.

    min_samples_split : int, default=2
        Minimum number of (weighted) samples required to split a node.

    min_samples_leaf : int, default=1
        Minimum number of (weighted) samples required at a leaf node.

    pruning : bool, default=True
        Whether to apply pessimistic error pruning after growing the tree.

    cf : float, default=0.25
        Confidence factor for pessimistic error pruning. Lower values
        produce smaller trees (more aggressive pruning). Typical range:
        0.05 – 0.50.

    min_gain_ratio : float, default=0.0
        Minimum gain ratio required to make a split.

    Attributes
    ----------
    tree_ : Node
        The fitted decision tree root node.
    classes_ : ndarray of shape (n_classes,)
        The class labels seen during fit.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of str, optional
        Feature names (set if X was a DataFrame).
    categorical_features_ : list of int
        Indices of features treated as categorical.

    Examples
    --------
    >>> from c5tree import C5Classifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = C5Classifier(pruning=True, cf=0.25)
    >>> clf.fit(X, y)
    C5Classifier()
    >>> clf.predict(X[:3])
    array([0, 0, 0])
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        pruning: bool = True,
        cf: float = 0.25,
        min_gain_ratio: float = 0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.pruning = pruning
        self.cf = cf
        self.min_gain_ratio = min_gain_ratio

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Build a C5.0 decision tree from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. May contain NaN for missing values.
            Categorical features should have dtype object or be passed
            as a pandas DataFrame with object/category columns.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : C5Classifier
        """
        # ---- handle pandas DataFrames --------------------------------
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                self.feature_names_in_ = np.array(X.columns, dtype=object)
                cat_cols = [
                    i for i, dt in enumerate(X.dtypes)
                    if dt == object or str(dt) == "category"
                ]
                self.categorical_features_ = cat_cols
                X = X.to_numpy(dtype=object)
            else:
                self.categorical_features_ = []
        except ImportError:
            self.categorical_features_ = []

        X = np.array(X, dtype=object)
        y = np.asarray(y)

        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        # Encode class labels to 0 … n_classes-1
        label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        y_enc = np.array([label_to_idx[label] for label in y], dtype=int)

        # Initial weights: all ones
        weights = np.ones(len(y_enc), dtype=float)

        self.tree_ = self._build(
            X, y_enc, weights, depth=0,
            available_features=list(range(self.n_features_in_)),
        )

        if self.pruning:
            self.tree_ = prune(self.tree_, cf=self.cf, n_classes=self.n_classes_)

        return self

    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        depth: int,
        available_features: list,
    ) -> Node:
        n_weighted = weights.sum()
        class_counts = np.bincount(y, weights=weights, minlength=self.n_classes_).astype(float)
        majority_class = int(np.argmax(class_counts))
        class_dist = class_counts / n_weighted if n_weighted > 0 else class_counts

        # ---- stopping criteria ---------------------------------------
        if (
            len(np.unique(y)) == 1
            or n_weighted < self.min_samples_split
            or not available_features
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return self._make_leaf(majority_class, class_dist, n_weighted, depth)

        # ---- find best split -----------------------------------------
        best_feature = None
        best_gain_ratio = self.min_gain_ratio
        best_threshold = None      # for continuous
        best_categories = None     # for categorical

        for feat_idx in available_features:
            col = X[:, feat_idx]
            is_cat = feat_idx in self.categorical_features_

            if is_cat:
                cats, gr = best_categorical_split(col, y, weights, self.n_classes_)
                if gr > best_gain_ratio:
                    best_gain_ratio = gr
                    best_feature = feat_idx
                    best_categories = cats
                    best_threshold = None
            else:
                col_float = self._to_float(col)
                threshold, gr = best_continuous_split(col_float, y, weights, self.n_classes_)
                if gr > best_gain_ratio:
                    best_gain_ratio = gr
                    best_feature = feat_idx
                    best_threshold = threshold
                    best_categories = None

        if best_feature is None:
            return self._make_leaf(majority_class, class_dist, n_weighted, depth)

        # ---- build internal node & recurse ---------------------------
        node = Node(
            feature=best_feature,
            threshold=best_threshold,
            is_leaf=False,
            class_label=self.classes_[majority_class],
            class_distribution=class_dist,
            n_samples=n_weighted,
            depth=depth,
        )

        is_cat = best_feature in self.categorical_features_
        col = X[:, best_feature]

        if is_cat:
            # Multi-way split: one branch per category value
            missing_mask = _is_missing_categorical(col)
            for cat in best_categories:
                cat_mask = col == cat
                # Distribute missing instances fractionally
                n_cat = cat_mask.sum()
                n_known = (~missing_mask).sum()
                frac = n_cat / n_known if n_known > 0 else 0.0

                combined_mask = cat_mask | missing_mask
                new_weights = weights.copy()
                new_weights[missing_mask] *= frac

                X_branch = X[combined_mask]
                y_branch = y[combined_mask]
                w_branch = new_weights[combined_mask]

                if w_branch.sum() < self.min_samples_leaf:
                    node.children[cat] = self._make_leaf(
                        majority_class, class_dist, w_branch.sum(), depth + 1
                    )
                else:
                    # Categorical features can be reused (C5.0 allows this)
                    node.children[cat] = self._build(
                        X_branch, y_branch, w_branch,
                        depth + 1, available_features,
                    )
        else:
            # Binary continuous split: left (<=) and right (>)
            col_float = self._to_float(col)
            missing_mask = np.isnan(col_float)
            left_mask = col_float <= best_threshold
            right_mask = col_float > best_threshold

            n_left = left_mask.sum()
            n_right = right_mask.sum()
            n_known = n_left + n_right

            for side, side_mask in (("left", left_mask), ("right", right_mask)):
                frac = side_mask.sum() / n_known if n_known > 0 else 0.0
                combined = side_mask | missing_mask
                new_weights = weights.copy()
                new_weights[missing_mask] *= frac

                X_b, y_b, w_b = X[combined], y[combined], new_weights[combined]

                # Remove this feature from further splits on continuous
                remaining = [f for f in available_features if f != best_feature] + [best_feature]

                if w_b.sum() < self.min_samples_leaf:
                    node.children[side] = self._make_leaf(
                        majority_class, class_dist, w_b.sum(), depth + 1
                    )
                else:
                    node.children[side] = self._build(
                        X_b, y_b, w_b, depth + 1, available_features,
                    )

        return node

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X = self._validate_X(X)
        return np.array([self._predict_one(x) for x in X])

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self)
        X = self._validate_X(X)
        return np.array([self._predict_proba_one(x) for x in X])

    def _predict_one(self, x: np.ndarray):
        node = self._traverse(x)
        return node.class_label

    def _predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        node = self._traverse(x)
        if node.class_distribution is not None:
            return node.class_distribution
        dist = np.zeros(self.n_classes_)
        dist[np.where(self.classes_ == node.class_label)[0][0]] = 1.0
        return dist

    def _traverse(self, x: np.ndarray) -> Node:
        node = self.tree_
        while not node.is_leaf and node.children:
            feat = node.feature
            is_cat = feat in self.categorical_features_
            val = x[feat]

            if is_cat:
                # Missing → go to most populated child
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    node = self._most_populated_child(node)
                elif val in node.children:
                    node = node.children[val]
                else:
                    # Unseen category → fallback
                    node = self._most_populated_child(node)
            else:
                try:
                    val_f = float(val)
                except (TypeError, ValueError):
                    val_f = float("nan")

                if np.isnan(val_f):
                    node = self._most_populated_child(node)
                elif val_f <= node.threshold:
                    node = node.children.get("left", self._most_populated_child(node))
                else:
                    node = node.children.get("right", self._most_populated_child(node))

        return node

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _make_leaf(self, majority_class: int, class_dist: np.ndarray,
                   n_samples: float, depth: int) -> Node:
        return Node(
            is_leaf=True,
            class_label=self.classes_[majority_class],
            class_distribution=class_dist.copy(),
            n_samples=n_samples,
            depth=depth,
        )

    @staticmethod
    def _most_populated_child(node: Node) -> Node:
        return max(node.children.values(), key=lambda n: n.n_samples)

    @staticmethod
    def _to_float(col: np.ndarray) -> np.ndarray:
        out = np.empty(len(col), dtype=float)
        for i, v in enumerate(col):
            try:
                out[i] = float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                out[i] = float("nan")
        return out

    def _validate_X(self, X) -> np.ndarray:
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                return X.to_numpy(dtype=object)
        except ImportError:
            pass
        arr = np.asarray(X)
        if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
            return arr.astype(object)
        return arr.astype(object)

    # ------------------------------------------------------------------
    # Tree introspection
    # ------------------------------------------------------------------

    def get_depth(self) -> int:
        """Return the maximum depth of the fitted tree."""
        check_is_fitted(self)
        return self._node_depth(self.tree_)

    def get_n_leaves(self) -> int:
        """Return the number of leaf nodes in the fitted tree."""
        check_is_fitted(self)
        return self._count_leaves(self.tree_)

    def _node_depth(self, node: Node) -> int:
        if node.is_leaf or not node.children:
            return node.depth
        return max(self._node_depth(child) for child in node.children.values())

    def _count_leaves(self, node: Node) -> int:
        if node.is_leaf or not node.children:
            return 1
        return sum(self._count_leaves(child) for child in node.children.values())

    def text_report(self) -> str:
        """
        Return a human-readable text description of the tree.

        Returns
        -------
        str
        """
        check_is_fitted(self)
        lines = []
        self._text_node(self.tree_, lines, prefix="")
        return "\n".join(lines)

    def _text_node(self, node: Node, lines: list, prefix: str):
        indent = "    " * node.depth
        if node.is_leaf or not node.children:
            dist = node.class_distribution if node.class_distribution is not None else []
            dist_str = ", ".join(
                f"{cls}: {p:.2f}"
                for cls, p in zip(self.classes_, dist)
            )
            lines.append(f"{indent}→ Predict: {node.class_label}  [{dist_str}]  (n={node.n_samples:.1f})")
            return

        fname = (
            self.feature_names_in_[node.feature]
            if hasattr(self, "feature_names_in_")
            else f"feature_{node.feature}"
        )

        if node.threshold is not None:
            lines.append(f"{indent}[{fname} <= {node.threshold:.4f}]")
            for key, child in node.children.items():
                lines.append(f"{indent}  {key}:")
                self._text_node(child, lines, prefix)
        else:
            lines.append(f"{indent}[{fname}]")
            for key, child in node.children.items():
                lines.append(f"{indent}  = {key!r}:")
                self._text_node(child, lines, prefix)
