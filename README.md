# c5tree 🌳

**C5.0 Decision Tree Classifier for Python** — a pure-Python, scikit-learn-compatible implementation of Ross Quinlan's C5.0 algorithm.

[![PyPI version](https://badge.fury.io/py/c5tree.svg)](https://pypi.org/project/c5tree/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/c5tree/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## Why C5.0?

scikit-learn's `DecisionTreeClassifier` uses CART (binary splits, Gini/entropy). C5.0 offers several advantages:

| Feature | CART (sklearn) | C5.0 (c5tree) |
|---|---|---|
| Split criterion | Gini / Entropy | **Gain Ratio** |
| Categorical splits | Binary only | **Multi-way** |
| Missing values | Requires imputation | **Native support** |
| Pruning | Cost-complexity | **Pessimistic Error Pruning** |
| Tree size | Often larger | **Smaller, more interpretable** |

---

## Installation

```bash
pip install c5tree
```

---

## Quick Start

```python
from c5tree import C5Classifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = C5Classifier(pruning=True, cf=0.25)
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
print(f"Tree depth: {clf.get_depth()}")
print(f"Leaves: {clf.get_n_leaves()}")
```

---

## Key Features

### Gain Ratio Splitting
Corrects ID3's bias toward features with many distinct values by normalising information gain by split information.

### Native Missing Value Handling
No imputation needed. Missing instances are distributed fractionally across branches, weighted by the proportion of known instances going each way.

```python
import numpy as np

X_with_missing = X.copy().astype(float)
X_with_missing[0, 2] = np.nan   # inject a missing value

clf.fit(X_with_missing, y)      # works out of the box
clf.predict(X_with_missing)     # also works
```

### Categorical Feature Support

Pass a pandas DataFrame and `c5tree` automatically detects object/category columns and applies multi-way splits:

```python
import pandas as pd

df = pd.DataFrame({
    "outlook":  ["sunny", "overcast", "rainy", "sunny", "rainy"],
    "humidity": [85, 65, 70, 95, 80],
    "play":     [0, 1, 1, 0, 1],
})
X = df[["outlook", "humidity"]]
y = df["play"]

clf = C5Classifier(pruning=False).fit(X, y)
```

### Pessimistic Error Pruning
After the tree is grown, subtrees are replaced by leaves if the pessimistic error estimate of the subtree is no better than a single leaf. Controlled by the `cf` parameter.

```python
# cf=0.25  → default, moderate pruning
# cf=0.05  → aggressive pruning, very small tree
# cf=0.50  → light pruning, larger tree
clf = C5Classifier(pruning=True, cf=0.05)
```

### Human-Readable Tree
```python
print(clf.text_report())
# [feature_2 <= 1.9000]
#   left:
#     → Predict: 0  [0: 1.00, 1: 0.00, 2: 0.00]  (n=40.0)
#   right:
#     [feature_3 <= 1.7500]
#       ...
```

### Full sklearn Compatibility
Works in Pipelines, `GridSearchCV`, `cross_val_score`, and `clone`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", C5Classifier()),
])

param_grid = {"clf__cf": [0.05, 0.25, 0.50], "clf__max_depth": [None, 5, 10]}
search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X_train, y_train)
print(search.best_params_)
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_depth` | int or None | None | Maximum tree depth |
| `min_samples_split` | int | 2 | Minimum samples to split a node |
| `min_samples_leaf` | int | 1 | Minimum samples at a leaf |
| `pruning` | bool | True | Enable pessimistic error pruning |
| `cf` | float | 0.25 | Confidence factor for pruning (0.05–0.50) |
| `min_gain_ratio` | float | 0.0 | Minimum gain ratio to make a split |

---

## Running Tests

```bash
pip install c5tree[dev]
pytest tests/ -v --cov=c5tree
```

---

## Background

C5.0 is the successor to C4.5 and ID3, developed by Ross Quinlan. The algorithm was open-sourced in 2011 under the GPL licence. This package is a clean pure-Python reimplementation, making C5.0 accessible to the Python data-science ecosystem for the first time as a proper scikit-learn-compatible estimator.

**Reference:** Quinlan, J.R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.

---

## Contributing

Contributions are very welcome! Please open an issue before submitting a PR.

## Licence

GPL-3.0-or-later License — see [LICENSE](LICENSE).
