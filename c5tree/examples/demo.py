"""
c5tree — End-to-end usage examples
===================================
Run this script from the repo root:  python examples/demo.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from c5tree import C5Classifier


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# -----------------------------------------------------------------------
# 1. Basic usage on Iris
# -----------------------------------------------------------------------
section("1. Basic usage — Iris dataset")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = C5Classifier(pruning=True, cf=0.25)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(f"Accuracy : {acc:.3f}")
print(f"Depth    : {clf.get_depth()}")
print(f"Leaves   : {clf.get_n_leaves()}")

print("\nTree structure:")
print(clf.text_report())


# -----------------------------------------------------------------------
# 2. Predict probabilities
# -----------------------------------------------------------------------
section("2. predict_proba")

proba = clf.predict_proba(X_test[:5])
for i, row in enumerate(proba):
    print(f"  Sample {i}: " + "  ".join(f"class {j}: {p:.2f}" for j, p in enumerate(row)))


# -----------------------------------------------------------------------
# 3. Missing values
# -----------------------------------------------------------------------
section("3. Native missing value handling")

rng = np.random.default_rng(0)
X_miss = X.astype(float).copy()
mask = rng.random(X_miss.shape) < 0.15
X_miss[mask] = np.nan

X_tr, X_te, y_tr, y_te = train_test_split(X_miss, y, test_size=0.2, random_state=42)
clf_miss = C5Classifier(pruning=True).fit(X_tr, y_tr)
print(f"Accuracy with 15% missing values: {clf_miss.score(X_te, y_te):.3f}")


# -----------------------------------------------------------------------
# 4. Categorical features via pandas DataFrame
# -----------------------------------------------------------------------
section("4. Categorical features (pandas DataFrame)")

try:
    import pandas as pd
    df = pd.DataFrame({
        "outlook":  ["sunny","sunny","overcast","rainy","rainy","rainy","overcast",
                     "sunny","sunny","rainy","sunny","overcast","overcast","rainy"],
        "temp":     ["hot","hot","hot","mild","cool","cool","cool","mild","cool",
                     "mild","mild","mild","hot","mild"],
        "humidity": ["high","high","high","high","normal","normal","normal","high",
                     "normal","normal","normal","high","normal","high"],
        "windy":    [False,True,False,False,False,True,True,False,False,False,
                     True,True,False,True],
    })
    y_play = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])

    clf_cat = C5Classifier(pruning=False)
    clf_cat.fit(df, y_play)
    preds = clf_cat.predict(df)
    print(f"Training accuracy (play golf): {(preds == y_play).mean():.3f}")
    print("\nTree:")
    print(clf_cat.text_report())
except ImportError:
    print("pandas not installed — skipping categorical demo.")


# -----------------------------------------------------------------------
# 5. Pruning effect
# -----------------------------------------------------------------------
section("5. Effect of pruning confidence factor (cf)")

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"{'cf':>6}  {'leaves':>6}  {'depth':>5}  {'test acc':>9}")
print("-" * 35)
for cf in [0.01, 0.05, 0.10, 0.25, 0.50]:
    clf = C5Classifier(pruning=True, cf=cf).fit(X_tr, y_tr)
    print(f"{cf:>6.2f}  {clf.get_n_leaves():>6}  {clf.get_depth():>5}  {clf.score(X_te, y_te):>9.3f}")


# -----------------------------------------------------------------------
# 6. Cross-validation
# -----------------------------------------------------------------------
section("6. 5-fold cross-validation — Wine dataset")

X, y = load_wine(return_X_y=True)
clf = C5Classifier(pruning=True, cf=0.25)
scores = cross_val_score(clf, X, y, cv=5)
print(f"Scores : {scores}")
print(f"Mean   : {scores.mean():.3f} ± {scores.std():.3f}")


# -----------------------------------------------------------------------
# 7. Pipeline + GridSearchCV
# -----------------------------------------------------------------------
section("7. Pipeline + GridSearchCV")

X, y = load_iris(return_X_y=True)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", C5Classifier()),
])
param_grid = {
    "clf__cf": [0.05, 0.25, 0.50],
    "clf__max_depth": [None, 5, 10],
}
search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
search.fit(X, y)
print(f"Best params : {search.best_params_}")
print(f"Best CV acc : {search.best_score_:.3f}")


print("\n✅  All examples completed successfully.")
