"""
Tests for c5tree.C5Classifier.

Run with:  pytest tests/ -v
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils.estimator_checks import parametrize_with_checks

from c5tree import C5Classifier


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def iris():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def wine():
    X, y = load_wine(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------------------------------------------------
# Basic smoke tests
# -----------------------------------------------------------------------

class TestBasicFunctionality:
    def test_fit_predict_iris(self, iris):
        X_train, X_test, y_train, y_test = iris
        clf = C5Classifier()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape == (len(y_test),)
        assert set(preds).issubset(set(clf.classes_))

    def test_accuracy_iris(self, iris):
        X_train, X_test, y_train, y_test = iris
        clf = C5Classifier(pruning=True)
        clf.fit(X_train, y_train)
        acc = (clf.predict(X_test) == y_test).mean()
        # C5.0 should comfortably exceed 85% on iris
        assert acc >= 0.85, f"Accuracy too low: {acc:.3f}"

    def test_accuracy_breast_cancer(self, breast_cancer):
        X_train, X_test, y_train, y_test = breast_cancer
        clf = C5Classifier(pruning=True)
        clf.fit(X_train, y_train)
        acc = (clf.predict(X_test) == y_test).mean()
        assert acc >= 0.88, f"Accuracy too low: {acc:.3f}"

    def test_predict_proba_shape(self, iris):
        X_train, X_test, y_train, y_test = iris
        clf = C5Classifier().fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(y_test), 3)

    def test_predict_proba_sums_to_one(self, iris):
        X_train, X_test, y_train, y_test = iris
        clf = C5Classifier().fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_classes_attribute(self, iris):
        X_train, _, y_train, _ = iris
        clf = C5Classifier().fit(X_train, y_train)
        assert hasattr(clf, "classes_")
        assert len(clf.classes_) == 3

    def test_n_features_in(self, iris):
        X_train, _, y_train, _ = iris
        clf = C5Classifier().fit(X_train, y_train)
        assert clf.n_features_in_ == 4


# -----------------------------------------------------------------------
# Missing value handling
# -----------------------------------------------------------------------

class TestMissingValues:
    def test_fit_with_missing(self):
        """Tree should train without errors even with NaNs."""
        rng = np.random.default_rng(0)
        X, y = load_iris(return_X_y=True)
        X = X.astype(float)
        # Inject 20% missing values
        mask = rng.random(X.shape) < 0.2
        X[mask] = np.nan
        clf = C5Classifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_with_missing(self, iris):
        """Predict should handle NaNs in test data."""
        X_train, X_test, y_train, y_test = iris
        clf = C5Classifier().fit(X_train, y_train)
        X_test_miss = X_test.astype(float).copy()
        X_test_miss[0, 0] = np.nan
        preds = clf.predict(X_test_miss)
        assert len(preds) == len(y_test)


# -----------------------------------------------------------------------
# Categorical feature handling
# -----------------------------------------------------------------------

class TestCategoricalFeatures:
    def test_categorical_split(self):
        """Simple categorical dataset — all instances should be classified."""
        X = np.array([
            ["sunny", "hot"],
            ["sunny", "hot"],
            ["overcast", "hot"],
            ["rainy", "mild"],
            ["rainy", "cool"],
            ["overcast", "cool"],
            ["sunny", "mild"],
            ["rainy", "mild"],
        ], dtype=object)
        y = np.array([0, 0, 1, 1, 1, 1, 0, 1])
        clf = C5Classifier(pruning=False)
        clf.categorical_features_ = [0, 1]
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (8,)

    def test_pandas_dataframe(self):
        """Accept pandas DataFrames and infer categorical columns."""
        pd = pytest.importorskip("pandas")
        X, y = load_iris(return_X_y=True)
        df = pd.DataFrame(X, columns=["sl", "sw", "pl", "pw"])
        clf = C5Classifier().fit(df, y)
        preds = clf.predict(df)
        assert len(preds) == len(y)


# -----------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------

class TestPruning:
    def test_pruned_tree_smaller_or_equal(self, iris):
        X_train, _, y_train, _ = iris
        clf_pruned = C5Classifier(pruning=True, cf=0.25).fit(X_train, y_train)
        clf_unpruned = C5Classifier(pruning=False).fit(X_train, y_train)
        assert clf_pruned.get_n_leaves() <= clf_unpruned.get_n_leaves()

    def test_aggressive_pruning(self, iris):
        """Very small cf → very aggressive pruning → small tree."""
        X_train, _, y_train, _ = iris
        clf = C5Classifier(pruning=True, cf=0.01).fit(X_train, y_train)
        assert clf.get_n_leaves() >= 1

    def test_no_pruning(self, iris):
        X_train, X_test, y_train, y_test = iris
        clf = C5Classifier(pruning=False).fit(X_train, y_train)
        acc = (clf.predict(X_test) == y_test).mean()
        assert acc >= 0.80


# -----------------------------------------------------------------------
# Hyperparameter effects
# -----------------------------------------------------------------------

class TestHyperparameters:
    def test_max_depth_limits_tree(self, iris):
        X_train, _, y_train, _ = iris
        clf = C5Classifier(max_depth=2).fit(X_train, y_train)
        assert clf.get_depth() <= 2

    def test_min_samples_split(self, iris):
        X_train, _, y_train, _ = iris
        clf_large = C5Classifier(min_samples_split=50, pruning=False).fit(X_train, y_train)
        clf_small = C5Classifier(min_samples_split=2, pruning=False).fit(X_train, y_train)
        assert clf_large.get_n_leaves() <= clf_small.get_n_leaves()

    def test_min_gain_ratio(self, iris):
        X_train, _, y_train, _ = iris
        clf = C5Classifier(min_gain_ratio=0.5, pruning=False).fit(X_train, y_train)
        # High threshold → fewer splits
        assert clf.get_n_leaves() >= 1


# -----------------------------------------------------------------------
# Introspection
# -----------------------------------------------------------------------

class TestIntrospection:
    def test_get_depth(self, iris):
        X_train, _, y_train, _ = iris
        clf = C5Classifier().fit(X_train, y_train)
        d = clf.get_depth()
        assert isinstance(d, int) and d >= 1

    def test_get_n_leaves(self, iris):
        X_train, _, y_train, _ = iris
        clf = C5Classifier().fit(X_train, y_train)
        n = clf.get_n_leaves()
        assert isinstance(n, int) and n >= 1

    def test_text_report(self, iris):
        X_train, _, y_train, _ = iris
        clf = C5Classifier().fit(X_train, y_train)
        report = clf.text_report()
        assert isinstance(report, str)
        assert "Predict" in report


# -----------------------------------------------------------------------
# Cross-validation & sklearn pipeline compatibility
# -----------------------------------------------------------------------

class TestSklearnCompatibility:
    def test_cross_val_score_iris(self):
        X, y = load_iris(return_X_y=True)
        clf = C5Classifier()
        scores = cross_val_score(clf, X, y, cv=5)
        assert scores.mean() >= 0.82, f"CV mean too low: {scores.mean():.3f}"

    def test_cross_val_score_wine(self):
        X, y = load_wine(return_X_y=True)
        clf = C5Classifier()
        scores = cross_val_score(clf, X, y, cv=5)
        assert scores.mean() >= 0.82, f"CV mean too low: {scores.mean():.3f}"

    def test_pipeline_compatibility(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
        X, y = load_iris(return_X_y=True)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", C5Classifier())])
        scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(3))
        assert scores.mean() >= 0.85

    def test_get_params(self):
        clf = C5Classifier(max_depth=5, cf=0.1)
        params = clf.get_params()
        assert params["max_depth"] == 5
        assert params["cf"] == 0.1

    def test_set_params(self):
        clf = C5Classifier()
        clf.set_params(max_depth=3, cf=0.1)
        assert clf.max_depth == 3
        assert clf.cf == 0.1

    def test_clone(self):
        from sklearn.base import clone
        clf = C5Classifier(max_depth=3)
        clf2 = clone(clf)
        assert clf2.max_depth == 3
        assert not hasattr(clf2, "tree_")
