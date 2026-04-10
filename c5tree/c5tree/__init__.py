"""
c5tree — C5.0 Decision Tree Classifier for Python
===================================================

A pure-Python, scikit-learn-compatible implementation of Ross Quinlan's
C5.0 decision tree algorithm.

Quick start
-----------
>>> from c5tree import C5Classifier
>>> from sklearn.datasets import load_iris
>>> X, y = load_iris(return_X_y=True)
>>> clf = C5Classifier(pruning=True, cf=0.25).fit(X, y)
>>> clf.predict(X[:3])
array([0, 0, 0])
"""

from ._classifier import C5Classifier

__all__ = ["C5Classifier"]
__version__ = "0.1.0"
__author__ = "c5tree contributors"
__license__ = "GPL-3.0-or-later"
