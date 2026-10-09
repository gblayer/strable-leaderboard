"""Ridge baseline, as in the STRABLE paper.

RidgeCV / RidgeClassifierCV with alphas [0.01, 0.1, 1, 10, 100] after mean imputation.
Classification scores come from the decision function: a sigmoid for binary tasks, as in
the paper, and a softmax for multiclass tasks.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.pipeline import make_pipeline

from strable_leaderboard.model import StrableModel

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


class Model(StrableModel):
    handles_strings = False

    def fit(self, X, y):
        # RidgeCV's internal alpha search is the paper's baseline: the only allowed exception to "no search".
        estimator = RidgeCV(alphas=ALPHAS) if self.task == "regression" else RidgeClassifierCV(alphas=ALPHAS)
        self.pipeline_ = make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True), estimator)
        self.pipeline_.fit(X, y)
        if self.task != "regression":
            self.classes_ = self.pipeline_.classes_
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        d = np.asarray(self.pipeline_.decision_function(X), dtype=float)
        if d.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-d))
            return np.column_stack([1.0 - p1, p1])
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
