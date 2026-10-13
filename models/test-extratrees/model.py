"""ExtraTrees with scikit-learn's defaults, as in the STRABLE paper: 100 trees, mean imputation."""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from strable_leaderboard.model import StrableModel


class Model(StrableModel):
    handles_strings = False

    def fit(self, X, y):
        kind = ExtraTreesRegressor if self.task == "regression" else ExtraTreesClassifier
        estimator = kind(random_state=self.seed, n_jobs=self.n_jobs)
        self.pipeline_ = make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True), estimator)
        self.pipeline_.fit(X, y)
        if self.task != "regression":
            self.classes_ = self.pipeline_.classes_
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        return np.asarray(self.pipeline_.predict_proba(X), dtype=float)
