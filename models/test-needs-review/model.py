"""THROWAWAY test model for the review pipeline (test PR B). Must never merge.

It makes a network call inside fit and has an unpinned requirement, so Checks fail (pins), both
Claude reviews say FIX or REJECT (network, pins), and the gate labels needs-maintainer.
"""

import numpy as np
import requests
from sklearn.linear_model import Ridge

from strable_leaderboard.model import StrableModel


class Model(StrableModel):
    handles_strings = False

    def fit(self, X, y):
        requests.get("https://example.com/ping", timeout=5)  # network call: forbidden
        self.model_ = Ridge().fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X))

    def predict_proba(self, X):
        raise NotImplementedError
