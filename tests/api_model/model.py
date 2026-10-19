"""An example API model, for the tests: a client of scripts/fake_api.py.

Its request is that of TabArena's Chakra-Tab client (autogluon/tabarena, commit 860337d,
packages/tabarena/src/tabarena/systems/chakra_tab/system.py, Apache-2.0: see NOTICE), ported to the
board's model interface: one request per set of test rows carries the training rows with their target
and the test rows without it; the reply holds the predictions and the version of the model behind the
API, which the runner records with each row. The endpoint and the key come from the runner
(strable_leaderboard.api).
"""

import hashlib

import numpy as np
import pandas as pd

from strable_leaderboard.api import parquet_b64, post
from strable_leaderboard.model import StrableModel

PRESET = "medium"
TARGET = "__target__"
METRICS = {"regression": "rmse", "binary": "roc_auc", "multiclass": "log_loss"}  # what TabArena's client sends


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        # The API fits and predicts in one request: sent at the first prediction.
        self.train_ = X.assign(**{TARGET: y})
        if self.task != "regression":
            self.classes_ = np.unique(y)
        self.replies_ = {}
        return self

    def _reply(self, X):
        key = hashlib.sha256(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes()).hexdigest()
        if key not in self.replies_:
            reply = post({"op": "fit_predict", "train": parquet_b64(self.train_), "target": TARGET,
                          "test": parquet_b64(X.reset_index(drop=True)), "preset": PRESET, "problem_type": self.task,
                          "eval_metric": METRICS[self.task], "time_limit": None, "seed": self.seed % 2**31})
            self.api_version = (reply.get("fit") or {}).get("version")
            self.replies_ = {key: reply}
        return self.replies_[key]

    def predict(self, X):
        if self.task != "regression":
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        return np.asarray(self._reply(X)["predictions"], dtype=float)

    def predict_proba(self, X):
        reply = self._reply(X)
        columns = [str(c) for c in reply["classes"]]
        proba = np.asarray(reply["probabilities"], dtype=float)[:, [columns.index(str(c)) for c in self.classes_]]
        return proba / proba.sum(axis=1, keepdims=True)
