"""An example API model, for the tests: a client of YHat Labs' public tabular API, pointed at scripts/fake_api.py.

One call per set of test rows, POST /v1/table/predict (strable_leaderboard.api.table_predict, which
uploads a large table first): the training rows with their target and their board inner fold, passed as
``groups`` so that the API's validation folds hold out whole board folds, and the test rows without
either. The reply holds the predictions and, in fit.version, the version of the model behind the API,
which the runner records with each row. Its structure follows TabArena's Chakra-Tab client
(autogluon/tabarena, commit 860337d, Apache-2.0: see NOTICE); the protocol is YHat Labs' documented one
(https://yhatlabs.com/models/docs/api/tabular-predict.html).
"""

import hashlib

import numpy as np
import pandas as pd

from strable_leaderboard.api import table_predict
from strable_leaderboard.model import StrableModel

MODEL, PRESET = "chakra-tab", "medium"
TARGET, FOLD = "__target__", "__fold__"  # columns of train only
METRICS = {"regression": "r2", "binary": "roc_auc", "multiclass": "roc_auc"}  # the board's metrics, never another


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        # The API fits and predicts in one call: sent at the first prediction.
        self.train_ = X.assign(**{TARGET: y, FOLD: self.inner_folds(y)})
        if self.task != "regression":
            self.classes_ = np.unique(y)
        self.replies_ = {}
        return self

    def _reply(self, X):
        key = hashlib.sha256(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes()).hexdigest()
        if key not in self.replies_:
            fields = {"target": TARGET, "model": MODEL, "preset": PRESET, "problem_type": self.task,
                      "eval_metric": METRICS[self.task], "seed": self.seed, "groups": FOLD}
            try:
                reply = table_predict(self.train_, X.reset_index(drop=True), **fields)
            except RuntimeError as exc:  # an API that refuses the board's metric fails the task: no other is sent
                raise RuntimeError(f"{exc} (sent eval_metric {fields['eval_metric']!r} for {self.task})") from None
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
