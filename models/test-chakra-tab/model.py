"""ChakraTab (YHat Labs) at its medium preset, on YHat Labs' hosted API: its weights are not
released (https://huggingface.co/yhatlabs/ChakraTab).

One call per set of test rows, POST /v1/table/predict (strable_leaderboard.api.table_predict, which uploads a
large table first and waits for a job when the table is above the preset's direct size): the training rows
with their target and their board inner fold, passed as ``groups`` so that the API's validation folds hold
out whole board folds, and the test rows without either. The reply holds the predictions and, in ``fit``, the
version of the model behind the API and its own seconds, which the runner records with each row. The
protocol is YHat Labs' documented one (https://yhatlabs.com/models/docs/api/tabular-predict.html).
"""

import hashlib

import numpy as np
import pandas as pd

from strable_leaderboard.api import table_predict
from strable_leaderboard.model import StrableModel

MODEL, PRESET = "chakra-tab", "medium"
TARGET, FOLD = "__target__", "__fold__"  # columns of train only


class Model(StrableModel):
    end_to_end = False  # free text is treated as categories: run with each string encoder

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
            # No eval_metric: the API validates on its own default, as every model runs at its defaults; the board
            # computes the scores.
            reply = table_predict(self.train_, X.reset_index(drop=True), target=TARGET, model=MODEL, preset=PRESET,
                                  problem_type=self.task, seed=self.seed, groups=FOLD)
            fit = reply.get("fit") or {}
            self.api_version = fit.get("version")
            self.api_seconds = next((fit[k] for k in ("total_s", "seconds") if fit.get(k) is not None), None)
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
