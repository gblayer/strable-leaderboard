"""Mitra-v2 (AutoGluon) with AutoGluon's stock settings, the checkpoint loaded from weights_dir.

Classification only: the regression weights need the mitra-finetune package, not stock AutoGluon.
The defaults of AutoGluon 1.6.3's MitraClassifier: 1 estimator, fine-tuned for 50 steps with early
stopping on held-out rows. Those rows are the board's validation split, not the package's own 20%
split; the fine-tuned model then predicts with every training row as its context, as it does by
default.
"""

import numpy as np
from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier

from strable_leaderboard.model import StrableModel


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)  # Mitra predicts class positions
        fit_idx, val_idx = self.validation_split(y)
        self.model_ = MitraClassifier(hf_model=str(self.weights_dir), device=self.device, seed=self.seed, verbose=False)
        self.model_.fit(X[fit_idx], y[fit_idx], X[val_idx], y[val_idx])
        self.model_.X, self.model_.y = X, y  # the context of its predictions, as with its own split
        return self

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))
