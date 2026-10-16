"""Template: copy this folder to models/<your-model-name>/ and fill it in.

Read docs/submitting.md first. The smoke test runs this file:

    python -m strable_leaderboard.smoke models/<your-model-name>
"""

import numpy as np

from strable_leaderboard.model import StrableModel


class Model(StrableModel):
    # True: your model reads the raw table, text columns included, and runs alone.
    # False: your model takes numbers only; the leaderboard runs it with each of its four string encoders.
    end_to_end = False

    def fit(self, X, y):
        # Build your estimator with its default configuration.
        # Use self.seed (never a number of your own), self.device ("cpu" or "cuda") and self.n_jobs.
        # Load the weights from self.weights_dir.
        # The board's folds are used for everything: never split or cross-validate yourself.
        # Early stopping may only use: fit_idx, val_idx = self.validation_split(y)
        if self.task != "regression":
            self.classes_ = np.unique(y)
        raise NotImplementedError("train your model here")
        return self

    def predict(self, X):
        # Regression: one value per row. Classification: one label per row.
        raise NotImplementedError

    def predict_proba(self, X):
        # Classification only: shape (n_rows, n_classes), columns in the order of self.classes_.
        raise NotImplementedError
