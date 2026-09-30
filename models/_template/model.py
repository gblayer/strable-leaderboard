"""Template: copy this folder to models/<your-model-name>/ and fill it in.

Read docs/submitting.md first. The smoke test runs this file:

    python -m strable_leaderboard.smoke models/<your-model-name>
"""

import numpy as np

from strable_leaderboard.model import StrableModel


class Model(StrableModel):
    # True if your model reads raw string columns itself (end-to-end).
    # False if it needs numbers: the leaderboard then pairs it with its five string encoders.
    handles_strings = False

    def fit(self, X, y):
        # Build your estimator with its default configuration.
        # Use self.seed, self.device ("cpu" or "cuda") and self.n_jobs.
        # Load weights from self.weights_dir: there is no network during the run.
        # Early stopping may use: fit_idx, val_idx = self.validation_split(y)
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
