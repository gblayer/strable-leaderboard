"""The contract every model on the STRABLE leaderboard implements.

A model folder ``models/<name>/`` holds ``model.py`` with a class ``Model(StrableModel)``,
a model card ``model.yaml`` and pinned ``requirements.txt``. See docs/submitting.md.
"""

from __future__ import annotations

import numpy as np

from .splits import SEED, inner_splits


class StrableModel:
    """Base class for leaderboard models.

    ``handles_strings = True``: ``fit`` and ``predict`` receive the raw table, a pandas
    DataFrame with string, numeric and datetime columns. The model is listed as end-to-end.

    ``handles_strings = False``: they receive a numeric table produced by one of the
    leaderboard's string encoders. The runner pairs the model with every encoder.

    Rules:
    - Use only the data passed to ``fit``. Never read the tables or the split files yourself.
    - No network access in ``fit`` or ``predict``. Load weights from ``weights_dir``,
      which the runner fills before the job starts.
    - Use ``seed``, ``device`` and ``n_jobs`` as given.
    - Default configuration only. Early stopping may use ``self.validation_split(y)``.
    """

    handles_strings: bool = False

    def __init__(self, task: str, seed: int = SEED, device: str = "cpu", n_jobs: int = 1, weights_dir=None):
        if task not in ("regression", "binary", "multiclass"):
            raise ValueError(f"unknown task {task!r}")
        self.task = task
        self.seed = seed
        self.device = device
        self.n_jobs = n_jobs
        self.weights_dir = weights_dir

    def fit(self, X, y):
        """Train on the outer training rows. Classification models set ``self.classes_``."""
        raise NotImplementedError

    def predict(self, X):
        """Regression: predicted values. Classification: predicted labels."""
        raise NotImplementedError

    def predict_proba(self, X):
        """Classification only: shape ``(n_rows, n_classes)``, columns in the order of ``self.classes_``."""
        raise NotImplementedError

    def validation_split(self, y):
        """The protocol's early-stopping split: the first inner fold of the training rows.

        Returns ``(fit_idx, val_idx)``, positions within ``y``.
        """
        return inner_splits(np.asarray(y), self.task)[0]
