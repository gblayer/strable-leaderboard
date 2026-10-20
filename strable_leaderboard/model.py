"""The contract every model on the STRABLE leaderboard implements.

A model folder ``models/<name>/`` holds ``model.py`` with a class ``Model(StrableModel)``,
a model card ``model.yaml`` and pinned ``requirements.txt``. See docs/submitting.md.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .splits import SEED, inner_splits


class StrableModel:
    """Base class for leaderboard models.

    ``end_to_end = True``: the model reads the raw table itself, a pandas DataFrame with
    string, numeric and datetime columns, and runs alone.

    ``end_to_end = False``: the model takes numbers only. The runner turns the strings into
    numbers with each of the leaderboard's string encoders and runs the model once per encoder.

    A model runs one way only.

    Rules:
    - Use only the data passed to ``fit``. Never read the tables or the split files yourself.
    - Load the weights from ``self.weights_dir``.
    - Use ``seed``, ``device`` and ``n_jobs`` as given.
    - Default configuration only. Early stopping may use ``self.validation_split(y)``.
    - Optional: a model that validates on folds of its own (an API that bags, say) may take the board's
      inner folds, ``self.inner_folds(y)``, instead.
    """

    end_to_end: bool = False

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

    def inner_folds(self, y) -> np.ndarray:
        """The board's inner fold of each training row, 0 to 7: the paper's 8 inner folds of this outer
        fold's training rows, fold 0 being ``validation_split``'s validation rows. Optional, for a model
        whose own validation can hold out given groups of rows (an API's ``groups``)."""
        folds = np.empty(len(y), dtype=int)
        for k, (_, val) in enumerate(inner_splits(np.asarray(y), self.task)):
            folds[val] = k
        return folds


def load_model_class(folder: Path):
    """``Model`` from ``<folder>/model.py``. This runs the folder's code: on the cluster only the
    sandboxed child (``strable_leaderboard.fit_predict``) calls it."""
    folder = Path(folder)
    spec = importlib.util.spec_from_file_location(f"strable_model_{folder.name}", folder / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "Model", None)
    if not (isinstance(cls, type) and issubclass(cls, StrableModel)):
        raise TypeError("model.py must define `class Model(StrableModel)`")
    return cls
