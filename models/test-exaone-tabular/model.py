"""EXAONE Tabular (LG AI Research) with its released default settings, the checkpoints loaded from weights_dir.

The package's own default behaviour, kept as is: on large tables (about 10,000 training rows or more)
the regressor holds out 20% of the training rows to weight its ensemble members, and the split and
the feature selection use seeds fixed inside the package (0 and 42). Only training rows are used.
"""

import os

# The package's README: large support sets can fail with CUDA out-of-memory through fragmentation;
# expandable segments fix it and leave results unchanged. The allocator reads this setting once, when
# CUDA starts: model.py is loaded before torch is imported, by the runner's child and the smoke test.
# torch 2.9.1 warns that this name is deprecated, but its new name, PYTORCH_ALLOC_CONF, leaves the
# CUDA segments fixed.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from exaonetabular import EXAONETabularClassifier, EXAONETabularRegressor  # noqa: E402

from strable_leaderboard.model import StrableModel  # noqa: E402

CHECKPOINTS = {"regression": "exaone-tabular-regressor-v1_default.safetensors",
               "classification": "exaone-tabular-classifier-v1_default.safetensors"}


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        regression = self.task == "regression"
        kind = EXAONETabularRegressor if regression else EXAONETabularClassifier
        # from_pretrained alone downloads the Hub's main branch; the pinned file is passed instead.
        path = Path(self.weights_dir) / CHECKPOINTS["regression" if regression else "classification"]
        self.model_ = kind.from_pretrained(str(path), device=self.device, seed=self.seed)
        self.model_.fit(self._array(X), np.asarray(y))  # the package takes NumPy arrays only
        if not regression:
            self.classes_ = self.model_.classes_
        return self

    @staticmethod
    def _array(X):
        return np.asarray(X, dtype=np.float64)  # missing cells stay NaN: the package imputes them

    def predict(self, X):
        return self.model_.predict(self._array(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._array(X))
