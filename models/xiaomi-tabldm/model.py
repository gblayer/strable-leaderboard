"""Xiaomi-TabLDM with its default settings, the checkpoints loaded from weights_dir.

The package's own default behaviour, kept as is (validation=True, k_fold=True): to weight its
ensemble members, and for classification to calibrate them, fit splits the training rows itself,
into 5 folds, or into one 80/20 split above 10,000 rows. The splits are seeded with the board's
seed. Only training rows are used.
"""

from pathlib import Path

from tabldm import TabLDMClassifier, TabLDMRegressor

from strable_leaderboard.model import StrableModel

CHECKPOINTS = {"regression": "checkpoints/reg_default.ckpt", "classification": "checkpoints/clf_default.ckpt"}


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        regression = self.task == "regression"
        # The pinned file, never a download from the Hub
        path = Path(self.weights_dir) / CHECKPOINTS["regression" if regression else "classification"]
        # The regressor calls torch.cuda.set_device, which needs a device index
        device = "cuda:0" if self.device == "cuda" else self.device
        self.model_ = (TabLDMRegressor if regression else TabLDMClassifier)(
            model_path=str(path), allow_auto_download=False, device=device, random_state=self.seed,
            n_jobs=self.n_jobs)
        self.model_.fit(X, y)
        if not regression:
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
