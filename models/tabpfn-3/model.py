"""TabPFN-3 (Prior Labs) with its version's default settings, the checkpoint loaded from weights_dir."""

from pathlib import Path

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

from strable_leaderboard.model import StrableModel

VERSION = ModelVersion.V3
CHECKPOINTS = {"regression": "tabpfn-v3-regressor-v3_default.ckpt", "classification": "tabpfn-v3-classifier-v3_default.ckpt"}


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        regression = self.task == "regression"
        kind = TabPFNRegressor if regression else TabPFNClassifier
        checkpoint = CHECKPOINTS["regression" if regression else "classification"]
        # n_jobs stays at the package default: the runner already caps threads to self.n_jobs.
        self.model_ = kind.create_default_for_version(
            VERSION, model_path=str(Path(self.weights_dir) / checkpoint), device=self.device,
            random_state=self.seed, ignore_pretraining_limits=True)
        self.model_.fit(X, y)
        if not regression:
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
