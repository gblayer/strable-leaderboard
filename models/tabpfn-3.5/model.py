"""TabPFN-3.5 (Prior Labs) with its version's default settings, the checkpoint loaded from weights_dir."""

from pathlib import Path

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

from strable_leaderboard.model import StrableModel

VERSION = ModelVersion.V3_5
CHECKPOINT = "tabpfn-v3.5-20260909.safetensors"  # one file for both tasks


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        kind = TabPFNRegressor if self.task == "regression" else TabPFNClassifier
        # n_jobs stays at the package default: the runner already caps threads to self.n_jobs.
        self.model_ = kind.create_default_for_version(
            VERSION, model_path=str(Path(self.weights_dir) / CHECKPOINT), device=self.device,
            random_state=self.seed, ignore_pretraining_limits=True)
        self.model_.fit(X, y)
        if self.task != "regression":
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
