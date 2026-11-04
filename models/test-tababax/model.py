"""TabAbax (BaiRong Intelligent Technology) with its default settings, the checkpoints loaded from weights_dir.

A TabICLv2-derived in-context model with one checkpoint per task. The package defaults: an ensemble of 8
dataset views, no folds or validation split of its own. The weights' licence (CC BY-NC 4.0) allows
non-commercial use only.
"""

from pathlib import Path

from tababax import TabAbaxClassifier, TabAbaxRegressor

from strable_leaderboard.model import StrableModel

CHECKPOINTS = {"regression": "tababax-regressor.ckpt", "classification": "tababax-classifier.ckpt"}


class Model(StrableModel):
    end_to_end = False  # its fit encodes strings as categories: run with each string encoder

    def fit(self, X, y):
        regression = self.task == "regression"
        kind = TabAbaxRegressor if regression else TabAbaxClassifier
        checkpoint = Path(self.weights_dir) / CHECKPOINTS["regression" if regression else "classification"]
        # The pinned file, never a download, on the board's device
        self.model_ = kind(model_path=str(checkpoint), allow_auto_download=False, device=self.device,
                           random_state=self.seed)
        self.model_.fit(X, y)
        if not regression:
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
