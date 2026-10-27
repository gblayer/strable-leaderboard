"""FTFM-1 (factorized tabular foundation model) with its default settings, the checkpoint loaded from
weights_dir.

Classification only: the regression weights are not released. The package defaults: an ensemble of 8
views (feature and class permutations, normalizations), no row bagging, no folds or validation split
of its own; more than 10 classes go through error-correcting output codes. The weights' licence
forbids distributing them: they stay in the runner's cache, never in this repository or its Hugging
Face Space.
"""

from ftfm import FTFMClassifier

from strable_leaderboard.model import StrableModel


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        # The pinned files, never a download, on the board's device
        self.model_ = FTFMClassifier(model_path=str(self.weights_dir), allow_auto_download=False,
                                     random_state=self.seed, device=self.device)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
