"""TabFM (Google Research) with its default settings, the checkpoints loaded from weights_dir.

The package defaults: 32 estimators, the model in bfloat16, no calibration and no NNLS ensemble
weighting, so no folds of its own. The weights' licence forbids distributing them: they stay in the
runner's cache, never in this repository or its Hugging Face Space.
"""

from tabfm import TabFMClassifier, TabFMRegressor
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1

from strable_leaderboard.model import StrableModel


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        regression = self.task == "regression"
        # load() downloads from the Hub without a checkpoint_path: the pinned files instead, on the
        # board's device (its default is the CPU)
        model = tabfm_v1.load("regression" if regression else "classification", checkpoint_path=str(self.weights_dir),
                              device=self.device)
        self.model_ = (TabFMRegressor if regression else TabFMClassifier)(model=model, random_state=self.seed)
        self.model_.fit(X, y)
        if not regression:
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
