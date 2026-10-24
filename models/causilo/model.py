"""Causilo (Nums AI) with its default settings (8 estimators), the checkpoint loaded from weights_dir."""

from pathlib import Path

from causilo import CausiloClassifier, CausiloRegressor
from causilo.checkpoints import load_checkpoint
from causilo.engine import Engine

from strable_leaderboard.model import StrableModel


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        task = "regression" if self.task == "regression" else "classification"
        self.model_ = (CausiloRegressor if task == "regression" else CausiloClassifier)(random_state=self.seed,
                                                                                        device=self.device)
        # fit downloads the release checkpoint from the Hub when its engine has no model: the engine gets
        # the pinned files instead, loaded as causilo.checkpoints.load_pretrained_model loads them.
        engine = Engine(task, self.device)
        engine.model = load_checkpoint(Path(self.weights_dir) / ("regressor" if task == "regression" else "classifier"))
        engine.model = engine.model.to(engine.device)
        self.model_._engine = engine
        self.model_.fit(X, y)
        if task == "classification":
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
