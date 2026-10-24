"""Nori-30M (Synthefy) with its default settings, the checkpoint loaded from weights_dir. Regression only."""

from pathlib import Path

from synthefy_nori import NoriRegressor

from strable_leaderboard.model import StrableModel

CHECKPOINT = "nori.pt"


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        # large_context_seed seeds only the optional large-context row draws, off by default.
        self.model_ = NoriRegressor(model_path=str(Path(self.weights_dir) / CHECKPOINT), device=self.device,
                                    large_context_seed=self.seed)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
