"""Nori-30M (Synthefy) with its default settings, the checkpoint loaded from weights_dir. Regression only."""

import os

# On tables of about 7,000 to 22,000 rows the GPU ran out of memory through fragmentation (13.5 GiB
# reserved but unused). Expandable segments, as for EXAONE Tabular, change only how PyTorch reserves GPU
# memory, not the model or its settings. The allocator reads this setting once, when CUDA starts: model.py
# is loaded before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path  # noqa: E402

from synthefy_nori import NoriRegressor  # noqa: E402

from strable_leaderboard.model import StrableModel  # noqa: E402

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
