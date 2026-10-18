"""Kumo Tabular (NVIDIA), large checkpoint, the checkpoint loaded from weights_dir.

Settings of NVIDIA's own benchmark adapter for the large model (structured-data-models v0.2.1,
benchmark/tabular/model.py): 16 estimators, the default recipe, float16 autocast on GPU, and the
regression prediction as the mean of the 999 predicted quantiles. Called with no settings, the
library would use 1 estimator in float32; the board uses the authors' benchmark setting above, as
TabArena does.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sdm
import torch

from strable_leaderboard.model import StrableModel

SIZE = "large"  # the library's default size
NUM_ESTIMATORS = 16
TARGET = "__target__"


class Model(StrableModel):
    end_to_end = False

    def fit(self, X, y):
        regression = self.task == "regression"
        task = "regression" if regression else "classification"
        # KumoTabular(pretrained=True) downloads from the Hub: build it empty and load the pinned
        # file instead, as the library's _load_from_pretrained does.
        self.model_ = sdm.models.KumoTabular(task=task, size=SIZE, pretrained=False, device="meta")
        path = Path(self.weights_dir) / SIZE / ("regressor.pt" if regression else "classifier.pt")
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model_.models[task].load_state_dict(state, assign=True)
        self.model_.eval()
        self.stypes_ = sdm.infer_stypes(X)
        self.x_ = sdm.TableTensor.from_pandas(df=X, stypes=self.stypes_, device=self.device)
        self.y_ = sdm.TableTensor.from_pandas(df=pd.DataFrame({TARGET: y}), device=self.device,
                                              stypes={TARGET: "numerical" if regression else "categorical"})
        if not regression:
            self.classes_ = np.unique(y)
        return self

    def _forward(self, X):
        query = sdm.TableTensor.from_pandas(df=X, stypes=self.stypes_, device=self.device)
        generator = torch.Generator(self.device).manual_seed(self.seed)
        cuda = torch.device(self.device).type == "cuda"
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=cuda):
            return self.model_(x_context=self.x_, y_context=self.y_, x_query=query, num_estimators=NUM_ESTIMATORS,
                               generator=generator)

    def predict(self, X):
        if self.task == "regression":
            return self._forward(X).numerical.float().mean(dim=-1).cpu().numpy()
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        out = self._forward(X)
        columns = list(out.columns[sdm.Stype.numerical])  # one column per class, named by its value
        return out.numerical.float().cpu().numpy()[:, [columns.index(str(c)) for c in self.classes_]]
