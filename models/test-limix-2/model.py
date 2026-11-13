"""LimiX-2 (Stable AI) with its default settings, the checkpoint loaded from weights_dir.

A 400M-parameter in-context model pretrained on synthetic data, one checkpoint for every task. The package
defaults: the packaged LimiX-2 configuration of the task (config/reg_default_noretrieval_v2.json or
config/cls_default_noretrieval_v2.json: an ensemble of preprocessing pipelines, no retrieval). The weights'
licence (StableAI LimiX Non-Commercial License v1.0) allows non-commercial use, benchmarking included.
"""

from pathlib import Path

import config as limix_configs  # the package's configuration files
import numpy as np
import torch
from inference.predictor import LimiXPredictor

from strable_leaderboard.model import StrableModel

CHECKPOINT = "LimiX-2.ckpt"
CONFIGS = {"Regression": "reg_default_noretrieval_v2.json", "Classification": "cls_default_noretrieval_v2.json"}


class Model(StrableModel):
    end_to_end = False  # its predict encodes strings as categories: run with each string encoder

    def fit(self, X, y):
        # In-context: the training rows are given again with every prediction
        self.kind_ = "Regression" if self.task == "regression" else "Classification"
        config = Path(limix_configs.__file__).parent / CONFIGS[self.kind_]
        # The pinned file, never a download, on the board's device
        self.model_ = LimiXPredictor(device=torch.device(self.device), model_path=str(Path(self.weights_dir) / CHECKPOINT),
                                     inference_config=str(config), seed=self.seed, preprocess_num_jobs=self.n_jobs)
        self.X_, self.y_ = X, np.asarray(y)
        if self.kind_ == "Classification":
            self.classes_ = np.unique(self.y_)  # its probabilities follow the sorted labels (LabelEncoder)
        return self

    def predict(self, X):
        if self.kind_ == "Regression":
            return np.asarray(self.model_.predict(self.X_, self.y_, X, task_type="Regression"))
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        return np.asarray(self.model_.predict(self.X_, self.y_, X, task_type="Classification"))
