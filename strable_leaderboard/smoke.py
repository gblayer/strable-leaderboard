"""Smoke test for one model folder: outer fold 0 of three small tables, one per task.

    python -m strable_leaderboard.smoke models/<name>
    python -m strable_leaderboard.smoke models/<name> --data-dir /path/to/STRABLE-benchmark

It checks the model card, loads ``Model`` from ``model.py``, encodes strings with Tf-Idf
when the model needs numbers, and checks the shapes and ranges of the predictions.
It does not measure leaderboard scores: the full run happens on the cluster.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

from .data import load_table
from .metrics import fold_score
from .model import StrableModel
from .splits import SEED, outer_splits

SMOKE_TABLES = {
    "regression": "china-overseas-finance-inventory",  # 550 rows
    "binary": "animalandveterinary-event",  # 1,345 rows
    "multiclass": "drug-ndc",  # 583 rows, 4 classes
}
REQUIRED = ["name", "version", "handles_strings", "tasks", "access", "hardware", "package", "checkpoint", "licence",
            "trained_on_strable"]
ACCESS = {"open", "closed", "api"}
HARDWARE = {"gpu", "cpu"}
TASKS = {"regression", "binary", "multiclass"}


def load_card(folder: Path) -> dict:
    """Read and validate model.yaml."""
    card = yaml.safe_load((folder / "model.yaml").read_text())
    missing = [k for k in REQUIRED if k not in card]
    if missing:
        raise ValueError(f"model.yaml misses {missing}")
    if card["access"] not in ACCESS:
        raise ValueError(f"access must be one of {sorted(ACCESS)}")
    if card["hardware"] not in HARDWARE:
        raise ValueError(f"hardware must be one of {sorted(HARDWARE)}")
    if not card["tasks"] or not set(card["tasks"]) <= TASKS:
        raise ValueError(f"tasks must be a non-empty subset of {sorted(TASKS)}")
    ckpt = card["checkpoint"]
    if ckpt is not None:
        rev = str(ckpt.get("revision", ""))
        if not ckpt.get("repo") or len(rev) < 7 or any(c not in "0123456789abcdef" for c in rev.lower()):
            raise ValueError("checkpoint needs a repo and a commit hash as revision")
    return card


def load_model_class(folder: Path):
    spec = importlib.util.spec_from_file_location(f"strable_model_{folder.name}", folder / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "Model", None)
    if cls is None or not issubclass(cls, StrableModel):
        raise TypeError("model.py must define `class Model(StrableModel)`")
    return cls


def check_one(model_cls, card: dict, table: str, data_dir=None) -> dict:
    X, y, task = load_table(table, data_dir=data_dir)
    train_idx, test_idx = outer_splits(y, task)[0]
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    if not model_cls.handles_strings:
        from .encoders import tfidf_encoder

        encoder = tfidf_encoder()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories")
            X_train = encoder.fit_transform(X_train, y_train)
            X_test = encoder.transform(X_test)

    model = model_cls(task=task, seed=SEED, device="cpu", n_jobs=1)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if task == "regression":
        pred = np.asarray(model.predict(X_test), dtype=float)
        predict_s = time.perf_counter() - t0
        assert pred.shape == (len(test_idx),), f"predict returned shape {pred.shape}"
        assert np.isfinite(pred).all(), "predict returned NaN or infinite values"
        score = fold_score(task, y_test, y_pred=pred)
    else:
        proba = np.asarray(model.predict_proba(X_test), dtype=float)
        predict_s = time.perf_counter() - t0
        classes = np.asarray(model.classes_)
        assert list(classes) == sorted(np.unique(y_train).tolist()), "classes_ must be the sorted training labels"
        assert proba.shape == (len(test_idx), len(classes)), f"predict_proba returned shape {proba.shape}"
        assert np.isfinite(proba).all() and (proba >= 0).all(), "probabilities must be finite and non-negative"
        assert np.allclose(proba.sum(axis=1), 1, atol=1e-4), "probabilities must sum to 1 per row"
        score = fold_score(task, y_test, proba=proba, classes=classes)
    return {"table": table, "task": task, "score": score, "fit_s": fit_s, "predict_s": predict_s,
            "n_train": len(train_idx), "n_test": len(test_idx)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="models/<name>")
    parser.add_argument("--data-dir", type=Path, default=None, help="local copy of the dataset (optional)")
    args = parser.parse_args(argv)

    card = load_card(args.folder)
    model_cls = load_model_class(args.folder)
    if bool(card["handles_strings"]) != bool(model_cls.handles_strings):
        raise ValueError("handles_strings differs between model.yaml and model.py")

    failures = 0
    print(f"{card['name']} {card['version']}  ({'end-to-end' if model_cls.handles_strings else 'numeric-only, Tf-Idf encoder'})")
    for task, table in SMOKE_TABLES.items():
        if task not in card["tasks"]:
            print(f"  skip  {task:<10} not in model.yaml tasks")
            continue
        try:
            r = check_one(model_cls, card, table, data_dir=args.data_dir)
            print(f"  ok    {task:<10} {table:<34} score {r['score']:.3f}  fit {r['fit_s']:.2f}s  predict {r['predict_s']:.2f}s")
        except Exception as exc:  # report every failure, then exit non-zero
            failures += 1
            print(f"  FAIL  {task:<10} {table:<34} {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
