"""The sandboxed child of ``strable_leaderboard.run``: fit one model, write its predictions, nothing else.

    python -m strable_leaderboard.fit_predict <model folder> <inputs> <outputs> --task regression [--warm-up]

Reads ``X_train.parquet``, ``y_train.npy`` and ``X_test.parquet`` from <inputs> (read-only in the
sandbox) and writes ``pred.npy`` (regression) or ``proba.npy`` and ``classes.npy``
(classification) to <outputs>. It never sees the test labels, the dataset or the folds, and it
neither times nor scores itself: the parent does both. ``--warm-up`` fits on a few rows, at least
one per class, predicts eight, and writes nothing, so that the parent can measure the start-up
overhead: imports, CUDA start-up, weight loading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .model import load_model_class
from .splits import SEED


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("--task", required=True, choices=["regression", "binary", "multiclass"])
    parser.add_argument("--strings", action="store_true", help="the inputs are the raw table")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--warm-up", action="store_true")
    args = parser.parse_args(argv)
    cls = load_model_class(args.folder)
    if cls.handles_strings != args.strings:
        sys.exit("handles_strings differs between model.yaml and model.py")
    X_train, X_test = pd.read_parquet(args.inputs / "X_train.parquet"), pd.read_parquet(args.inputs / "X_test.parquet")
    y_train = np.load(args.inputs / "y_train.npy")
    model = cls(task=args.task, seed=SEED, device=args.device, n_jobs=args.n_jobs, weights_dir=args.weights_dir)
    regression = args.task == "regression"
    if args.warm_up:
        rows = np.arange(min(64, len(y_train)))
        if not regression:
            rows = np.union1d(rows, np.unique(y_train, return_index=True)[1])
        try:
            model.fit(X_train.iloc[rows], y_train[rows])
            (model.predict if regression else model.predict_proba)(X_test.iloc[:8])
        except Exception as exc:  # the overhead is then smaller, so the timed run counts more, never less
            print(f"warm-up: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0
    model.fit(X_train, y_train)
    if regression:
        np.save(args.outputs / "pred.npy", np.asarray(model.predict(X_test), dtype=float))
    else:
        np.save(args.outputs / "proba.npy", np.asarray(model.predict_proba(X_test), dtype=float))
        np.save(args.outputs / "classes.npy", np.asarray(model.classes_))
    return 0


if __name__ == "__main__":
    sys.exit(main())
