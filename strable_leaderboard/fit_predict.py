"""The sandboxed child of ``strable_leaderboard.run``: fit one model, write its predictions, nothing else.

    python -m strable_leaderboard.fit_predict <model folder> <inputs> <outputs> --task regression [--warm-up]

Reads ``X_train.parquet``, ``y_train.npy`` and ``X_test.parquet`` from <inputs> (read-only in the
sandbox) and writes ``pred.npy`` (regression) or ``proba.npy`` and ``classes.npy``
(classification) to <outputs>. It never sees the test labels, the dataset or the folds, and it
neither times nor scores itself: the parent does both. ``--warm-up`` fits on a few rows, at least
one per class, predicts eight, and writes nothing, so that the parent can measure the start-up
overhead: imports, CUDA start-up, weight loading. BLAS, OpenMP and torch threads are capped to
``--n-jobs`` before the model is built (``limit_threads``).

``--api``: the model runs on its provider's servers (``hardware: api``). The warm-up only builds it and
sends nothing, so the time keeps the request, the network and the provider's work. A model that sets
``self.api_version`` (the version its provider reports) gets it written to ``api_version.txt``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .model import load_model_class
from .splits import SEED

# Libraries size their thread pools from the machine's cores, not the job's CPUs (tabicl 2.x even
# takes max(cores, n_jobs)), so the runner sets these to n_jobs in the child's environment.
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def thread_env(n_jobs: int) -> dict[str, str]:
    return {v: str(n_jobs) for v in THREAD_VARS}


def limit_threads(n_jobs: int) -> None:
    """Cap threads to ``n_jobs``: the variables for libraries loaded from now on, threadpoolctl for
    those already loaded, and torch's own setting if it is imported."""
    from threadpoolctl import threadpool_limits

    os.environ.update(thread_env(n_jobs))
    threadpool_limits(n_jobs)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(n_jobs)


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
    parser.add_argument("--api", action="store_true", help="the model calls its provider's API")
    args = parser.parse_args(argv)
    cls = load_model_class(args.folder)
    if cls.end_to_end != args.strings:
        sys.exit("end_to_end differs between model.yaml and model.py")
    limit_threads(args.n_jobs)
    X_train, X_test = pd.read_parquet(args.inputs / "X_train.parquet"), pd.read_parquet(args.inputs / "X_test.parquet")
    y_train = np.load(args.inputs / "y_train.npy")
    model = cls(task=args.task, seed=SEED, device=args.device, n_jobs=args.n_jobs, weights_dir=args.weights_dir)
    regression = args.task == "regression"
    if args.warm_up and args.api:
        return 0
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
    if getattr(model, "api_version", None):
        (args.outputs / "api_version.txt").write_text(str(model.api_version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
