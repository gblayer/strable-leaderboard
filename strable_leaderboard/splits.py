"""Outer and inner cross-validation, exactly as in the STRABLE paper.

Outer loop: 3 folds, one repetition, seed 1234, on the row order of ``data.parquet``.

- regression: ``RepeatedKFold(n_splits=3, n_repeats=1, random_state=1234)``
- classification: ``RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1234)``

Inner loop: the same classes with ``n_splits=8``, applied to each outer training set.
It is used for early stopping (the first inner fold is the validation set) and, later,
for tuning. Default configurations do not tune.

Source: soda-inria/strable, ``set_split_cv`` in ``src/utils_evaluation.py`` and the inner
``cv`` in ``scripts/evaluate_scripts/script_evaluate.py``.
The precomputed folds for all 108 tables are in ``splits/``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

SEED = 1234
N_OUTER = 3
N_INNER = 8


def _cv(task: str, n_splits: int):
    if task == "regression":
        return RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=SEED)
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=SEED)


def _split(task: str, n_splits: int, y) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    if task != "regression":
        # The paper label-encodes classification targets before splitting. Folds depend only
        # on which rows share a class, so this changes nothing, but it keeps sklearn from
        # reading labels such as Yelp's 1.0 to 5.0 stars as a continuous target.
        y = np.unique(y, return_inverse=True)[1]
    with warnings.catch_warnings():
        # Rare classes with fewer members than folds trigger a warning; the split is still valid.
        warnings.filterwarnings("ignore", message="The least populated class")
        return list(_cv(task, n_splits).split(np.zeros(len(y)), y))


def outer_splits(y, task: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The 3 outer folds as ``(train_idx, test_idx)``, positions in the full table."""
    return _split(task, N_OUTER, y)


def inner_splits(y_train, task: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The 8 inner folds of one outer training set as ``(fit_idx, val_idx)``.

    Positions are relative to the outer training set, in its row order.
    """
    return _split(task, N_INNER, y_train)


def fold_table(y, task: str) -> pd.DataFrame:
    """One row per table row, with the fold ids stored in ``splits/<table>.parquet``.

    - ``outer_fold``: the outer fold (0, 1 or 2) in which the row is a test row.
    - ``inner_fold_k`` for k in 0..2: inside outer training set k, the inner fold (0 to 7)
      in which the row is a validation row, or -1 when the row is in outer test fold k.
    """
    n = len(y)
    y = np.asarray(y)
    out = {"outer_fold": np.full(n, -1, dtype=np.int8)}
    for k, (train_idx, test_idx) in enumerate(outer_splits(y, task)):
        out["outer_fold"][test_idx] = k
        inner = np.full(n, -1, dtype=np.int8)
        for j, (_, val_idx) in enumerate(inner_splits(y[train_idx], task)):
            inner[train_idx[val_idx]] = j
        out[f"inner_fold_{k}"] = inner
    return pd.DataFrame(out)


def splits_from_table(folds: pd.DataFrame, outer: int) -> tuple[np.ndarray, np.ndarray, list]:
    """Rebuild outer fold ``outer`` and its inner folds from a ``splits/<table>.parquet`` frame.

    Returns ``(train_idx, test_idx, inner)`` where ``inner`` is the list of 8
    ``(fit_idx, val_idx)`` pairs relative to the training set, as ``inner_splits`` gives.
    """
    test_idx = np.flatnonzero(folds["outer_fold"].to_numpy() == outer)
    train_idx = np.flatnonzero(folds["outer_fold"].to_numpy() != outer)
    inner_ids = folds[f"inner_fold_{outer}"].to_numpy()[train_idx]
    inner = [(np.flatnonzero(inner_ids != j), np.flatnonzero(inner_ids == j)) for j in range(N_INNER)]
    return train_idx, test_idx, inner
