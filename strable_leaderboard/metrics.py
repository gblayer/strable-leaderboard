"""Scores, fold counts and the Pareto front.

Per fold: R² for regression, ROC-AUC for classification (one-vs-rest, macro average of the
softmax of the probabilities for multiclass), as in the paper. Per pipeline and view: the median
over all fold rows of the view, raw, as in the paper's Figure 4a, so a negative R² stays a fold
row like any other. Time is the total run time per 1,000 samples, also the median over fold rows:
per fold, encoding plus training plus prediction seconds (or the paper's run time) divided by the
table's rows, times 1,000. No per-table normalization, so a pipeline's numbers never change when
others join.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import r2_score, roc_auc_score

from .splits import N_OUTER

VIEWS = {
    "both": ("regression", "binary", "multiclass"),
    "regression": ("regression",),
    "classification": ("binary", "multiclass"),
}
VARIANTS = ("default", "tuned")


def expected_folds(tables: dict) -> dict[str, int]:
    """Fold rows per view of a pipeline that covers every table and outer fold of the manifest's ``tables``."""
    return {view: N_OUTER * sum(m["task"] in tasks for m in tables.values()) for view, tasks in VIEWS.items()}


def fold_score(task: str, y_true, y_pred=None, proba=None, classes=None) -> float:
    """R² for regression; ROC-AUC from probabilities for classification, as in the paper.

    ``proba`` has one column per class, in the order of ``classes`` (sorted labels). Binary: the
    positive-class column. Multiclass: one-vs-rest macro AUC of ``softmax(proba, axis=1)``.
    """
    if task == "regression":
        return float(r2_score(y_true, y_pred))
    proba = np.asarray(proba, dtype=float)
    if task == "binary":
        return float(roc_auc_score(np.asarray(y_true) == classes[1], proba[:, 1]))
    # The paper applies a softmax to the probabilities before the multiclass AUC
    # (reshape_pred_output in src/utils_evaluation.py); kept so re-runs match the paper's rows.
    return float(roc_auc_score(y_true, softmax(proba, axis=1), multi_class="ovr", average="macro", labels=classes))


def checked_score(task: str, y_train, y_test, pred=None, proba=None, classes=None) -> float:
    """``fold_score`` after checking the predictions' shape and range; raises ValueError."""
    n = len(y_test)
    if task == "regression":
        pred = np.asarray(pred, dtype=float)
        if pred.shape != (n,):
            raise ValueError(f"predict returned shape {pred.shape}, expected ({n},)")
        if not np.isfinite(pred).all():
            raise ValueError("predict returned NaN or infinite values")
        return fold_score(task, y_test, y_pred=pred)
    classes, proba = np.asarray(classes), np.asarray(proba, dtype=float)
    if classes.tolist() != sorted(np.unique(y_train).tolist()):
        raise ValueError("classes_ must be the sorted training labels")
    if proba.shape != (n, len(classes)):
        raise ValueError(f"predict_proba returned shape {proba.shape}, expected ({n}, {len(classes)})")
    if not (np.isfinite(proba).all() and (proba >= 0).all()):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(proba.sum(axis=1), 1, atol=1e-4):
        raise ValueError("probabilities must sum to 1 per row")
    return fold_score(task, y_test, proba=proba, classes=classes)


def per_1k(results: pd.DataFrame) -> pd.DataFrame:
    """Add the total run time (``total``, s), the table rows (``n_rows``) and the total run time per
    1,000 samples (``total_per_1k``) to fold rows.

    ``results`` has ``n_train, n_test`` and ``total_s``, or ``fit_s`` and ``predict_s`` whose sum
    is the total.
    """
    split = results["fit_s"] + results["predict_s"] if "fit_s" in results else np.nan
    total = results["total_s"].fillna(split) if "total_s" in results else split
    n_rows = results["n_train"] + results["n_test"]
    return results.assign(total=total, n_rows=n_rows, total_per_1k=total / n_rows * 1000)


def pareto_front(board: pd.DataFrame) -> list[str]:
    """Pipelines that no other pipeline beats on both score and total run time per 1K samples.

    ``board`` has one row per pipeline with ``score`` and ``total_per_1k``. Sort by
    ``total_per_1k``, then keep a pipeline when its score is strictly higher than every faster one.
    """
    ordered = board.sort_values(["total_per_1k", "score"], ascending=[True, False])
    front, best = [], -np.inf
    for name, row in ordered.iterrows():
        if row["score"] > best + 1e-12:
            front.append(name)
            best = row["score"]
    return front
