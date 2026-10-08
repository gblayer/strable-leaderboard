"""Scores, aggregation and the Pareto front.

Per fold: R² for regression, ROC-AUC for classification (one-vs-rest, macro average for
multiclass), as in the paper. Per pipeline and view: the median over all fold rows of the view,
raw, as in the paper's Figure 4a, so a negative R² stays a fold row like any other. Times are
seconds per 1,000 rows, also the median over fold rows: fit per 1,000 training rows, predict
per 1,000 test rows, and the total (fit plus predict, or the paper's single time) per 1,000 table
rows. No per-table normalization, so a pipeline's numbers never change when others join.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

N_FOLDS = 3
VIEWS = {
    "both": ("regression", "binary", "multiclass"),
    "regression": ("regression",),
    "classification": ("binary", "multiclass"),
}


def fold_score(task: str, y_true, y_pred=None, proba=None, classes=None) -> float:
    """R² for regression; ROC-AUC from probabilities for classification.

    ``proba`` has one column per class, in the order of ``classes`` (sorted labels).
    """
    if task == "regression":
        return float(r2_score(y_true, y_pred))
    proba = np.asarray(proba, dtype=float)
    if task == "binary":
        return float(roc_auc_score(np.asarray(y_true) == classes[1], proba[:, 1]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes))


def per_1k(results: pd.DataFrame) -> pd.DataFrame:
    """Add fit, predict and total seconds per 1,000 rows to fold rows.

    ``results`` has ``fit_s, predict_s, n_train, n_test`` and optionally ``total_s``; a row
    without ``total_s`` totals fit and predict.
    """
    total = results["total_s"].fillna(results["fit_s"] + results["predict_s"]) if "total_s" in results else (
        results["fit_s"] + results["predict_s"])
    return results.assign(
        fit_per_1k=results["fit_s"] / results["n_train"] * 1000,
        predict_per_1k=results["predict_s"] / results["n_test"] * 1000,
        total_per_1k=total / (results["n_train"] + results["n_test"]) * 1000,
    )


def leaderboard(results: pd.DataFrame, view: str = "both", n_tables: int | None = None) -> pd.DataFrame:
    """One row per pipeline: median score and median times per 1K rows over the fold rows of the view.

    ``results`` has one row per (pipeline, table, fold) with columns
    ``pipeline, table, task, fold, score, fit_s, predict_s, n_train, n_test`` and optionally
    ``total_s``. ``n_folds`` counts the fold rows; with ``n_tables``, a pipeline that misses a
    table or a fold of the view is flagged ``complete=False``.
    """
    r = per_1k(results[results["task"].isin(VIEWS[view])])
    board = r.groupby("pipeline").agg(
        score=("score", "median"),
        fit_per_1k=("fit_per_1k", "median"),
        predict_per_1k=("predict_per_1k", "median"),
        total_per_1k=("total_per_1k", "median"),
        n_folds=("score", "size"),
    )
    if n_tables is not None:
        board["complete"] = board["n_folds"] == n_tables * N_FOLDS
    return board.sort_values("score", ascending=False)


def pareto_front(board: pd.DataFrame, time_col: str = "total_per_1k", exclude=()) -> list[str]:
    """Pipelines that no other pipeline beats on both score and time.

    Sort by time, then keep a pipeline when its score is strictly higher than every faster
    one. ``exclude`` removes pipelines from the front, if ever needed.
    """
    ordered = board.drop(index=[p for p in exclude if p in board.index])
    ordered = ordered.sort_values([time_col, "score"], ascending=[True, False])
    front, best = [], -np.inf
    for name, row in ordered.iterrows():
        if row["score"] > best + 1e-12:
            front.append(name)
            best = row["score"]
    return front
