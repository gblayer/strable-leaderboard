"""Scores, aggregation and the Pareto front.

Per fold: R² for regression, ROC-AUC for classification (one-vs-rest, macro average for
multiclass). Per table: mean over the 3 outer folds, then an R² below 0 counts as 0.
Per pipeline: mean of the table scores over the tables of the chosen view, with no
per-table normalization, so a pipeline's score never changes when others join.
Times are seconds per 1,000 rows, averaged over folds and then summarized by the median
over tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

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


def table_scores(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (pipeline, table).

    ``results`` has one row per (pipeline, table, fold) with columns
    ``pipeline, table, task, fold, score, fit_s, predict_s, n_train, n_test``.
    """
    r = results.assign(
        fit_per_1k=results["fit_s"] / results["n_train"] * 1000,
        predict_per_1k=results["predict_s"] / results["n_test"] * 1000,
    )
    t = r.groupby(["pipeline", "table", "task"], as_index=False).agg(
        score=("score", "mean"),
        n_folds=("fold", "nunique"),
        fit_per_1k=("fit_per_1k", "mean"),
        predict_per_1k=("predict_per_1k", "mean"),
    )
    is_reg = t["task"] == "regression"
    t.loc[is_reg, "score"] = t.loc[is_reg, "score"].clip(lower=0.0)
    return t


def leaderboard(results: pd.DataFrame, view: str = "both", n_tables: int | None = None) -> pd.DataFrame:
    """One row per pipeline: mean score, median fit and predict time per 1K rows.

    With ``n_tables``, pipelines that miss a table or a fold are flagged ``complete=False``.
    """
    t = table_scores(results)
    t = t[t["task"].isin(VIEWS[view])]
    board = t.groupby("pipeline").agg(
        score=("score", "mean"),
        fit_per_1k=("fit_per_1k", "median"),
        predict_per_1k=("predict_per_1k", "median"),
        n_tables=("table", "nunique"),
        min_folds=("n_folds", "min"),
    )
    if n_tables is not None:
        board["complete"] = (board["n_tables"] == n_tables) & (board["min_folds"] == 3)
    return board.sort_values("score", ascending=False)


def pareto_front(board: pd.DataFrame, time_col: str = "fit_per_1k", exclude=()) -> list[str]:
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
