import numpy as np
import pandas as pd
import pytest

from strable_leaderboard.codenames import codename
from strable_leaderboard.metrics import fold_score, leaderboard, pareto_front, table_scores


def _results():
    rows = []
    # pipeline A: R² 0.8, 0.7 and a failed table (-3.0); AUC 0.9
    for table, task, score in [("r1", "regression", 0.8), ("r2", "regression", 0.7), ("r3", "regression", -3.0), ("c1", "binary", 0.9)]:
        for fold in range(3):
            rows.append(dict(pipeline="A", table=table, task=task, fold=fold, score=score,
                             fit_s=1.0, predict_s=0.5, n_train=1000, n_test=500))
    return pd.DataFrame(rows)


def test_negative_r2_counts_as_zero_after_fold_mean():
    t = table_scores(_results()).set_index("table")
    assert t.loc["r3", "score"] == 0.0
    assert t.loc["r1", "score"] == pytest.approx(0.8)


def test_leaderboard_views():
    both = leaderboard(_results(), "both", n_tables=4)
    assert both.loc["A", "score"] == pytest.approx((0.8 + 0.7 + 0.0 + 0.9) / 4)
    assert bool(both.loc["A", "complete"])
    reg = leaderboard(_results(), "regression")
    assert reg.loc["A", "score"] == pytest.approx(0.5)
    assert reg.loc["A", "fit_per_1k"] == pytest.approx(1.0)
    assert reg.loc["A", "predict_per_1k"] == pytest.approx(1.0)


def test_pareto_front_keeps_only_undominated_pipelines():
    board = pd.DataFrame({"score": [0.6, 0.7, 0.65, 0.8], "fit_per_1k": [0.1, 1.0, 2.0, 10.0]}, index=list("abcd"))
    assert pareto_front(board) == ["a", "b", "d"]
    assert pareto_front(board, exclude=["b"]) == ["a", "c", "d"]


def test_fold_scores():
    assert fold_score("regression", [1, 2, 3], y_pred=[1, 2, 3]) == pytest.approx(1.0)
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert fold_score("binary", np.array(["no", "yes"]), proba=proba, classes=np.array(["no", "yes"])) == 1.0


def test_codename_format_and_uniqueness():
    names = set()
    for _ in range(50):
        names.add(codename(taken=names))
    assert len(names) == 50
    assert all(n.count("-") == 1 for n in names)
