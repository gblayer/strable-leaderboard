import numpy as np
import pandas as pd
import pytest

from strable_leaderboard.metrics import expected_folds, fold_score, pareto_front, per_1k


def test_total_time_per_1k_samples():
    runner = pd.DataFrame({"fit_s": [1.0], "predict_s": [0.5], "total_s": [np.nan], "n_train": [1000], "n_test": [500]})
    assert per_1k(runner)["total_per_1k"].iloc[0] == pytest.approx(1.0)  # (1.0 + 0.5) s over 1,500 rows
    paper = runner.assign(fit_s=None, predict_s=None, total_s=3.0)  # rows imported from the paper
    assert per_1k(paper)["total_per_1k"].iloc[0] == pytest.approx(2.0)


def test_expected_folds():
    tables = {"r1": {"task": "regression"}, "c1": {"task": "binary"}, "c2": {"task": "multiclass"}}
    assert expected_folds(tables) == {"both": 9, "regression": 3, "classification": 6}


def test_pareto_front_keeps_only_undominated_pipelines():
    board = pd.DataFrame({"score": [0.6, 0.7, 0.65, 0.8], "total_per_1k": [0.1, 1.0, 2.0, 10.0]}, index=list("abcd"))
    assert pareto_front(board) == ["a", "b", "d"]


def test_fold_scores():
    assert fold_score("regression", [1, 2, 3], y_pred=[1, 2, 3]) == pytest.approx(1.0)
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert fold_score("binary", np.array(["no", "yes"]), proba=proba, classes=np.array(["no", "yes"])) == 1.0


def test_multiclass_auc_applies_the_papers_softmax():
    """salts src/utils_evaluation.py: reshape_pred_output softmaxes (n, K>2) outputs, then return_score
    computes roc_auc_score(multi_class="ovr", average="macro")."""
    from scipy.special import softmax
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 20)
    proba = rng.dirichlet([0.3, 0.3, 0.3], size=60)
    proba[y == 1, 1] += 0.2
    proba /= proba.sum(axis=1, keepdims=True)
    paper = roc_auc_score(y, softmax(proba, axis=1), multi_class="ovr", average="macro")
    assert fold_score("multiclass", y, proba=proba, classes=np.array([0, 1, 2])) == pytest.approx(paper)
    as_returned = roc_auc_score(y, proba, multi_class="ovr", average="macro")
    assert abs(paper - as_returned) > 1e-3  # the softmax changes the score
    labels = np.array(["a", "b", "c"])  # original labels, sorted
    assert fold_score("multiclass", labels[y], proba=proba, classes=labels) == pytest.approx(paper)

