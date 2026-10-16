"""Checks that the runner gives a model the same score as calling its fit and predict directly, rejects malformed
predictions and limits CPU threads."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strable_leaderboard import run as run_module
from strable_leaderboard.encoders import encode
from strable_leaderboard.fit_predict import THREAD_VARS
from strable_leaderboard.metrics import checked_score, fold_score
from strable_leaderboard.model import load_model_class
from strable_leaderboard.splits import SEED, splits_from_table

ROOT = Path(__file__).resolve().parent.parent
TABLES = json.loads((ROOT / "splits" / "manifest.json").read_text())["tables"]


def fake_table(table):
    n, task = TABLES[table]["n_rows"], TABLES[table]["task"]
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=n), "s": rng.choice(["red apple", "green pear", "blue plum"], size=n)})
    y = rng.normal(size=n) + X["x"].to_numpy() if task == "regression" else rng.integers(0, 3, size=n)
    return X, y, task


@pytest.mark.parametrize("table", ["china-overseas-finance-inventory", "drug-ndc"])
def test_child_process_scores_like_in_process(table, monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)  # the child finds the package as `python -m` does
    monkeypatch.setattr(run_module, "load_table", lambda name, data_dir=None: fake_table(name))
    ridge = ROOT / "models" / "ridge"
    row = run_module.run(ridge, "tfidf", table, 0, None, None, "cpu", 1, sandbox=False)

    X, y, task = fake_table(table)
    train, test, _ = splits_from_table(pd.read_parquet(ROOT / "splits" / f"{table}.parquet"), 0)
    X_train, X_test, _ = encode(X.iloc[train], X.iloc[test], "tfidf", table)
    model = load_model_class(ridge)(task=task, seed=SEED).fit(X_train, y[train])
    if task == "regression":
        expected = fold_score(task, y[test], y_pred=model.predict(X_test))
    else:
        expected = fold_score(task, y[test], proba=model.predict_proba(X_test), classes=model.classes_)
    assert row["score"] == expected
    assert row["fit_s"] is None and row["predict_s"] is None and row["total_s"] > 0


def test_predictions_are_checked():
    y_train, y_test = np.array([0, 1, 2, 0]), np.array([0, 1, 2])
    good = np.array([[0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    assert checked_score("multiclass", y_train, y_test, proba=good, classes=[0, 1, 2]) == 1.0
    for proba, classes, msg in [(good, [0, 2, 1], "sorted"), (good[:1], [0, 1, 2], "shape"),
                                (good * 2, [0, 1, 2], "sum to 1"), (-good, [0, 1, 2], "non-negative")]:
        with pytest.raises(ValueError, match=msg):
            checked_score("multiclass", y_train, y_test, proba=proba, classes=classes)
    with pytest.raises(ValueError, match="NaN"):
        checked_score("regression", y_train, y_test, pred=[np.nan, 1.0, 2.0])


def test_threads_are_capped_to_n_jobs():
    code = ("import os, numpy; from threadpoolctl import threadpool_info; "
            "from strable_leaderboard.fit_predict import limit_threads; limit_threads(2); "
            "print(max(p['num_threads'] for p in threadpool_info()), os.environ['OMP_NUM_THREADS'])")
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT, env=env)
    assert out.stdout.split() == ["2", "2"]
