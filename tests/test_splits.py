import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strable_leaderboard.splits import N_INNER, N_OUTER, fold_table, inner_splits, outer_splits, splits_from_table

SPLITS = Path(__file__).resolve().parent.parent / "splits"


def test_outer_folds_partition_rows():
    y = np.random.default_rng(0).normal(size=101)
    folds = outer_splits(y, "regression")
    assert len(folds) == N_OUTER
    tests = np.concatenate([te for _, te in folds])
    assert sorted(tests.tolist()) == list(range(101))


def test_outer_classification_is_stratified():
    y = np.array([0] * 60 + [1] * 30)
    for _, te in outer_splits(y, "binary"):
        assert (y[te] == 1).sum() == 10


def test_splits_are_deterministic():
    y = np.random.default_rng(1).normal(size=50)
    a = [te.tolist() for _, te in outer_splits(y, "regression")]
    b = [te.tolist() for _, te in outer_splits(y, "regression")]
    assert a == b


def test_fold_table_round_trip():
    y = np.array(list("abc") * 40)
    table = fold_table(y, "multiclass")
    for k, (tr, te) in enumerate(outer_splits(y, "multiclass")):
        train_idx, test_idx, inner = splits_from_table(table, k)
        assert train_idx.tolist() == tr.tolist()
        assert test_idx.tolist() == te.tolist()
        expected = inner_splits(y[tr], "multiclass")
        assert len(inner) == N_INNER
        for (fit_a, val_a), (fit_b, val_b) in zip(inner, expected):
            assert sorted(val_a.tolist()) == sorted(val_b.tolist())
            assert sorted(fit_a.tolist()) == sorted(fit_b.tolist())


@pytest.mark.skipif(not (SPLITS / "manifest.json").exists(), reason="splits not exported")
def test_exported_splits_match_manifest():
    manifest = json.loads((SPLITS / "manifest.json").read_text())
    assert len(manifest["tables"]) == 108
    for name, info in manifest["tables"].items():
        folds = pd.read_parquet(SPLITS / f"{name}.parquet")
        assert len(folds) == info["n_rows"]
        assert set(folds["outer_fold"].unique()) == {0, 1, 2}
        for k in range(N_OUTER):
            in_test = folds["outer_fold"] == k
            assert (folds.loc[in_test, f"inner_fold_{k}"] == -1).all()
            assert set(folds.loc[~in_test, f"inner_fold_{k}"].unique()) == set(range(N_INNER))
