import pandas as pd

from strable_leaderboard.check_results import check

MANIFEST = {"tables": {
    "r1": {"task": "regression", "n_rows": 30, "test_rows_per_outer_fold": [10, 10, 10]},
    "c1": {"task": "binary", "n_rows": 30, "test_rows_per_outer_fold": [10, 10, 10]},
}}


def _rows(encoder="tfidf", tables=("r1", "c1"), folds=(0, 1, 2)):
    rows = []
    for table in tables:
        for fold in folds:
            rows.append(dict(model="ridge", encoder=encoder, variant="default", table=table, task=MANIFEST["tables"][table]["task"],
                             fold=fold, score=0.5, fit_s=1.0, predict_s=0.2, total_s=None, n_train=20, n_test=10, hardware="cpu",
                             package="scikit-learn==1.9.1", checkpoint_revision="", protocol_version="0.1",
                             date="2026-11-15"))
    return pd.DataFrame(rows)


def test_complete_file_has_no_problems(tmp_path):
    path = tmp_path / "ridge.csv"
    _rows().to_csv(path, index=False)
    problems, coverage = check(path, MANIFEST)
    assert problems == []
    assert any(line.endswith("both            6/6 folds") for line in coverage)


def test_problems_are_reported(tmp_path):
    df = _rows(folds=(0, 1))
    df.loc[0, "score"] = 1.5  # an R² above 1
    df.loc[3, "n_test"] = 11  # disagrees with the manifest
    path = tmp_path / "other.csv"  # file name differs from the model column
    df.to_csv(path, index=False)
    problems, coverage = check(path, MANIFEST)
    assert any("model must be 'other'" in p for p in problems)
    assert any("R² cannot exceed 1" in p for p in problems)
    assert any("n_train or n_test" in p for p in problems)
    assert any(line.endswith("both            4/6 folds incomplete") for line in coverage)
    df.loc[1, "score"] = float("nan")  # a fold without a score: a problem, and not a covered fold
    df.to_csv(path, index=False)
    problems, coverage = check(path, MANIFEST)
    assert "1 rows without a score" in problems
    assert any(line.endswith("both            3/6 folds incomplete") for line in coverage)


def test_paper_rows_need_only_a_total_time(tmp_path):
    df = _rows()
    df["fit_s"] = None; df["predict_s"] = None; df["total_s"] = 1.5
    path = tmp_path / "ridge.csv"
    df.to_csv(path, index=False)
    assert check(path, MANIFEST)[0] == []
    df["total_s"] = None
    df.to_csv(path, index=False)
    assert any("needs fit_s and predict_s, or total_s" in p for p in check(path, MANIFEST)[0])


def test_tuned_rows_sit_next_to_default_rows(tmp_path):
    tuned = _rows().assign(variant="tuned")
    path = tmp_path / "ridge.csv"
    pd.concat([_rows(), tuned]).to_csv(path, index=False)
    problems, coverage = check(path, MANIFEST)
    assert problems == [] and sum(line.endswith("6/6 folds") for line in coverage) == 2
    pd.concat([_rows(), tuned.assign(variant="ensembled")]).to_csv(path, index=False)
    assert any("variant must be one of" in p for p in check(path, MANIFEST)[0])
