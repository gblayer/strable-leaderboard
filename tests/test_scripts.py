"""Checks the maintainer scripts: the board files, the paper import and its Figure 4a selection."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_site_data import (PER_FOLD, build, flags_problems, leakage_problems, parquet_problems,  # noqa: E402
                             per_fold_problems, problems)
from import_paper_results import convert  # noqa: E402
from strable_leaderboard.encoders import NAMES  # noqa: E402
from strable_leaderboard.metrics import pareto_front  # noqa: E402

TABLES = json.loads((ROOT / "splits" / "manifest.json").read_text())["tables"]
A_REGRESSION_TABLE = next(t for t, m in TABLES.items() if m["task"] == "regression")


def _complete_results(model="ridge", encoder="tfidf"):
    rows = []
    for table, meta in TABLES.items():
        for fold in range(3):
            n_test = meta["test_rows_per_outer_fold"][fold]
            rows.append(dict(model=model, encoder=encoder, variant="default", table=table, task=meta["task"], fold=fold,
                             score=-0.5 if table == A_REGRESSION_TABLE else 0.6, fit_s=1.0, predict_s=0.5, total_s=None,
                             n_train=meta["n_rows"] - n_test, n_test=n_test, hardware="cpu", package="x==1",
                             checkpoint_revision="", protocol_version="0.1", date="2026-11-15"))
    return pd.DataFrame(rows)


def test_build_site_data(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    full = _complete_results()
    tuned = full[full["table"] != A_REGRESSION_TABLE].assign(variant="tuned", fit_s=None, predict_s=None, total_s=3.0,
                                                             hardware="paper")
    tuned.loc[tuned["task"] != "regression", "score"] = [float("nan")] + [0.6] * 95  # a fold without a score
    pd.concat([full, tuned]).to_csv(results / "ridge.csv", index=False)
    (results / "versions.yaml").write_text("- version: v0.1.0\n  date: 2026-11-15\n  note: First version\n")
    data = build(results, tmp_path / "results.json")
    assert data["version"] == "v0.1.0" and data["history"][0]["note"] == "First version"
    p, t = data["pipelines"]
    assert p["id"] == "tfidf+ridge" and p["enc"] == "Tf-Idf" and p["lrn"] == "Ridge" and p["kind"] == "modular"
    assert p["variant"] == "default" and p["hw"] == "cpu" and "paper" not in p
    assert p["both"][0] == 0.6 and p["clf"][0] == 0.6  # the median: one negative table does not move it
    assert all(len(p[v]) == 4 for v in ("both", "reg", "clf"))  # [score, time, fold rows, expected]: no fit or predict
    assert p["reg"][1] > 0  # total run time per 1K samples
    assert p["both"][2:] == [324, 324] and p["reg"][2:] == [228, 228] and p["clf"][2:] == [96, 96]
    assert t["id"] == "tfidf+ridge:tuned" and t["variant"] == "tuned" and "hw" not in t  # imported rows record no hardware
    assert t["reg"][1] > 0  # imported rows: their total run time
    assert t["reg"][2:] == [225, 228] and t["clf"][2:] == [95, 96]  # a table missing in reg, a score in clf: both incomplete
    assert problems(data) == []
    long = pd.read_parquet(tmp_path / "results.parquet")  # all views, one row per pipeline and view
    assert len(long) == 6 and set(long["view"]) == {"both", "reg", "clf"} and long["pareto_front"].dtype == bool
    assert parquet_problems(data, tmp_path / "results.parquet") == []
    folds = pd.read_parquet(tmp_path / "results_per_fold.parquet")  # one row per pipeline, table and fold with a score
    assert list(folds.columns) == PER_FOLD and len(folds) == 324 + 320
    row = folds[(folds["id"] == "tfidf+ridge") & (folds["table"] == A_REGRESSION_TABLE)].iloc[0]
    assert row["encoder"] == "Tf-Idf" and row["learner"] == "Ridge" and row["total_s"] == 1.5
    assert row["time_per_1k_s"] == pytest.approx(1500 / row["n_rows"])
    assert per_fold_problems(data, tmp_path / "results_per_fold.parquet") == []
    t["both"][2] = 325
    assert parquet_problems(data, tmp_path / "results.parquet")
    assert per_fold_problems(data, tmp_path / "results_per_fold.parquet") == ["tfidf+ridge:tuned: medians or fold rows differ from results_per_fold.parquet"]
    assert problems(data)
    t["both"][2] = 320
    p["reg"][0] = 0.5  # a median that the fold rows do not give
    assert per_fold_problems(data, tmp_path / "results_per_fold.parquet") == ["tfidf+ridge: medians or fold rows differ from results_per_fold.parquet"]
    folds.iloc[:-1].to_csv(tmp_path / "results_per_fold.csv", index=False)
    assert "results_per_fold.csv does not match results_per_fold.parquet: rebuild both" in per_fold_problems(data, tmp_path / "results_per_fold.parquet")


def test_build_site_data_leakage_flags(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _complete_results().to_csv(results / "ridge.csv", index=False)
    _complete_results(model="tabpfn-2.5").to_csv(results / "tabpfn-2.5.csv", index=False)
    (results / "versions.yaml").write_text("- version: v0.1.1\n  date: 2026-11-20\n  note: Added TabPFN 2.5\n")
    flags = {"tabpfn-2.5": {"leakage": "possible", "pr": 12}}
    (tmp_path / "flags.json").write_text(json.dumps(flags))  # read from next to the output, like site/flags.json
    data = build(results, tmp_path / "results.json")
    ridge, tabpfn = data["pipelines"]
    assert "leakage" not in ridge and "leakage_pr" not in ridge
    assert tabpfn["id"] == "tfidf+tabpfn-2.5" and tabpfn["leakage"] == "possible" and tabpfn["leakage_pr"] == 12
    assert problems(data) == [] and flags_problems(flags) == [] and leakage_problems(data, flags) == []
    long = pd.read_parquet(tmp_path / "results.parquet")
    assert set(long.loc[long["id"] == "tfidf+tabpfn-2.5", "leakage"]) == {"possible"}
    assert long.loc[long["id"] == "tfidf+ridge", "leakage"].isna().all()
    assert parquet_problems(data, tmp_path / "results.parquet") == []
    for folds in [pd.read_parquet(tmp_path / "results_per_fold.parquet"), pd.read_csv(tmp_path / "results_per_fold.csv")]:
        assert (folds.loc[folds["id"] == "tfidf+tabpfn-2.5", "leakage"] == "possible").all()
        assert folds.loc[folds["id"] == "tfidf+ridge", "leakage"].isna().all()
    assert per_fold_problems(data, tmp_path / "results_per_fold.parquet") == []
    assert leakage_problems(data, {}) == ["tfidf+tabpfn-2.5: leakage fields do not match flags.json"]
    assert leakage_problems(data, {**flags, "ridge": {"leakage": "possible", "pr": 3}}) == [
        "tfidf+ridge: leakage fields do not match flags.json"]
    assert flags_problems([]) and flags_problems({"Bad Name": {"leakage": "possible", "pr": 1}})
    for bad in [{"leakage": "none", "pr": 1}, {"leakage": "possible", "pr": None}, {"leakage": "possible"},
                {"leakage": "possible", "pr": 1, "note": "x"}, "possible"]:
        assert flags_problems({"m": bad}), bad
    tabpfn["leakage_pr"] = "12"
    assert problems(data) == ['tfidf+tabpfn-2.5: leakage must be "possible" with leakage_pr, the model pull request']


def test_import_paper_rows():
    table = next(iter(TABLES))
    paper = pd.DataFrame([
        dict(data_name=table, method="num-str_tabvec_tabpfn_default", fold_index=0, r2=0.7, roc_auc=None, run_time=12.0),
        dict(data_name=table, method="num-str_catboost_catboost_tune", fold_index=1, r2=0.5, roc_auc=None, run_time=30.0),
        dict(data_name=table, method="num-str_llm-e5-base-v2_ridge_default", fold_index=0, r2=0.7, roc_auc=None, run_time=1.0),
        dict(data_name="not-a-table", method="num-str_tabvec_ridge_default", fold_index=0, r2=0.1, roc_auc=None, run_time=1.0),
    ])
    pipelines = {"a.csv": {"num-str_tabvec_tabpfn_default": dict(model="tabpfn-2.5", encoder="tfidf", variant="default"),
                           "num-str_catboost_catboost_tune": dict(model="catboost", encoder="none", variant="tuned"),
                           "num-str_tabvec_ridge_default": dict(model="ridge", encoder="tfidf", variant="default"),
                           "num-str_tabvec_xgb_default": dict(model="xgboost", encoder="tfidf", variant="default")}}
    df, found = convert({"a.csv": paper}, TABLES, pipelines)
    assert list(df["model"]) == ["tabpfn-2.5", "catboost"] and list(df["encoder"]) == ["tfidf", "none"]
    assert list(df["variant"]) == ["default", "tuned"]  # unlisted methods (E5-base) are never imported
    assert df["fit_s"].isna().all() and list(df["total_s"]) == [12.0, 30.0] and (df["hardware"] == "paper").all()
    assert df.loc[0, "n_test"] == TABLES[table]["test_rows_per_outer_fold"][0]
    assert found == ["a.csv: no rows for num-str_tabvec_xgb_default", "a.csv: unknown table not-a-table"]


def test_paper_selection_is_figure_4a():
    sel = yaml.safe_load((ROOT / "scripts" / "paper_selection.yaml").read_text())
    rows = [(csv, m, p) for csv, methods in sel["pipelines"].items() for m, p in methods.items()]
    assert len(rows) == 75
    keys = {(p["model"], p["encoder"], p["variant"]) for _, _, p in rows}
    assert len(keys) == 75  # one pipeline per entry
    for csv, m, p in rows:
        assert p["model"] in sel["learners"] and p["encoder"] in sel["encoders"]
        assert p["variant"] == ("tuned" if m.endswith("_tune") else "default")
        assert (p["encoder"] == "tfidf") == ("_tabvec_" in m)  # only the paper's Tf-Idf is the leaderboard's tfidf
        # a paper row shares a leaderboard encoder id only when it is the same pipeline
        assert (p["encoder"] == "qwen-3-8b-first30") == ("qwen3-8b" in m and "nopca" in m.replace("no_pca", "nopca"))
        assert (p["encoder"] == "minilm-l6-v2-pca30") == ("MiniLM-L6-v2" in m)
        assert p["encoder"].endswith("ss-pca30") == ("standscal" in csv + m)
        assert p["encoder"] != "llama-3.1-8b-first30"  # the paper's LLaMA rows are PCA-30
        if "realmlp" in m:
            assert "jasper" not in m and "e5-base" not in m
    assert ("catboost", "none", "tuned") in keys and ("mambular", "none", "default") in keys
    assert ("tabpfn-2.5", "nemotron-1b-pca30", "default") in keys
    assert all(NAMES[e] == sel["encoders"][e] for e in NAMES if e in sel["encoders"])  # one display name per id
    assert not any("tabpfn3" in m or "30_thresh" in csv for csv, m, _ in rows)


def test_published_board_reproduces_table_e10():
    """The six pipelines of the paper's Table E.10, checked by id: their numbers, and the Pareto front of the
    "both" view among the 75 Figure 4a pipelines.

    Valid until a re-run replaces the fold rows of one of these 75 ids; then update or drop this test.
    """
    data = json.loads((ROOT / "site" / "results.json").read_text())
    assert problems(data) == []
    assert parquet_problems(data, ROOT / "site" / "results.parquet") == []  # the download holds the same numbers and front
    assert per_fold_problems(data, ROOT / "site" / "results_per_fold.parquet") == []  # the medians come from the fold rows
    board = pd.DataFrame([{"id": p["id"], "score": p["both"][0], "total_per_1k": p["both"][1]}
                          for p in data["pipelines"] if p.get("both")]).set_index("id")
    e10 = {"targetencoder+ridge": (0.6593, 0.0924), "targetencoder+extratrees": (0.7290, 0.1695),
          "tfidf+extratrees": (0.7407, 0.9679), "e5-small-v2-pca30+extratrees": (0.7422, 2.5152),
          "tfidf+tabicl-v2": (0.7799, 2.9988), "tfidf+tabpfn-2.5": (0.7891, 6.6700)}
    assert {i: tuple(board.loc[i]) for i in e10} == e10
    sel = yaml.safe_load((ROOT / "scripts" / "paper_selection.yaml").read_text())  # the 75 pipelines of Figure 4a
    ids = [(p["model"] if p["encoder"] == "none" else p["encoder"] + "+" + p["model"]) + ("" if p["variant"] == "default" else ":" + p["variant"])
           for methods in sel["pipelines"].values() for p in methods.values()]
    assert pareto_front(board.loc[ids]) == list(e10)
