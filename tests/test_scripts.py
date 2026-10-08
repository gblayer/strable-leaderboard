import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_site_data import build, problems  # noqa: E402
from import_paper_results import convert  # noqa: E402
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
    pd.concat([full, tuned]).to_csv(results / "ridge.csv", index=False)
    (results / "versions.yaml").write_text("- version: v0.1.0\n  date: 2026-11-15\n  note: First version\n")
    data = build(results, tmp_path / "results.json")
    assert data["version"] == "v0.1.0" and data["history"][0]["note"] == "First version"
    p, t = data["pipelines"]
    assert p["id"] == "tfidf+ridge" and p["enc"] == "Tf-Idf" and p["lrn"] == "Ridge" and p["kind"] == "modular"
    assert p["variant"] == "default" and "paper" not in p
    assert p["both"][0] == 0.6 and p["clf"][0] == 0.6  # the median: one negative table does not move it
    assert p["reg"][1] > 0 and p["reg"][2] > 0 and p["reg"][3] > 0  # fit, predict, total per 1K rows
    assert p["both"][4:] == [324, 324] and p["reg"][4:] == [228, 228] and p["clf"][4:] == [96, 96]
    assert t["id"] == "tfidf+ridge:tuned" and t["variant"] == "tuned" and t["paper"] is True
    assert t["reg"][1] is None and t["reg"][2] is None and t["reg"][3] > 0  # paper rows: total only
    assert t["reg"][4:] == [225, 228] and t["clf"][4:] == [96, 96]  # one table missing: incomplete in reg only
    assert problems(data) == []
    t["both"][4] = 325
    assert problems(data)


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
        assert p["encoder"] not in ("minilm-l6-v2", "llama-3.1-8b")  # the paper's PCA rows are not first-30
        if "realmlp" in m:
            assert "jasper" not in m and "e5-base" not in m
    assert ("catboost", "none", "tuned") in keys and ("mambular", "none", "default") in keys
    assert ("tabpfn-2.5", "nemotron-1b-pca30", "default") in keys
    assert not any("tabpfn3" in m or "30_thresh" in csv for csv, m, _ in rows)


def test_published_board_reproduces_table_e8():
    """site/results.json as built from the paper's Figure 4a rows: Table E.8, the Pareto front of the "both" view."""
    data = json.loads((ROOT / "site" / "results.json").read_text())
    assert problems(data) == []
    paper = [p for p in data["pipelines"] if p.get("paper")]
    assert len(paper) == 75
    board = pd.DataFrame([{"id": p["id"], "score": p["both"][0], "total_per_1k": p["both"][3]} for p in paper]).set_index("id")
    e8 = {"targetencoder+ridge": (0.6593, 0.0924), "targetencoder+extratrees": (0.7290, 0.1695),
          "tfidf+extratrees": (0.7407, 0.9679), "e5-small-v2-pca30+extratrees": (0.7422, 2.5152),
          "tfidf+tabicl-v2": (0.7799, 2.9988), "tfidf+tabpfn-2.5": (0.7891, 6.6700)}
    assert pareto_front(board, "total_per_1k") == list(e8)
    assert {i: tuple(board.loc[i]) for i in e8} == e8
    incomplete = {p["id"] for p in paper if p["both"][4] < p["both"][5]}
    assert "mambular" in incomplete
