import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_site_data import build  # noqa: E402
from import_paper_results import convert  # noqa: E402

TABLES = json.loads((ROOT / "splits" / "manifest.json").read_text())["tables"]
A_REGRESSION_TABLE = next(t for t, m in TABLES.items() if m["task"] == "regression")


def _complete_results(model="ridge", encoder="tfidf"):
    rows = []
    for table, meta in TABLES.items():
        for fold in range(3):
            n_test = meta["test_rows_per_outer_fold"][fold]
            rows.append(dict(model=model, encoder=encoder, table=table, task=meta["task"], fold=fold,
                             score=-0.5 if table == A_REGRESSION_TABLE else 0.6, fit_s=1.0, predict_s=0.5, total_s=None,
                             n_train=meta["n_rows"] - n_test, n_test=n_test, hardware="cpu", package="x==1",
                             checkpoint_revision="", protocol_version="0.1", date="2026-11-15"))
    return pd.DataFrame(rows)


def test_build_site_data(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _complete_results().to_csv(results / "ridge.csv", index=False)
    (results / "versions.yaml").write_text("- version: v0.1.0\n  date: 2026-11-15\n  note: First version\n")
    data = build(results, tmp_path / "results.json")
    assert data["version"] == "v0.1.0" and data["history"][0]["note"] == "First version"
    (p,) = data["pipelines"]
    assert p["id"] == "tfidf+ridge" and p["enc"] == "Tf-Idf" and p["lrn"] == "Ridge" and p["kind"] == "modular"
    assert p["both"][0] < 0.6 and p["reg"][0] < 0.6 and p["clf"][0] == 0.6  # one floored table pulls the mean down
    assert p["reg"][1] > 0 and p["reg"][2] > 0 and p["reg"][3] > 0  # fit, predict, total per 1K rows


def test_import_paper_rows(tmp_path):
    table = next(iter(TABLES))
    paper = pd.DataFrame([
        dict(data_name=table, method="num-str_tabvec_tabpfn_default", fold_index=0, r2=0.7, roc_auc=None, run_time=12.0),
        dict(data_name=table, method="num-str_contexttab_contexttab", fold_index=1, r2=0.5, roc_auc=None, run_time=30.0),
        dict(data_name=table, method="num-str_tabvec_xgb_tune", fold_index=0, r2=0.7, roc_auc=None, run_time=1.0),
        dict(data_name="not-a-table", method="num-str_tabvec_ridge_default", fold_index=0, r2=0.1, roc_auc=None, run_time=1.0),
    ])
    df, skipped = convert(paper, TABLES)
    assert list(df["model"]) == ["tabpfn-2.5", "contexttab"] and list(df["encoder"]) == ["tfidf", "none"]
    assert df["fit_s"].isna().all() and list(df["total_s"]) == [12.0, 30.0] and (df["hardware"] == "paper").all()
    assert df.loc[0, "n_test"] == TABLES[table]["test_rows_per_outer_fold"][0]
    assert skipped == ["num-str_tabvec_xgb_tune", "table not-a-table"]
