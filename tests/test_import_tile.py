import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_site_data import PAPER  # noqa: E402
from import_paper_results import convert  # noqa: E402
from import_tile_results import merge, select  # noqa: E402
from strable_leaderboard.encoders import ENCODERS, NAMES  # noqa: E402

TABLES = json.loads((ROOT / "splits" / "manifest.json").read_text())["tables"]
SELECTION = yaml.safe_load((ROOT / "scripts" / "tile_selection.yaml").read_text())


def test_tile_selection():
    rows = [p for methods in SELECTION["pipelines"].values() for p in methods.values()]
    ids = {f"{p['encoder']}+{p['model']}" for p in rows}
    assert ids == {"tile+tabpfn-2.5", "tile+tabicl-v2", "tile+tabpfn-3", "tile+causilo", "tfidf+causilo"}
    assert all(p["variant"] == "default" for p in rows)
    assert "tfidf" in ENCODERS  # Tf-Idf + Causilo: the id the runner gives models/causilo with its tfidf encoder
    windows = {p["where"]["feature_window"] for p in rows if p["encoder"] == "tile"}
    assert windows == {"100-300", "50-300", "50-250", "75-400"}  # one default setting per learner
    assert all(p["where"]["model"] == "llama3_14_layers|cell_meanpool|include_numbers" for p in rows if "where" in p)
    assert PAPER["encoders"]["tile"] == "TILE" and PAPER["learners"]["causilo"] == "Causilo"  # the board's names
    paper = yaml.safe_load((ROOT / "scripts" / "paper_selection.yaml").read_text())
    for key in ("learners", "encoders"):  # one display name per id
        assert all(paper[key].get(k, v) == v and NAMES.get(k, v) == v for k, v in SELECTION[key].items())


def test_import_tile_rows():
    table = next(iter(TABLES))
    settings = {"pca": 150, "feature_window": "100-300", "standard_scaling": True}
    source = pd.DataFrame([
        dict(data_name=table, method="num-str_latte_tabpfn", fold_index=0, r2=0.7, roc_auc=None, run_time="[12.5]", **settings),
        dict(data_name=table, method="num-str_latte_tabpfn", fold_index=0, r2=0.1, roc_auc=None, run_time="[1.0]",
             **{**settings, "feature_window": "50-250"}),  # another setting of the same method
        dict(data_name=table, method="num-str_latte_tabicl", fold_index=0, r2=0.2, roc_auc=None, run_time="[1.0]", **settings),
    ])
    methods = {"num-str_latte_tabpfn": dict(model="tabpfn-2.5", encoder="tile", variant="default",
                                            where={"pca": 150, "feature_window": "100-300", "standard_scaling": True})}
    sel = select(source, methods)
    assert len(sel) == 1 and sel["run_time"].tolist() == [12.5]
    new, problems = convert({"a.csv": sel}, TABLES, {"a.csv": methods})
    assert problems == [] and new["total_s"].tolist() == [12.5] and new["score"].tolist() == [0.7]
    old = pd.concat([new.assign(encoder="tfidf"), new.assign(score=0.1), new.assign(fold=1, score=0.2)])
    merged = merge(old, new)  # the tfidf row and fold 1 stay, the old fold 0 TILE row is replaced
    assert len(merged) == 3 and sorted(merged["score"]) == [0.2, 0.7, 0.7]
    assert list(merged.columns[:len(new.columns)]) == list(new.columns)
