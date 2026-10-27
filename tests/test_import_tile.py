import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_site_data import PAPER  # noqa: E402
from import_paper_results import convert  # noqa: E402
from import_tile_results import hardware_of, imported, merge, select  # noqa: E402
from strable_leaderboard import RESERVED  # noqa: E402
from strable_leaderboard.encoders import ENCODERS, NAMES  # noqa: E402

TABLES = json.loads((ROOT / "splits" / "manifest.json").read_text())["tables"]
SELECTION = yaml.safe_load((ROOT / "scripts" / "tile_selection.yaml").read_text())


def test_tile_selection():
    rows = [p for methods in SELECTION["pipelines"].values() for p in methods.values()]
    ids = {f"{p['encoder']}+{p['model']}" for p in rows}
    assert ids == {"tile+tabpfn-2.5", "tile+tabicl-v2", "tile+tabpfn-3", "tile+tabpfn-3.5", "tile+causilo", "tfidf+causilo",
                   "tile+exaone-tabular", "tile+kumo-tabular", "tile+kumo-tabular-medium", "tile+kumo-tabular-small"}
    assert all(p["variant"] == "default" for p in rows)
    held = {f"{p['encoder']}+{p['model']}" for p in rows if p.get("hold")}
    assert held == {"tile+exaone-tabular"}  # listed, not imported
    assert {f"{p['encoder']}+{p['model']}" for ms in imported(SELECTION["pipelines"]).values() for p in ms.values()} == ids - held
    assert "tfidf" in ENCODERS  # Tf-Idf + Causilo: the id the runner gives models/causilo with its tfidf encoder
    windows = {p["model"]: p["where"]["feature_window"] for p in rows if p["encoder"] == "tile"}  # one per learner
    assert windows["tabpfn-2.5"] == "200-400" and windows["exaone-tabular"] == "150-400"
    assert {windows[k] for k in ("kumo-tabular", "kumo-tabular-medium", "kumo-tabular-small")} == {"200-600"}
    assert [p.get("date") for p in rows if p["encoder"] == "tfidf"] == ["2026-09"]  # its own date
    tile = [p["where"] for p in rows if p["encoder"] == "tile"]
    assert all(w["model"] == "llama3_14_layers_v2|cell_meanpool|include_numbers|clean" and w["window_total"] for w in tile)
    assert PAPER["encoders"]["tile"] == "TILE" and PAPER["learners"]["causilo"] == "Causilo"  # the board's names
    assert {m for m in SELECTION["learners"]} <= RESERVED  # learners without a folder: names only the maintainer adds
    paper = yaml.safe_load((ROOT / "scripts" / "paper_selection.yaml").read_text())
    for key in ("learners", "encoders"):  # one display name per id
        assert all(paper[key].get(k, v) == v and NAMES.get(k, v) == v for k, v in SELECTION[key].items())


def test_the_hardware_of_imported_rows():
    table = next(iter(TABLES))
    pipelines = {"a.csv": {"m": dict(model="tabpfn-3.5", encoder="tile", variant="default")},
                 "b.csv": {"m": dict(model="kumo-tabular", encoder="tile", variant="default", hardware="NVIDIA A40")},
                 "c.csv": {"m": dict(model="causilo", encoder="tfidf", variant="default")}}
    frames = {"a.csv": pd.DataFrame([dict(method="m", data_name=table, fold_index=f, gpu=g) for f, g in
                                     enumerate(["NVIDIA A40", "NVIDIA A100-PCIE-40GB", "NVIDIA A40"])]),
              "b.csv": pd.DataFrame([dict(method="m", data_name=table, fold_index=0)]),
              "c.csv": pd.DataFrame([dict(method="m", data_name=table, fold_index=0)])}
    df = pd.DataFrame([dict(model="tabpfn-3.5", encoder="tile", variant="default", table=table, fold=1),
                       dict(model="kumo-tabular", encoder="tile", variant="default", table=table, fold=0),
                       dict(model="causilo", encoder="tfidf", variant="default", table=table, fold=0)])
    assert hardware_of(df, frames, pipelines) == ["NVIDIA A100-PCIE-40GB", "NVIDIA A40", "paper"]


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
