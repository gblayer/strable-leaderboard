import json
import sys
from pathlib import Path

import pandas as pd
import pytest
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
                   "tile+exaone-tabular", "tile+kumo-tabular", "tile+kumo-tabular-medium", "tile+kumo-tabular-small",
                   "tile+chakra-tab-medium", "tile-70b+chakra-tab-medium",
                   "tfidf+kumo-tabular-small", "tfidf+kumo-tabular-medium",
                   *(f"{e}+{m}" for e in ("tile-70b", "case", "lighttar", "tile-gemma-12b", "tile-1b", "tile-gemma-1b")
                     for m in ("tabpfn-2.5", "tabicl-v2", "tabpfn-3", "tabpfn-3.5", "causilo", "kumo-tabular",
                               "kumo-tabular-medium", "kumo-tabular-small"))}
    assert all(p["variant"] == "default" for p in rows)
    held = {f"{p['encoder']}+{p['model']}" for p in rows if p.get("hold")}
    assert held == {"tile+exaone-tabular"} | {i for i in ids if i.startswith("case+")}  # listed, not imported
    assert {f"{p['encoder']}+{p['model']}" for ms in imported(SELECTION["pipelines"]).values() for p in ms.values()} == ids - held
    assert "tfidf" in ENCODERS  # Tf-Idf + Causilo: the id the runner gives models/causilo with its tfidf encoder
    windows = {p["model"]: p["where"]["feature_window"] for p in rows if p["encoder"] == "tile"}  # one per learner
    assert windows["tabpfn-2.5"] == "200-400" and windows["exaone-tabular"] == "150-400"
    assert {windows[k] for k in ("kumo-tabular", "kumo-tabular-medium", "kumo-tabular-small", "chakra-tab-medium")} == {"200-600"}
    api = [p for p in rows if p.get("hardware") == "api"]  # an API learner: its version, no GPU
    assert {p["model"] for p in api} == {"chakra-tab-medium"} and all(p.get("api_version") for p in api)
    assert [p.get("date") for p in rows if p["encoder"] == "tfidf"] == ["2026-09", None, None]  # Causilo: its own date
    tile = [p["where"] for p in rows if p["encoder"] == "tile"]
    assert all(w["model"] == "llama3_14_layers_v2|cell_meanpool|include_numbers|clean" and w["window_total"] for w in tile)
    tile70b = {p["model"]: p["where"] for p in rows if p["encoder"] == "tile-70b"}  # TILE-Llama3.1-70B: the same windows
    assert all(w["model"] == "llama-3.1-70b_32_layers_v2|cell_meanpool|include_numbers|clean" for w in tile70b.values())
    assert {m: w["feature_window"] for m, w in tile70b.items()} == {m: w for m, w in windows.items() if m in tile70b}
    gemma = {p["model"]: p["where"] for p in rows if p["encoder"] == "tile-gemma-12b"}  # TILE-Gemma3-12B: the same windows
    assert all(w["model"] == "gemma-3-12b_21_layers_v2|cell_meanpool|include_numbers|clean" for w in gemma.values())
    assert {m: w["feature_window"] for m, w in gemma.items()} == {m: w for m, w in windows.items() if m in gemma}
    for enc, model in [("tile-1b", "llama-3.2-1b_10_layers_v2"), ("tile-gemma-1b", "gemma-3-1b_9_layers_v2")]:  # the same windows
        small = {p["model"]: p["where"] for p in rows if p["encoder"] == enc}
        assert len(small) == 8 and all(w["model"] == model + "|cell_meanpool|include_numbers|clean" for w in small.values())
        assert {m: w["feature_window"] for m, w in small.items()} == {m: w for m, w in windows.items() if m in small}
    assert tuple(PAPER["encoders"][e] for e in ("tile", "tile-70b", "tile-gemma-12b", "tile-1b", "tile-gemma-1b")) == (
        "TILE-Llama3.1-8B", "TILE-Llama3.1-70B", "TILE-Gemma3-12B", "TILE-Llama3.2-1B", "TILE-Gemma3-1B")  # the board's names
    assert PAPER["learners"]["causilo"] == "Causilo"
    case = [p for p in rows if p["encoder"] == "case"]  # CASE alone, at its pinned commit, every column raw
    assert PAPER["encoders"]["case"] == "CASE" and len(case) == 8
    assert all(p["where"] == {"model": "case-f0bbd5a", "raw_text_card": "all"} and p["date"] == "2026-11" for p in case)
    lighttar = [p for p in rows if p["encoder"] == "lighttar"]  # LightTAR alone, at its package defaults
    assert PAPER["encoders"]["lighttar"] == "LightTAR" and len(lighttar) == 8 and not any(p.get("hold") for p in lighttar)
    assert all(p["where"] == {"model": "lighttar-0.1.0"} and p["date"] == "2026-11" for p in lighttar)
    assert {m for m in SELECTION["learners"]} <= RESERVED  # learners without a folder: names only the maintainer adds
    paper = yaml.safe_load((ROOT / "scripts" / "paper_selection.yaml").read_text())
    for key in ("learners", "encoders"):  # one display name per id
        assert all(paper[key].get(k, v) == v and NAMES.get(k, v) == v for k, v in SELECTION[key].items())


def test_the_hardware_of_imported_rows():
    table = next(iter(TABLES))
    pipelines = {"a.csv": {"m": dict(model="tabpfn-3.5", encoder="tile", variant="default"),
                           "n": dict(model="chakra-tab-medium", encoder="tile", variant="default", hardware="api")},
                 "b.csv": {"m": dict(model="kumo-tabular", encoder="tile", variant="default", hardware="NVIDIA A40")},
                 "c.csv": {"m": dict(model="causilo", encoder="tfidf", variant="default")}}
    frames = {"a.csv": pd.DataFrame([dict(method="m", data_name=table, fold_index=f, gpu=g) for f, g in
                                     enumerate(["NVIDIA A40", "NVIDIA A100-PCIE-40GB", "NVIDIA A40"])]
                                    + [dict(method="n", data_name=table, fold_index=0, gpu=None)]),
              "b.csv": pd.DataFrame([dict(method="m", data_name=table, fold_index=0)]),
              "c.csv": pd.DataFrame([dict(method="m", data_name=table, fold_index=0)])}
    df = pd.DataFrame([dict(model="tabpfn-3.5", encoder="tile", variant="default", table=table, fold=1),
                       dict(model="kumo-tabular", encoder="tile", variant="default", table=table, fold=0),
                       dict(model="causilo", encoder="tfidf", variant="default", table=table, fold=0),
                       dict(model="chakra-tab-medium", encoder="tile", variant="default", table=table, fold=0)])
    # a row without a GPU (a CPU row) takes its pipeline's own hardware
    assert hardware_of(df, frames, pipelines) == ["NVIDIA A100-PCIE-40GB", "NVIDIA A40", "paper", "api"]


def test_the_hardware_of_rows_whose_encoder_ran_on_another_gpu():
    table = next(iter(TABLES))
    pipelines = {"a.csv": {"m": dict(model="tabpfn-3.5", encoder="case", variant="default", encoder_gpu="case_gpu"),
                           "n": dict(model="causilo", encoder="tile", variant="default")}}
    frames = {"a.csv": pd.DataFrame([dict(method="m", data_name=table, fold_index=0, gpu="NVIDIA A40", case_gpu="NVIDIA H100 NVL"),
                                     dict(method="m", data_name=table, fold_index=1, gpu="NVIDIA A40", case_gpu="NVIDIA A40"),
                                     dict(method="n", data_name=table, fold_index=0, gpu="NVIDIA A40", case_gpu="NVIDIA H100 NVL")])}
    df = pd.DataFrame([dict(model="tabpfn-3.5", encoder="case", variant="default", table=table, fold=k) for k in (0, 1)]
                      + [dict(model="causilo", encoder="tile", variant="default", table=table, fold=0)])
    # the encoder's GPU first, once when both are the same; a pipeline without encoder_gpu keeps the learner's only
    assert hardware_of(df, frames, pipelines) == ["NVIDIA H100 NVL, NVIDIA A40", "NVIDIA A40", "NVIDIA A40"]
    # a row without a learner GPU (an API learner) keeps its pipeline's own hardware, not the encoder's GPU
    api = {"a.csv": {"m": dict(model="chakra-tab-medium", encoder="case", variant="default", hardware="api", encoder_gpu="case_gpu")}}
    rows = pd.DataFrame([dict(method="m", data_name=table, fold_index=0, gpu=None, case_gpu="NVIDIA H100 NVL")])
    one = pd.DataFrame([dict(model="chakra-tab-medium", encoder="case", variant="default", table=table, fold=0)])
    assert hardware_of(one, {"a.csv": rows}, api) == ["api"]
    with pytest.raises(ValueError, match="encoder_gpu"):  # a named column the CSV lacks is an error
        hardware_of(one, {"a.csv": rows.drop(columns="case_gpu")}, api)


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
