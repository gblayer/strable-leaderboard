"""Build the site data from the results folder, the model cards and its versions.yaml.

    python scripts/build_site_data.py --results $STRABLE_RESULTS
    python scripts/build_site_data.py --check site/results.json

The results folder holds one CSV per model and ``versions.yaml`` (newest first: version, date,
note). It writes site/results.json, site/results.parquet and the per-fold results,
site/results_per_fold.parquet and site/results_per_fold.csv. ``--check`` validates a JSON file
instead, which the Checks workflow runs on every pull request that changes it.

One record per pipeline (model, encoder, variant), with a list per view (both, reg, clf):
[score, time, fold rows, fold rows expected], or null when the pipeline has no row in that view.
As in the paper's Figure 4a, score is the median over the fold rows of the view of R² or AUC, and
time the median of the total run time per 1,000 samples (encoding, training and prediction
seconds over the table's rows). A view with fewer fold rows than expected is shown with an
incomplete tag. ``hw`` lists the hardware the rows ran on, when recorded. The page computes the
ranking and the Pareto front from this file.

results.parquet holds the same numbers for all views in long format, one row per pipeline and view,
with its place on the Pareto front, for download. results_per_fold.parquet and .csv hold one row per
pipeline, table and outer fold with a score (PER_FOLD columns). ``--check`` also compares the
Parquet with the JSON, and recomputes the JSON's medians and fold rows from the per-fold Parquet,
when they sit next to it.
"""

import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strable_leaderboard.data import MANIFEST  # noqa: E402
from strable_leaderboard.encoders import NAMES  # noqa: E402
from strable_leaderboard.metrics import VARIANTS, VIEWS, expected_folds, pareto_front, per_1k  # noqa: E402

VIEW_KEYS = {"both": "both", "regression": "reg", "classification": "clf"}
VIEW_NAMES = {"both": "Both (R² & AUC)", "reg": "Regression (R²)", "clf": "Classification (AUC)"}
SITE = "https://gblayer.github.io/strable-leaderboard/"
CITATION = ("Blayer et al., STRABLE: Benchmarking Tabular Machine Learning with Strings, arXiv:2605.12292 (2026), "
            "https://arxiv.org/abs/2605.12292")
COLUMNS = {
    "median_score": "median over the view's fold rows (tables x 3 outer folds) of R² (regression) and ROC-AUC (classification)",
    "median_time_per_1k_s": "median over the same fold rows of the total run time (encoding, training, prediction) in s per 1,000 samples",
    "fold_rows": "fold rows present; fewer than fold_rows_expected means incomplete",
    "pareto_front": "the score is strictly higher than that of every faster pipeline of the view",
}
PER_FOLD = ["id", "encoder", "learner", "variant", "table", "task", "fold", "score", "total_s", "n_rows", "time_per_1k_s"]
PAPER = yaml.safe_load((ROOT / "scripts" / "paper_selection.yaml").read_text())  # display names of imported pipelines


def card_of(model: str) -> dict:
    path = ROOT / "models" / model / "model.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {"name": PAPER["learners"].get(model, model)}


def folds_expected() -> dict:
    """Fold rows expected per view key (both, reg, clf)."""
    tables = json.loads(MANIFEST.read_text())["tables"]
    return {VIEW_KEYS[v]: n for v, n in expected_folds(tables).items()}


def summarize(folds: pd.DataFrame, expected: dict) -> dict:
    """Per view: [median score, median total s per 1K samples, fold rows, fold rows expected] or None.

    ``folds`` are one pipeline's rows of the per-fold table.
    """
    out = {}
    for view, tasks in VIEWS.items():
        r = folds[folds["task"].isin(tasks)]
        out[VIEW_KEYS[view]] = [round(float(r["score"].median()), 4), _num(r["time_per_1k_s"].median()), len(r),
                                expected[VIEW_KEYS[view]]] if len(r) else None
    return out


def per_fold(rows: pd.DataFrame, record: dict) -> pd.DataFrame:
    """One pipeline's fold rows in PER_FOLD columns. Only rows with a score count; check_results flags the others."""
    rows = per_1k(rows[rows["score"].notna()])
    out = rows.drop(columns="total_s", errors="ignore").rename(columns={"total": "total_s", "total_per_1k": "time_per_1k_s"})
    out = out.assign(id=record["id"], encoder=record["enc"], learner=record["lrn"])
    return out[PER_FOLD].sort_values(["table", "fold"]).reset_index(drop=True)


def _num(x):
    return None if x is None or math.isnan(float(x)) else round(float(x), 4)


def build(results_dir: Path, out: Path = ROOT / "site" / "results.json") -> dict:
    expected = folds_expected()
    pipelines, folds = [], []
    for path in sorted(results_dir.glob("*.csv")):
        card = card_of(path.stem)
        df = pd.read_csv(path)
        for (encoder, variant), rows in df.groupby(["encoder", "variant"], sort=True):
            e2e = encoder == "none"  # the runner writes "none" for models that handle strings
            pid = path.stem if e2e else f"{encoder}+{path.stem}"
            record = {
                "id": pid if variant == "default" else f"{pid}:{variant}",
                "enc": NAMES.get(encoder, PAPER["encoders"].get(encoder, encoder)), "lrn": card["name"],
                "kind": "e2e" if e2e else "modular", "variant": variant,
            }
            hw = sorted(set(rows["hardware"].astype(str)) - {"paper", "nan"})  # imported rows record no hardware
            if hw:
                record["hw"] = ", ".join(hw)
            folds.append(per_fold(rows, record))
            record.update(summarize(folds[-1], expected))
            pipelines.append(record)
    versions = yaml.safe_load((results_dir / "versions.yaml").read_text()) if (results_dir / "versions.yaml").exists() else []
    data = {"version": versions[0]["version"] if versions else "v0.0.0", "built": date.today().isoformat(),
            "history": [{"version": v["version"], "note": v["note"]} for v in versions], "pipelines": pipelines}
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    write_parquet(data, out.with_suffix(".parquet"))
    fold_table = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame(columns=PER_FOLD)
    fold_table.to_parquet(per_fold_path(out), index=False)
    fold_table.to_csv(per_fold_path(out).with_suffix(".csv"), index=False)
    print(out, len(pipelines), "pipelines,", len(fold_table), "fold rows")
    return data


def long_table(data: dict) -> pd.DataFrame:
    """All views in long format: one row per pipeline and view, with its place on the Pareto front."""
    rows = []
    for view in VIEW_KEYS.values():
        have = [p for p in data["pipelines"] if p.get(view)]
        board = pd.DataFrame([{"id": p["id"], "score": p[view][0], "total_per_1k": p[view][1]} for p in have
                              if p[view][1] is not None], columns=["id", "score", "total_per_1k"]).set_index("id")
        front = set(pareto_front(board))
        for p in have:
            score, time, n, expected = p[view]
            rows.append({"version": data["version"], "date": data["built"], "view": view, "view_name": VIEW_NAMES[view],
                         "id": p["id"], "encoder": p["enc"], "learner": p["lrn"], "kind": p["kind"], "variant": p["variant"],
                         "median_score": score, "median_time_per_1k_s": time, "fold_rows": n, "fold_rows_expected": expected,
                         "pareto_front": p["id"] in front})
    return pd.DataFrame(rows)


def write_parquet(data: dict, out: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(long_table(data), preserve_index=False)
    meta = {"board": "STRABLE leaderboard", "version": data["version"], "date": data["built"], "url": SITE,
            "views": VIEW_NAMES, "columns": COLUMNS, "citation": CITATION}
    pq.write_table(table.replace_schema_metadata({**table.schema.metadata, b"strable": json.dumps(meta, ensure_ascii=False).encode()}), out)


def parquet_problems(data: dict, path: Path) -> list[str]:
    """The Parquet next to a results.json must hold the same numbers."""
    found = pd.read_parquet(path).astype(str).reset_index(drop=True)
    return [] if found.equals(long_table(data).astype(str)) else [f"{path.name} does not match the JSON: rebuild both"]


def per_fold_path(json_path: Path) -> Path:
    return json_path.with_name("results_per_fold.parquet")


def per_fold_problems(data: dict, path: Path) -> list[str]:
    """The JSON's medians and fold rows, recomputed from the per-fold Parquet; the CSV must hold the same rows."""
    folds = pd.read_parquet(path)
    if list(folds.columns) != PER_FOLD:
        return [f"{path.name}: columns must be {PER_FOLD}"]
    csv = pd.read_csv(path.with_suffix(".csv"), float_precision="round_trip") if path.with_suffix(".csv").exists() else None
    same = csv is not None and csv.fillna("").astype(str).equals(folds.fillna("").astype(str))
    out = [] if same else [f"{path.with_suffix('.csv').name} does not match {path.name}: rebuild both"]
    expected = folds_expected()
    by_id = dict(iter(folds.groupby("id")))
    if set(by_id) != {p["id"] for p in data["pipelines"]}:
        out.append(f"{path.name}: not the same pipelines as the JSON")
    for p in data["pipelines"]:
        if p["id"] in by_id and summarize(by_id[p["id"]], expected) != {v: p.get(v) for v in VIEW_KEYS.values()}:
            out.append(f"{p['id']}: medians or fold rows differ from {path.name}")
    return out


def problems(data: dict) -> list[str]:
    """What is wrong with a results.json, as a list of messages."""
    out = []
    for key in ["version", "built", "history", "pipelines"]:
        if key not in data:
            out.append(f"missing {key}")
    if out:
        return out
    if not str(data["version"]).startswith("v") or not data["history"] or data["history"][0]["version"] != data["version"]:
        out.append("version must look like v0.1.2 and match the first history entry")
    ids = [p.get("id") for p in data["pipelines"]]
    if len(set(ids)) != len(ids):
        out.append("duplicate pipeline ids")
    expected = folds_expected()
    for p in data["pipelines"]:
        if not p.get("lrn") or p.get("kind") not in ("modular", "e2e"):
            out.append(f"{p.get('id')}: needs lrn and kind")
        if p.get("variant") not in VARIANTS:
            out.append(f"{p.get('id')}: variant must be one of {list(VARIANTS)}")
        if not any(p.get(view) for view in VIEW_KEYS.values()):
            out.append(f"{p.get('id')}: no view has results")
        for view in VIEW_KEYS.values():
            v = p.get(view)
            if v is None:
                continue
            if (len(v) != 4 or v[0] is None or v[0] > 1 or v[1] is None or v[1] <= 0
                    or not 0 < v[2] <= v[3] or v[3] != expected[view]):
                out.append(f"{p.get('id')} {view}: expected [score <= 1, time > 0, fold rows, {expected[view]}]")
            elif view == "clf" and v[0] < 0:
                out.append(f"{p.get('id')} {view}: AUC below 0")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=os.environ.get("STRABLE_RESULTS"))
    parser.add_argument("--check", type=Path, help="validate this JSON instead of building")
    args = parser.parse_args()
    if args.check:
        data = json.loads(args.check.read_text())
        found = problems(data)
        if not found and args.check.with_suffix(".parquet").exists():
            found = parquet_problems(data, args.check.with_suffix(".parquet"))
        if not found and per_fold_path(args.check).exists():
            found = per_fold_problems(data, per_fold_path(args.check))
        print("\n".join(found) if found else f"{args.check}: ok")
        sys.exit(1 if found else 0)
    if not args.results:
        sys.exit("give --results or set STRABLE_RESULTS")
    build(args.results)
