"""Build site/results.json and site/results.parquet from the private results folder, the model cards and its versions.yaml.

    python scripts/build_site_data.py --results $STRABLE_RESULTS
    python scripts/build_site_data.py --check site/results.json

The results folder holds one CSV per model and ``versions.yaml`` (newest first: version, date,
note). Only the JSON written here is public. ``--check`` validates a JSON file instead, which the
Checks workflow runs on every pull request that changes it.

One record per pipeline (model, encoder, variant), with a list per view (both, reg, clf):
[score, time, fold rows, fold rows expected], or null when the pipeline has no row in that view.
As in the paper's Figure 4a, score is the median over the fold rows of the view of R² or AUC, and
time the median of the total run time per 1,000 samples (encoding, training and prediction
seconds over the table's rows). A view with fewer fold rows than expected is shown with an
incomplete tag. ``paper`` marks rows imported from the paper. The page computes the ranking and
the Pareto front from this file.

results.parquet holds the same numbers for all views in long format, one row per pipeline and view,
with its place on the Pareto front, for download. ``--check`` also compares it with the JSON when
it sits next to it.
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
from strable_leaderboard.metrics import N_FOLDS, VIEWS, leaderboard, pareto_front  # noqa: E402

VIEW_KEYS = {"both": "both", "regression": "reg", "classification": "clf"}
VARIANTS = ("default", "tuned")
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
PAPER = yaml.safe_load((ROOT / "scripts" / "paper_selection.yaml").read_text())  # display names of paper rows


def card_of(model: str) -> dict:
    path = ROOT / "models" / model / "model.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {
        "name": PAPER["learners"].get(model, model), "handles_strings": False, "access": "open", "hardware": ""}


def expected_folds() -> dict:
    tables = json.loads(MANIFEST.read_text())["tables"]
    return {VIEW_KEYS[v]: N_FOLDS * sum(1 for m in tables.values() if m["task"] in tasks) for v, tasks in VIEWS.items()}


def summarize(rows: pd.DataFrame, expected: dict) -> dict:
    """Per view: [median score, median total s per 1K samples, fold rows, fold rows expected] or None."""
    out = {}
    for view in VIEWS:
        key = VIEW_KEYS[view]
        b = leaderboard(rows.assign(pipeline="p"), view)
        out[key] = None if b.empty else [
            round(float(b["score"].iloc[0]), 4), _num(b["total_per_1k"].iloc[0]), int(b["n_folds"].iloc[0]), expected[key]]
    return out


def _num(x):
    return None if x is None or math.isnan(float(x)) else round(float(x), 4)


def build(results_dir: Path, out: Path = ROOT / "site" / "results.json") -> dict:
    expected = expected_folds()
    pipelines = []
    for path in sorted(results_dir.glob("*.csv")):
        card = card_of(path.stem)
        df = pd.read_csv(path)
        df["variant"] = df.get("variant", "default")  # files written before the variant column
        for (encoder, variant), rows in df.groupby(["encoder", "variant"], sort=True):
            e2e = encoder == "none" or card.get("handles_strings", False)
            pid = path.stem if e2e else f"{encoder}+{path.stem}"
            record = {
                "id": pid if variant == "default" else f"{pid}:{variant}",
                "enc": NAMES.get(encoder, PAPER["encoders"].get(encoder, encoder)), "lrn": card["name"],
                "kind": "e2e" if e2e else "modular", "variant": variant,
                "hw": ", ".join(sorted(set(rows["hardware"].astype(str)))),
            }
            if (rows["hardware"] == "paper").all():
                record["paper"] = True
            if card.get("anonymous"):
                record["anon"] = True
            elif card.get("access") == "closed":
                record["closed"] = True
            record.update(summarize(rows, expected))
            if any(record[k] for k in VIEW_KEYS.values()):
                pipelines.append(record)
    versions = yaml.safe_load((results_dir / "versions.yaml").read_text()) if (results_dir / "versions.yaml").exists() else []
    data = {"version": versions[0]["version"] if versions else "v0.0.0", "built": date.today().isoformat(),
            "history": [{"version": v["version"], "note": v["note"]} for v in versions], "pipelines": pipelines}
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    write_parquet(data, out.with_suffix(".parquet"))
    print(out, len(pipelines), "pipelines")
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
                         "paper": bool(p.get("paper")), "anonymous": bool(p.get("anon")), "closed": bool(p.get("closed")),
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
    expected = expected_folds()
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
        print("\n".join(found) if found else f"{args.check}: ok")
        sys.exit(1 if found else 0)
    if not args.results:
        sys.exit("give --results or set STRABLE_RESULTS")
    build(args.results)
