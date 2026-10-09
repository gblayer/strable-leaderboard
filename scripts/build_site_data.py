"""Build site/results.json from the private results folder, the model cards and its versions.yaml.

    python scripts/build_site_data.py --results $STRABLE_RESULTS [--models ridge tabpfn-2.5 ...]
    python scripts/build_site_data.py --check site/results.json

The results folder holds one CSV per model and ``versions.yaml`` (newest first: version, date,
note). ``--models`` limits the board to those results files (the dispatcher passes the models
already published, the paper's files and the one being published). Only the JSON written here is public. ``--check`` validates a JSON file instead, which the
Checks workflow runs on every pull request that changes it.

One record per pipeline, with a [score, fit, predict, total] quadruple per view (both,
regression, classification), or null when the pipeline does not cover every table and fold of
that view. Times are seconds per 1,000 rows, median over tables. Scores are means over tables
of the raw fold means, as in the paper. The page computes the ranking and the Pareto front from
this file.
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
from strable_leaderboard.metrics import VIEWS, table_scores  # noqa: E402

VIEW_KEYS = {"both": "both", "regression": "reg", "classification": "clf"}


def card_of(model: str) -> dict:
    path = ROOT / "models" / model / "model.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {"name": model, "handles_strings": False,
                                                                   "access": "open", "hardware": ""}


def summarize(rows: pd.DataFrame, n_tables: dict) -> dict:
    """Per view: [mean score, median fit/1k, median predict/1k, median total/1k] or None."""
    rows = rows.assign(total_s=rows["total_s"].fillna(rows["fit_s"] + rows["predict_s"]))
    t = table_scores(rows.assign(pipeline="p"))
    t["total_per_1k"] = rows.groupby(["table"]).apply(
        lambda g: (g["total_s"] / (g["n_train"] + g["n_test"]) * 1000).mean(), include_groups=False).reindex(t["table"]).to_numpy()
    out = {}
    for view, tasks in VIEWS.items():
        v = t[t["task"].isin(tasks)]
        complete = len(v) == n_tables[view] and (v["n_folds"] == 3).all()
        out[VIEW_KEYS[view]] = None if not complete else [
            round(float(v["score"].mean()), 4), _num(v["fit_per_1k"].median()), _num(v["predict_per_1k"].median()),
            _num(v["total_per_1k"].median())]
    return out


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def build(results_dir: Path, out: Path = ROOT / "site" / "results.json", models=None) -> dict:
    tables = json.loads(MANIFEST.read_text())["tables"]
    n_tables = {view: sum(1 for m in tables.values() if m["task"] in tasks) for view, tasks in VIEWS.items()}
    pipelines = []
    paths = [results_dir / f"{m}.csv" for m in sorted(models)] if models else sorted(results_dir.glob("*.csv"))
    for path in paths:
        card = card_of(path.stem)
        df = pd.read_csv(path)
        for encoder, rows in df.groupby("encoder", sort=True):
            e2e = encoder == "none" or card.get("handles_strings", False)
            record = {
                "id": path.stem if e2e else f"{encoder}+{path.stem}",
                "enc": NAMES.get(encoder, encoder), "lrn": card["name"], "kind": "e2e" if e2e else "modular",
                "hw": ", ".join(sorted(set(rows["hardware"].astype(str)))),
            }
            if card.get("anonymous"):
                record["anon"] = True
            elif card.get("access") == "closed":
                record["closed"] = True
            record.update(summarize(rows, n_tables))
            if any(record[k] for k in VIEW_KEYS.values()):
                pipelines.append(record)
    versions = yaml.safe_load((results_dir / "versions.yaml").read_text()) if (results_dir / "versions.yaml").exists() else []
    data = {"version": versions[0]["version"] if versions else "v0.0.0", "built": date.today().isoformat(),
            "history": [{"version": v["version"], "note": v["note"]} for v in versions], "pipelines": pipelines}
    out.write_text(json.dumps(data, indent=1))
    print(out, len(pipelines), "pipelines")
    return data


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
    for p in data["pipelines"]:
        if not p.get("lrn") or p.get("kind") not in ("modular", "e2e"):
            out.append(f"{p.get('id')}: needs lrn and kind")
        for view in VIEW_KEYS.values():
            v = p.get(view)
            if v is None:
                continue
            if len(v) != 4 or v[0] is None or v[0] > 1 or any(x is not None and x <= 0 for x in v[1:]):
                out.append(f"{p.get('id')} {view}: expected [score <= 1, fit, predict, total] with positive times")
            if view != "reg" and v[0] < 0:
                out.append(f"{p.get('id')} {view}: AUC below 0")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=os.environ.get("STRABLE_RESULTS"))
    parser.add_argument("--models", nargs="+", help="only these results files")
    parser.add_argument("--check", type=Path, help="validate this JSON instead of building")
    args = parser.parse_args()
    if args.check:
        found = problems(json.loads(args.check.read_text()))
        print("\n".join(found) if found else f"{args.check}: ok")
        sys.exit(1 if found else 0)
    if not args.results:
        sys.exit("give --results or set STRABLE_RESULTS")
    build(args.results, models=args.models)
