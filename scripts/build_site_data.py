"""Build site/results.json from results/*.csv, the model cards and results/versions.yaml.

    python scripts/build_site_data.py

One record per pipeline, with a [score, fit, predict, total] quadruple per view (both,
regression, classification), or null when the pipeline does not cover every table and fold of
that view. Times are seconds per 1,000 rows, median over tables. Scores are means over tables,
R² floored at 0 per table. The page computes the ranking and the Pareto front from this file.
"""

import json
import math
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


def build(results_dir: Path = ROOT / "results", out: Path = ROOT / "site" / "results.json") -> dict:
    tables = json.loads(MANIFEST.read_text())["tables"]
    n_tables = {view: sum(1 for m in tables.values() if m["task"] in tasks) for view, tasks in VIEWS.items()}
    pipelines = []
    for path in sorted(results_dir.glob("*.csv")):
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


if __name__ == "__main__":
    build()
