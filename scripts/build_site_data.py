"""Build site/results.json from the private results folder, the model cards and its versions.yaml.

    python scripts/build_site_data.py --results $STRABLE_RESULTS
    python scripts/build_site_data.py --check site/results.json

The results folder holds one CSV per model and ``versions.yaml`` (newest first: version, date,
note). Only the JSON written here is public. ``--check`` validates a JSON file instead, which the
Checks workflow runs on every pull request that changes it.

One record per pipeline (model, encoder, variant), with a list per view (both, reg, clf):
[score, fit, predict, total, fold rows, fold rows expected], or null when the pipeline has no
row in that view. Score and times are medians over the fold rows of the view, as in the paper's
Figure 4a; times are seconds per 1,000 rows (fit: training rows, predict: test rows, total: table
rows), null when not measured. A view with fewer fold rows than expected is shown with an
incomplete tag. ``paper`` marks rows imported from the paper. The page computes the ranking and
the Pareto front from this file.
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
from strable_leaderboard.metrics import N_FOLDS, VIEWS, leaderboard  # noqa: E402

VIEW_KEYS = {"both": "both", "regression": "reg", "classification": "clf"}
VARIANTS = ("default", "tuned")
PAPER = yaml.safe_load((ROOT / "scripts" / "paper_selection.yaml").read_text())  # display names of paper rows


def card_of(model: str) -> dict:
    path = ROOT / "models" / model / "model.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {
        "name": PAPER["learners"].get(model, model), "handles_strings": False, "access": "open", "hardware": ""}


def expected_folds() -> dict:
    tables = json.loads(MANIFEST.read_text())["tables"]
    return {VIEW_KEYS[v]: N_FOLDS * sum(1 for m in tables.values() if m["task"] in tasks) for v, tasks in VIEWS.items()}


def summarize(rows: pd.DataFrame, expected: dict) -> dict:
    """Per view: [median score, fit/1k, predict/1k, total/1k, fold rows, fold rows expected] or None."""
    out = {}
    for view in VIEWS:
        key = VIEW_KEYS[view]
        b = leaderboard(rows.assign(pipeline="p"), view)
        out[key] = None if b.empty else [
            round(float(b["score"].iloc[0]), 4), *(_num(b[c].iloc[0]) for c in ["fit_per_1k", "predict_per_1k", "total_per_1k"]),
            int(b["n_folds"].iloc[0]), expected[key]]
    return out


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


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
            if (len(v) != 6 or v[0] is None or v[0] > 1 or v[3] is None or any(x is not None and x <= 0 for x in v[1:4])
                    or not 0 < v[4] <= v[5] or v[5] != expected[view]):
                out.append(f"{p.get('id')} {view}: expected [score <= 1, fit, predict, total > 0, fold rows, "
                           f"{expected[view]}] with positive times")
            elif view == "clf" and v[0] < 0:
                out.append(f"{p.get('id')} {view}: AUC below 0")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=os.environ.get("STRABLE_RESULTS"))
    parser.add_argument("--check", type=Path, help="validate this JSON instead of building")
    args = parser.parse_args()
    if args.check:
        found = problems(json.loads(args.check.read_text()))
        print("\n".join(found) if found else f"{args.check}: ok")
        sys.exit(1 if found else 0)
    if not args.results:
        sys.exit("give --results or set STRABLE_RESULTS")
    build(args.results)
