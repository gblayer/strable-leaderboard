"""Build the site data from the results folder, the model cards and its versions.yaml.

    python scripts/build_site_data.py --results $STRABLE_RESULTS [--models ridge tabpfn-2.5 ...]
    python scripts/build_site_data.py --check site/results.json

The results folder holds one CSV per model and ``versions.yaml`` (newest first: version, date,
note). It writes site/results.json, site/results.parquet and the per-fold results,
site/results_per_fold.parquet and site/results_per_fold.csv. ``--models`` limits the board to those
results files (the dispatcher passes the models already published, the paper's files and the one
being published). Only the files written here are public. ``--check`` validates a JSON file
instead, which the Checks workflow runs on every pull request that changes it.

One record per pipeline (model, encoder, variant), with a list per view (both, reg, clf):
[score, time, fold rows, fold rows expected, mean score], or null when the pipeline has no row in that
view. A model whose card lists the tasks of one family only (``tasks: [regression]``, say) gets that
view alone: no "both" view, no view for the other family.
As in the paper's Figure 4a, score is the median over the fold rows of the view of R² or AUC, and
time the median of the total run time per 1,000 samples (encoding, training and prediction
seconds over the table's rows); mean score, their mean, is the page's other Score. A view with fewer
fold rows than expected is shown with an incomplete tag. ``hw`` lists the hardware the rows ran on,
when recorded. The page computes the ranking and the Pareto front from this file.

API models: a pipeline whose rows all ran on hardware "api" (on its provider's servers) gets
``"api": true`` and ``"api_version"``, the versions its provider reported, which the page marks API,
and the Parquet and per-fold files an ``api`` column (true or false).

Overlap flags: when site/flags.json sits next to the output (``{"<model>": {"overlap": true,
"evidence": "<url>"}}``, set by the maintainer), every pipeline of a flagged model gets
``"overlap": true`` and ``"overlap_evidence": "<url>"``, which the page shows as a ∩ mark, and the
Parquet and per-fold files an ``overlap`` column (true or false).
``--check`` validates flags.json's shape and that the JSON carries exactly its flags.

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
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strable_leaderboard import SLUG  # noqa: E402
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
    "mean_score": "mean over the same fold rows of R² and ROC-AUC; a fold where a pipeline fails (R² far below 0) pulls it down",
    "median_time_per_1k_s": "median over the same fold rows of the total run time (encoding, training, prediction) in s per 1,000 samples",
    "fold_rows": "fold rows present; fewer than fold_rows_expected means incomplete",
    "pareto_front": "the median score is strictly higher than that of every faster pipeline of the view",
    "overlap": "true when the maintainer marked the model's pre-training data as possibly overlapping STRABLE's tables "
               "(the page's ∩ mark; results.json links the evidence)",
    "api": "true when the model runs on its provider's servers, behind an API: its time includes the network and "
           "the provider's hardware (the page's API mark; results.json gives the version the provider reported)",
}
PER_FOLD = ["id", "encoder", "learner", "variant", "table", "task", "fold", "score", "total_s", "n_rows", "time_per_1k_s",
            "overlap", "api"]
FAMILIES = {"reg": {"regression"}, "clf": {"binary", "multiclass"}}  # the views of a one-family model
URL = re.compile(r"^https?://[^\s\"'<>]+$")
# Display names of imported pipelines: the paper's and the TILE paper's selections.
IMPORTS = [yaml.safe_load((ROOT / "scripts" / f).read_text()) for f in ("paper_selection.yaml", "tile_selection.yaml")]
PAPER = {key: {k: v for s in IMPORTS for k, v in s[key].items()} for key in ("learners", "encoders")}


def card_of(model: str) -> dict:
    """The card in models/, else a paper learner's name."""
    path = ROOT / "models" / model / "model.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {"name": PAPER["learners"].get(model, model)}


def card_views(card: dict) -> set[str]:
    """The views of a model's pipelines: only reg or only clf when the card's ``tasks`` are one family, else all
    three (also for the paper's learners, which have no card)."""
    tasks = set(card.get("tasks") or [])
    return next(({view} for view, family in FAMILIES.items() if tasks and tasks <= family), set(VIEW_KEYS.values()))


def flags_path(json_path: Path) -> Path:
    return json_path.with_name("flags.json")


def read_flags(path: Path) -> dict:
    """Per model folder, {"overlap": true, "evidence": "<url>"}; empty when the file is absent."""
    return json.loads(path.read_text()) if path.exists() else {}


def flags_problems(flags) -> list[str]:
    """What is wrong with the shape of a flags.json."""
    if not isinstance(flags, dict):
        return ["flags.json must map model folders to {\"overlap\": true, \"evidence\": \"<url>\"}"]
    out = []
    for model, flag in flags.items():
        if not SLUG.match(model):
            out.append(f"flags.json: {model!r} is not a model folder name")
        if (not isinstance(flag, dict) or set(flag) != {"overlap", "evidence"} or flag["overlap"] is not True
                or not isinstance(flag["evidence"], str) or not URL.match(flag["evidence"])):
            out.append(f"flags.json: {model} must be {{\"overlap\": true, \"evidence\": \"<url>\"}}")
    return out


def model_of(pid: str) -> str:
    """The results file's stem of a pipeline id, ``[<encoder>+]<model>[:<variant>]``."""
    return pid.split(":")[0].split("+")[-1]


def overlap_problems(data: dict, flags: dict) -> list[str]:
    """The JSON must carry exactly the flags: overlap and overlap_evidence on every pipeline of a flagged model only."""
    out = []
    for p in data["pipelines"]:
        flag = flags.get(model_of(p["id"]))
        want = {"overlap": True, "overlap_evidence": flag["evidence"]} if flag else {}
        if {k: p[k] for k in ("overlap", "overlap_evidence") if k in p} != want:
            out.append(f"{p['id']}: overlap fields do not match flags.json")
    return out


def folds_expected() -> dict:
    """Fold rows expected per view key (both, reg, clf)."""
    tables = json.loads(MANIFEST.read_text())["tables"]
    return {VIEW_KEYS[v]: n for v, n in expected_folds(tables).items()}


def summarize(folds: pd.DataFrame, expected: dict, views=tuple(VIEW_KEYS.values())) -> dict:
    """Per view: [median score, median total s per 1K samples, fold rows, fold rows expected, mean score] or None.

    ``folds`` are one pipeline's rows of the per-fold table; views not in ``views`` (card_views) are None.
    """
    out = {}
    for view, tasks in VIEWS.items():
        r = folds[folds["task"].isin(tasks)]
        out[VIEW_KEYS[view]] = ([round(float(r["score"].median()), 4), _num(r["time_per_1k_s"].median()), len(r),
                                 expected[VIEW_KEYS[view]], round(float(r["score"].mean()), 4)]
                                if len(r) and VIEW_KEYS[view] in views else None)
    return out


def per_fold(rows: pd.DataFrame, record: dict) -> pd.DataFrame:
    """One pipeline's fold rows in PER_FOLD columns. Only rows with a score count; check_results flags the others."""
    rows = per_1k(rows[rows["score"].notna()])
    out = rows.drop(columns="total_s", errors="ignore").rename(columns={"total": "total_s", "total_per_1k": "time_per_1k_s"})
    out = out.assign(id=record["id"], encoder=record["enc"], learner=record["lrn"], overlap=bool(record.get("overlap")),
                     api=bool(record.get("api")))
    return out[PER_FOLD].sort_values(["table", "fold"]).reset_index(drop=True)


def _num(x):
    return None if x is None or math.isnan(float(x)) else round(float(x), 4)


def build(results_dir: Path, out: Path = ROOT / "site" / "results.json", models=None) -> dict:
    expected = folds_expected()
    flags = read_flags(flags_path(out))
    if flags_problems(flags):
        sys.exit("\n".join(flags_problems(flags)))
    pipelines, folds = [], []
    paths = [results_dir / f"{m}.csv" for m in sorted(models)] if models else sorted(results_dir.glob("*.csv"))
    for path in paths:
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
            if hw == ["api"]:
                versions = sorted(set(rows["api_version"].dropna().astype(str))) if "api_version" in rows else []
                record.update(api=True, **({"api_version": ", ".join(versions)} if versions else {}))
            if path.stem in flags:
                record.update(overlap=True, overlap_evidence=flags[path.stem]["evidence"])
            folds.append(per_fold(rows, record))
            record.update(summarize(folds[-1], expected, card_views(card)))
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
            score, time, n, expected, mean = p[view]
            rows.append({"version": data["version"], "date": data["built"], "view": view, "view_name": VIEW_NAMES[view],
                         "id": p["id"], "encoder": p["enc"], "learner": p["lrn"], "kind": p["kind"], "variant": p["variant"],
                         "median_score": score, "mean_score": mean, "median_time_per_1k_s": time, "fold_rows": n, "fold_rows_expected": expected,
                         "pareto_front": p["id"] in front, "overlap": bool(p.get("overlap")), "api": bool(p.get("api"))})
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
    """The JSON's medians, means and fold rows, recomputed from the per-fold Parquet; the CSV must hold the same rows."""
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
        views = card_views(card_of(model_of(p["id"])))
        if p["id"] in by_id and summarize(by_id[p["id"]], expected, views) != {v: p.get(v) for v in VIEW_KEYS.values()}:
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
    for p in data["pipelines"]:
        if ("overlap" in p or "overlap_evidence" in p) and (p.get("overlap") is not True
                                                            or not URL.match(str(p.get("overlap_evidence")))):
            out.append(f"{p.get('id')}: overlap must be true with overlap_evidence, a URL")
        if ("api" in p or "api_version" in p) and (p.get("api") is not True or not isinstance(p.get("api_version", ""), str)):
            out.append(f"{p.get('id')}: api must be true, with api_version a string")
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
            if (len(v) != 5 or v[0] is None or v[0] > 1 or v[1] is None or v[1] <= 0
                    or not 0 < v[2] <= v[3] or v[3] != expected[view] or v[4] is None or v[4] > 1):
                out.append(f"{p.get('id')} {view}: expected [score <= 1, time > 0, fold rows, {expected[view]}, mean <= 1]")
            elif view == "clf" and min(v[0], v[4]) < 0:
                out.append(f"{p.get('id')} {view}: AUC below 0")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=os.environ.get("STRABLE_RESULTS"))
    parser.add_argument("--models", nargs="+", help="only these results files")
    parser.add_argument("--check", type=Path, help="validate this JSON instead of building")
    args = parser.parse_args()
    if args.check:
        data = json.loads(args.check.read_text())
        found = problems(data)
        if not found and flags_path(args.check).exists():
            flags = read_flags(flags_path(args.check))
            found = flags_problems(flags) or overlap_problems(data, flags)
        if not found and args.check.with_suffix(".parquet").exists():
            found = parquet_problems(data, args.check.with_suffix(".parquet"))
        if not found and per_fold_path(args.check).exists():
            found = per_fold_problems(data, per_fold_path(args.check))
        print("\n".join(found) if found else f"{args.check}: ok")
        sys.exit(1 if found else 0)
    if not args.results:
        sys.exit("give --results or set STRABLE_RESULTS")
    build(args.results, models=args.models)
