"""Check a results file before it reaches the board.

    python -m strable_leaderboard.check_results results/<model>.csv

One row per (encoder, variant, table, outer fold). Prints the problems found and the coverage
per view (fold rows with a score out of those expected), and exits non-zero on any problem. Missing
tables or folds are not a problem: the board shows the pipeline with its coverage and an
incomplete tag. The Checks workflow runs it on every results pull request, and the Claude
review reads its output.

A row carries ``total_s`` (the runner and the paper time fit and predict together) or, in older
files, ``fit_s`` and ``predict_s``. ``variant`` is ``default`` or ``tuned``. Rows of an API model
(``hardware`` "api") also carry ``api_version``, the version its provider reported: one, on every row,
since a version that changes during the run mixes two models in one pipeline.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

from .data import MANIFEST
from .metrics import VARIANTS, VIEWS, expected_folds
from .splits import N_OUTER

COLUMNS = ["model", "encoder", "variant", "table", "task", "fold", "score", "fit_s", "predict_s", "total_s", "n_train",
           "n_test", "hardware", "package", "checkpoint_revision", "protocol_version", "date"]
ENCODER_ID = re.compile(r"^[a-z0-9][a-z0-9.+-]*$")


def check(path: Path, manifest: dict) -> tuple[list[str], list[str]]:
    """Return (problems, coverage lines) for one results file."""
    problems: list[str] = []
    tables = manifest["tables"]
    df = pd.read_csv(path)

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns {missing}"], []
    if set(df["model"]) != {path.stem}:
        problems.append(f"model must be {path.stem!r} on every row, found {sorted(set(df['model']))}")
    unknown = sorted(set(df["table"]) - set(tables))
    if unknown:
        problems.append(f"unknown tables {unknown[:5]}")
    bad_enc = sorted(e for e in set(df["encoder"].astype(str)) if not ENCODER_ID.match(e))
    if bad_enc:
        problems.append(f"encoder ids must be lowercase slugs, found {bad_enc}")
    bad_variant = sorted(set(df["variant"].astype(str)) - set(VARIANTS))
    if bad_variant:
        problems.append(f"variant must be one of {list(VARIANTS)}, found {bad_variant}")
    dup = df.duplicated(["encoder", "variant", "table", "fold"]).sum()
    if dup:
        problems.append(f"{dup} duplicate (encoder, variant, table, fold) rows")
    if df["score"].isna().any():
        problems.append(f"{df['score'].isna().sum()} rows without a score")
    if not df["fold"].isin(range(N_OUTER)).all():
        problems.append("fold must be 0, 1 or 2")
    for col in ["hardware", "package", "protocol_version", "date"]:
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            problems.append(f"empty values in {col}")
    api = df["hardware"].astype(str) == "api"
    if api.any():
        versions = df.loc[api, "api_version"] if "api_version" in df else pd.Series([None] * int(api.sum()))
        if versions.isna().any() or (versions.astype(str).str.strip() == "").any():
            problems.append("rows with hardware api need the api_version their provider reported")
        elif versions.nunique() > 1:
            problems.append(f"api_version changed during the run: {sorted(versions.astype(str).unique())}")
    split = df["fit_s"].notna() & df["predict_s"].notna()
    if not (split | df["total_s"].notna()).all():
        problems.append("each row needs fit_s and predict_s, or total_s")
    if (df.loc[split, ["fit_s", "predict_s"]] <= 0).any().any() or (df["total_s"].dropna() <= 0).any():
        problems.append("times must be positive")

    known = df[df["table"].isin(tables)]
    task = known["table"].map(lambda t: tables[t]["task"])
    if (known["task"] != task).any():
        problems.append("task does not match the manifest")
    clf = known[task != "regression"]
    if ((clf["score"] < 0) | (clf["score"] > 1)).any():
        problems.append("classification scores must be within [0, 1]")
    if (known[task == "regression"]["score"] > 1).any():
        problems.append("R² cannot exceed 1")
    n_test = [tables[t]["test_rows_per_outer_fold"][f] for t, f in zip(known["table"], known["fold"])]
    n_rows = known["table"].map(lambda t: tables[t]["n_rows"])
    if (known["n_test"] != n_test).any() or (known["n_train"] != n_rows - known["n_test"]).any():
        problems.append("n_train or n_test disagree with the manifest's folds")

    coverage, expected = [], expected_folds(tables)
    for (encoder, variant), g in known[known["score"].notna()].groupby(["encoder", "variant"]):
        for view, view_tasks in VIEWS.items():
            n = int(g["task"].isin(view_tasks).sum())
            coverage.append(f"{encoder:<22} {variant:<8} {view:<15} {n}/{expected[view]} folds"
                            + ("" if n == expected[view] else " incomplete"))
    return problems, coverage


def main(argv=None) -> int:
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        print(__doc__)
        return 2
    manifest = json.loads(Path(MANIFEST).read_text())
    status = 0
    for path in paths:
        problems, coverage = check(path, manifest)
        print(path)
        for line in coverage:
            print("  ", line)
        for p in problems:
            print("   PROBLEM", p)
        status |= bool(problems)
    return status


if __name__ == "__main__":
    sys.exit(main())
