"""Import the paper's Figure 4a pipelines into results/<model>.csv.

    python scripts/import_paper_results.py salts/results/compiled_results --out $STRABLE_RESULTS

Run it where the paper's CSVs are, on the cluster. Exactly the pipelines listed in
``scripts/paper_selection.yaml`` are imported, by source CSV and method token, nothing else; a
listed pipeline missing from its CSV is an error. The paper timed fit and predict together, so
imported rows carry ``total_s`` only (the paper's ``run_time``), with ``hardware`` and
``package`` set to ``paper``. Scores are the raw per-fold R² or ROC-AUC.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strable_leaderboard.check_results import COLUMNS  # noqa: E402
from strable_leaderboard.data import MANIFEST  # noqa: E402
from strable_leaderboard.run import PROTOCOL  # noqa: E402

SELECTION = ROOT / "scripts" / "paper_selection.yaml"


def convert(frames: dict[str, pd.DataFrame], tables: dict, pipelines: dict) -> tuple[pd.DataFrame, list[str]]:
    """Rows of the selected pipelines, and the problems found (missing pipelines, unknown tables).

    ``frames`` maps a source CSV name to its contents, ``pipelines`` is the selection's
    ``pipelines`` section: CSV name -> method token -> model, encoder, variant.
    """
    problems, rows = [], []
    for csv, methods in pipelines.items():
        paper = frames[csv]
        for method, p in methods.items():
            sel = paper[paper["method"] == method]
            if sel.empty:
                problems.append(f"{csv}: no rows for {method}")
            for _, r in sel.iterrows():
                table = r["data_name"]
                if table not in tables:
                    problems.append(f"{csv}: unknown table {table}")
                    continue
                fold = int(r["fold_index"])
                n_test = tables[table]["test_rows_per_outer_fold"][fold]
                score = r["r2"] if pd.notna(r["r2"]) else r["roc_auc"]
                rows.append(dict(model=p["model"], encoder=p["encoder"], variant=p["variant"], table=table,
                                 task=tables[table]["task"], fold=fold, score=score, fit_s=None, predict_s=None,
                                 total_s=r["run_time"], n_train=tables[table]["n_rows"] - n_test, n_test=n_test,
                                 hardware="paper", package="paper", checkpoint_revision="", protocol_version=PROTOCOL,
                                 date="2026-05"))
    df = pd.DataFrame(rows, columns=COLUMNS)
    dup = df.duplicated(["model", "encoder", "variant", "table", "fold"]).sum()
    if dup:
        problems.append(f"{dup} duplicate (model, encoder, variant, table, fold) rows")
    return df, sorted(set(problems))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("compiled", type=Path, help="the paper's results/compiled_results folder")
    parser.add_argument("--out", type=Path, default=os.environ.get("STRABLE_RESULTS", "results"))
    args = parser.parse_args(argv)
    tables = json.loads(MANIFEST.read_text())["tables"]
    selection = yaml.safe_load(SELECTION.read_text())
    pipelines = selection["pipelines"]
    frames = {csv: pd.read_csv(args.compiled / csv) for csv in pipelines}
    df, problems = convert(frames, tables, pipelines)
    if problems:
        print("\n".join(problems))
        return 1
    args.out.mkdir(exist_ok=True)
    for model, rows in df.groupby("model"):
        out = args.out / f"{model}.csv"
        rows.sort_values(["encoder", "variant", "table", "fold"]).to_csv(out, index=False)
        print(out, len(rows), "rows,", len(rows.groupby(["encoder", "variant"])), "pipelines")
    print(len(df.groupby(["model", "encoder", "variant"])), "pipelines imported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
