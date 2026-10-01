"""Import the paper's result_comparison*.csv files into results/<model>.csv.

    python scripts/import_paper_results.py result_comparison.csv [more.csv ...] --out $STRABLE_RESULTS

Run it where the paper's CSVs are, on the cluster. The paper timed fit and predict together,
so imported rows carry ``total_s`` only and ``hardware: paper`` (mixed GPUs and CPUs). Scores
are the raw per-fold R² or ROC-AUC. Only default-configuration ``num-str`` pipelines are
imported; add ``--encoders`` or ``--learners`` to restrict further.

Paper method tokens are ``num-str_<encoder>_<learner>``. Unknown tokens are listed, not
imported: extend the maps below for them. A ``data_name`` that is not a table of the dataset
(a table renamed before the Hugging Face release) goes into ``RENAMES`` as old name -> new name.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strable_leaderboard.check_results import COLUMNS  # noqa: E402
from strable_leaderboard.data import MANIFEST  # noqa: E402

# paper encoder token -> leaderboard encoder id. The decoders use their first-30-dims rows.
ENCODERS = {
    "tabvec": "tfidf", "tarenc": "targetencoder", "tarte": "tarte",
    "llm-all-MiniLM-L6-v2": "minilm-l6-v2", "llm-e5-small-v2": "e5-small-v2", "llm-fasttext": "fasttext",
    "llm-llama-3.1-8b": "llama-3.1-8b", "llm-qwen3-8b": "qwen-3-8b", "llm-jasper-token-comp-0.6b": "jasper-0.6b",
}
# paper learner token -> model folder. End-to-end models have the learner as encoder token too.
LEARNERS = {
    "ridge_default": "ridge", "xgb_default": "xgboost", "extrees_default": "extratrees",
    "tabpfn_default": "tabpfn-2.5", "tabicl": "tabicl-v2", "tabm": "tabm", "realmlp": "realmlp",
    "catboost_default": "catboost", "contexttab": "contexttab", "tabstar": "tabstar", "mambular": "mambular",
}
END_TO_END = {"catboost", "contexttab", "tabstar", "mambular"}
RENAMES: dict[str, str] = {}  # paper data_name -> dataset table name, for tables renamed on Hugging Face


def convert(paper: pd.DataFrame, tables: dict, encoders=None, learners=None) -> tuple[pd.DataFrame, list[str]]:
    skipped = []
    rows = []
    for _, r in paper.iterrows():
        parts = str(r["method"]).split("_", 2)
        if len(parts) != 3 or parts[0] != "num-str":
            skipped.append(r["method"])
            continue
        enc_tok, lrn_tok = parts[1], parts[2]
        if lrn_tok.endswith("_tune") or lrn_tok not in LEARNERS or (enc_tok not in ENCODERS and lrn_tok not in END_TO_END):
            skipped.append(r["method"])
            continue
        model = LEARNERS[lrn_tok]
        encoder = "none" if lrn_tok in END_TO_END else ENCODERS[enc_tok]
        if (encoders and encoder not in encoders) or (learners and model not in learners):
            continue
        table = RENAMES.get(r["data_name"], r["data_name"])
        if table not in tables:
            skipped.append(f"table {r['data_name']}")
            continue
        fold = int(r["fold_index"])
        n_test = tables[table]["test_rows_per_outer_fold"][fold]
        score = r["r2"] if pd.notna(r["r2"]) else r["roc_auc"]
        rows.append(dict(model=model, encoder=encoder, table=table, task=tables[table]["task"], fold=fold, score=score,
                         fit_s=None, predict_s=None, total_s=r["run_time"], n_train=tables[table]["n_rows"] - n_test,
                         n_test=n_test, hardware="paper", package="paper", checkpoint_revision="",
                         protocol_version="0.1", date="2026-05"))
    return pd.DataFrame(rows, columns=COLUMNS), sorted(set(skipped))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=os.environ.get("STRABLE_RESULTS", "results"))
    parser.add_argument("--encoders", nargs="*", help="leaderboard encoder ids to keep")
    parser.add_argument("--learners", nargs="*", help="model folders to keep")
    args = parser.parse_args(argv)
    tables = json.loads(MANIFEST.read_text())["tables"]
    paper = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    df, skipped = convert(paper, tables, args.encoders, args.learners)
    df = df.drop_duplicates(["model", "encoder", "table", "fold"], keep="last")
    args.out.mkdir(exist_ok=True)
    for model, rows in df.groupby("model"):
        out = args.out / f"{model}.csv"
        rows.sort_values(["encoder", "table", "fold"]).to_csv(out, index=False)
        print(out, len(rows), "rows,", rows["encoder"].nunique(), "encoders")
    if skipped:
        print("not imported:", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
