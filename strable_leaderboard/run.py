"""Run one pipeline on one table and one outer fold, and write its result row.

    python -m strable_leaderboard.run models/<name> --encoder tfidf --table <table> --fold 0 \\
        --data-dir <cache>/data --cache-dir <cache> --out runs

The row lands in ``<out>/<model>/<encoder>/<table>/<fold>.json`` and is skipped when it exists,
so a job array can be resubmitted after preemption. ``scripts/gather_results.py`` turns the rows
into ``results/<model>.csv``.

Times: fit = embedding of the training rows + training; predict = embedding of the test rows
+ prediction. The language-model embedding of a table is measured once and shared between
fit and predict in proportion to the rows. An untimed warm-up on 64 rows loads the weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .data import load_table
from .encoders import encode, warm_up
from .metrics import fold_score
from .smoke import load_card, load_model_class
from .splits import SEED, splits_from_table

PROTOCOL = "0.1"
SPLITS = Path(__file__).resolve().parent.parent / "splits"


def hardware_name(device: str) -> str:
    """What the row ran on: "cpu", or the GPU model."""
    if device.startswith("cuda"):
        import torch

        return torch.cuda.get_device_name(0)
    return "cpu"


def run(folder: Path, encoder: str, table: str, fold: int, data_dir, cache_dir, device: str, n_jobs: int) -> dict:
    card, cls = load_card(folder), load_model_class(folder)
    X, y, task = load_table(table, data_dir=data_dir)
    train_idx, test_idx, _ = splits_from_table(pd.read_parquet(SPLITS / f"{table}.parquet"), fold)
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    weights_dir = Path(cache_dir) / "weights" / folder.name if cache_dir else None
    make = lambda: cls(task=task, seed=SEED, device=device, n_jobs=n_jobs, weights_dir=weights_dir)  # noqa: E731

    fit_s = predict_s = 0.0
    if not cls.handles_strings:
        warm_up(encoder)
        t0 = time.perf_counter()
        X_train, X_test, embed_s = encode(X_train, X_test, encoder, table, cache_dir, device)
        encode_s = time.perf_counter() - t0
        n = len(train_idx) + len(test_idx)
        fit_s += (embed_s + encode_s) * len(train_idx) / n
        predict_s += (embed_s + encode_s) * len(test_idx) / n

    warm = make()  # untimed: imports, CUDA start-up, weight loading
    small = slice(0, min(64, len(train_idx)))
    try:
        warm.fit(X_train[small] if isinstance(X_train, np.ndarray) else X_train.iloc[small], y_train[small])
        warm.predict(X_test[:8] if isinstance(X_test, np.ndarray) else X_test.iloc[:8])
    except Exception:
        pass

    model = make()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s += time.perf_counter() - t0
    t0 = time.perf_counter()
    if task == "regression":
        pred = np.asarray(model.predict(X_test), dtype=float)
        predict_s += time.perf_counter() - t0
        score = fold_score(task, y_test, y_pred=pred)
    else:
        proba = np.asarray(model.predict_proba(X_test), dtype=float)
        predict_s += time.perf_counter() - t0
        score = fold_score(task, y_test, proba=proba, classes=np.asarray(model.classes_))

    checkpoint = card.get("checkpoint") or {}
    return {
        "model": folder.name, "encoder": encoder if not cls.handles_strings else "none",
        "variant": card.get("variant", "default"), "table": table, "task": task,
        "fold": fold, "score": score, "fit_s": round(fit_s, 4), "predict_s": round(predict_s, 4), "total_s": None,
        "n_train": len(train_idx), "n_test": len(test_idx), "hardware": hardware_name(device),
        "package": card["package"], "checkpoint_revision": checkpoint.get("revision", ""),
        "protocol_version": PROTOCOL, "date": date.today().isoformat(), "leaderboard_version": __version__,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="models/<name>")
    parser.add_argument("--encoder", default="tfidf", help="tfidf, minilm-l6-v2, llama-3.1-8b, or none for end-to-end models")
    parser.add_argument("--table", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--data-dir", type=Path, default=None, help="local copy of the dataset")
    parser.add_argument("--cache-dir", type=Path, default=None, help="embeddings and weights")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.device == "auto":
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = load_model_class(args.folder)
    encoder = "none" if cls.handles_strings else args.encoder
    path = args.out / args.folder.name / encoder / args.table / f"{args.fold}.json"
    if path.exists():
        print("exists", path)
        return 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories")
        row = run(args.folder, encoder, args.table, args.fold, args.data_dir, args.cache_dir, args.device, args.n_jobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1))
    print(f"{row['model']} {row['encoder']} {row['table']} fold {row['fold']}: score {row['score']:.4f} "
          f"fit {row['fit_s']:.2f}s predict {row['predict_s']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
