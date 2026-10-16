"""Run one pipeline on one table and one outer fold, and write its result row.

    python -m strable_leaderboard.run models/<name> --encoder tfidf --table <table> --fold 0 \\
        --data-dir <cache>/data --cache-dir <cache> --out runs
    python -m strable_leaderboard.run models/<name> --smoke --data-dir <cache>/data --cache-dir <cache>

This process is trusted code and never imports the model. It loads the table and the fold,
encodes the strings, and writes only X_train, y_train and X_test to a fresh temporary folder. The
model runs in a child process, ``strable_leaderboard.fit_predict``, inside scripts/sandbox.sh
(Singularity, no network, no dataset, no folds, no results), which writes its predictions and
nothing else. This process checks them, scores them and writes the row to
``<out>/<model>/<encoder>/<table>/<fold>.json``, a folder the child cannot see. An existing row is
skipped, so a job array can be resubmitted after a preemption.

Time, measured here and never by the model: two warm-up children run
first; the wall-clock time of the second is the start-up overhead O. The timed child fits on the
training rows and predicts the test rows; its wall-clock time is T. Then
``total_s = encoding + max(T - O, 0.001)``, where encoding is the table's cached language-model
extraction time plus fitting and applying the encoder. Fit and predict are not timed apart.
Both processes cap their BLAS, OpenMP and torch threads to ``--n-jobs``, the job's CPUs.

``--smoke``: the static checks, then outer fold 0 of the three smoke tables through the same
sandbox; nothing is written. ``--no-sandbox`` runs the child as a plain process, for local
development only: the cluster never passes it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .card import load_card
from .data import load_table
from .encoders import encode, warm_up
from .fit_predict import limit_threads, thread_env
from .metrics import checked_score
from .smoke import SMOKE_TABLES, static_problems
from .splits import splits_from_table

PROTOCOL = "0.1"
ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "splits"
SANDBOX = ROOT / "scripts" / "sandbox.sh"
MIN_SECONDS = 0.001


def hardware_name(device: str) -> str:
    """What the row ran on: "cpu", or the GPU model."""
    if device.startswith("cuda"):
        import torch

        return torch.cuda.get_device_name(0)
    return "cpu"


def wall_clock(cmd: list, env: dict) -> float:
    t0 = time.perf_counter()
    subprocess.run([str(c) for c in cmd], check=True, env=env)
    return time.perf_counter() - t0


def fit_predict(folder: Path, X_train, y_train, X_test, task: str, strings: bool, weights_dir, device: str,
                n_jobs: int, sandbox: bool = True) -> tuple[dict, float]:
    """The child's outputs as arrays, and its time: timed run minus the start-up overhead."""
    with tempfile.TemporaryDirectory(prefix="strable-") as tmp:
        inputs, outputs = Path(tmp) / "in", Path(tmp) / "out"
        inputs.mkdir()
        outputs.mkdir()
        X_train.to_parquet(inputs / "X_train.parquet")
        X_test.to_parquet(inputs / "X_test.parquet")
        np.save(inputs / "y_train.npy", y_train)
        cmd = [sys.executable, "-m", "strable_leaderboard.fit_predict", folder.resolve(), inputs, outputs,
               "--task", task, "--device", device, "--n-jobs", n_jobs] + (["--strings"] if strings else [])
        if weights_dir:
            cmd += ["--weights-dir", weights_dir]
        if sandbox:
            cmd = ["bash", SANDBOX, folder, inputs, outputs] + cmd
        env = dict(os.environ, **thread_env(n_jobs))  # scripts/sandbox.sh passes them on
        overhead = [wall_clock(cmd + ["--warm-up"], env) for _ in range(2)][-1]  # the first one warms the caches
        seconds = max(wall_clock(cmd, env) - overhead, MIN_SECONDS)
        names = ["pred"] if task == "regression" else ["proba", "classes"]
        return {n: np.load(outputs / f"{n}.npy", allow_pickle=False) for n in names}, seconds


def run(folder: Path, encoder: str, table: str, fold: int, data_dir, cache_dir, device: str, n_jobs: int,
        sandbox: bool = True) -> dict:
    folder = Path(folder)
    card = load_card(folder)
    strings = card["end_to_end"]
    X, y, task = load_table(table, data_dir=data_dir)
    train_idx, test_idx, _ = splits_from_table(pd.read_parquet(SPLITS / f"{table}.parquet"), fold)
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    encode_s = 0.0
    if not strings:
        warm_up(encoder)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories")
            X_train, X_test, encode_s = encode(X_train, X_test, encoder, table, cache_dir, device)
    weights_dir = Path(cache_dir) / "weights" / folder.name if cache_dir else None
    out, model_s = fit_predict(folder, X_train, y_train, X_test, task, strings, weights_dir, device, n_jobs, sandbox)
    score = checked_score(task, y_train, y_test, **out)
    checkpoint = card.get("checkpoint") or {}
    return {
        "model": folder.name, "encoder": "none" if strings else encoder, "variant": card["variant"], "table": table,
        "task": task, "fold": fold, "score": score, "fit_s": None, "predict_s": None,
        "total_s": round(encode_s + model_s, 4),
        "n_train": len(train_idx), "n_test": len(test_idx), "hardware": hardware_name(device),
        "package": card["package"], "checkpoint_revision": checkpoint.get("revision", ""),
        "protocol_version": PROTOCOL, "date": date.today().isoformat(), "leaderboard_version": __version__,
    }


def smoke(folder: Path, data_dir, cache_dir, device: str, n_jobs: int, sandbox: bool) -> int:
    """The smoke test through the sandbox: static checks, then fold 0 of the smoke tables."""
    found = static_problems(folder)
    for p in found:
        print(f"  FAIL  {p}")
    if found:
        return 1
    card, failures = load_card(folder), 0
    for task, table in SMOKE_TABLES.items():
        if task not in card["tasks"]:
            continue
        try:
            row = run(folder, "tfidf", table, 0, data_dir, cache_dir, device, n_jobs, sandbox)
            print(f"  ok    {task:<10} {table:<34} score {row['score']:.3f}  total {row['total_s']:.2f}s")
        except Exception as exc:  # report every table, then exit non-zero
            failures += 1
            print(f"  FAIL  {task:<10} {table:<34} {type(exc).__name__}: {exc}")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="models/<name>")
    parser.add_argument("--encoder", default="tfidf", help="tfidf, qwen-3-8b-first30, llama-3.1-8b-first30, minilm-l6-v2-pca30; ignored for end-to-end models")
    parser.add_argument("--table")
    parser.add_argument("--fold", type=int, choices=[0, 1, 2])
    parser.add_argument("--smoke", action="store_true", help="fold 0 of the three smoke tables, nothing written")
    parser.add_argument("--data-dir", type=Path, default=None, help="local copy of the dataset")
    parser.add_argument("--cache-dir", type=Path, default=None, help="embeddings and weights")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-sandbox", action="store_true", help="local development only")
    args = parser.parse_args(argv)
    limit_threads(args.n_jobs)  # the encoders, here; the child gets the same cap
    if args.device == "auto":
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    sandbox = not args.no_sandbox
    if args.smoke:
        return smoke(args.folder, args.data_dir, args.cache_dir, args.device, args.n_jobs, sandbox)
    if args.table is None or args.fold is None:
        parser.error("give --table and --fold, or --smoke")
    encoder = "none" if load_card(args.folder)["end_to_end"] else args.encoder
    path = args.out / args.folder.name / encoder / args.table / f"{args.fold}.json"
    if path.exists():
        print("exists", path)
        return 0
    row = run(args.folder, encoder, args.table, args.fold, args.data_dir, args.cache_dir, args.device, args.n_jobs,
              sandbox)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1))
    print(f"{row['model']} {row['encoder']} {row['table']} fold {row['fold']}: score {row['score']:.4f} "
          f"total {row['total_s']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
