"""Write the outer and inner fold ids of every STRABLE table to splits/.

    python scripts/export_splits.py --data-dir /path/to/STRABLE-benchmark

``--data-dir`` is a local copy of the Hugging Face dataset at the pinned revision, for
example from ``huggingface_hub.snapshot_download``. Without it, tables are downloaded one
by one. Each ``splits/<table>.parquet`` has one row per table row, in the row order of
``data.parquet``; see ``strable_leaderboard.splits.fold_table`` for the columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sklearn

from strable_leaderboard import HF_DATASET, HF_REVISION
from strable_leaderboard.data import load_table
from strable_leaderboard.splits import N_INNER, N_OUTER, SEED, fold_table

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "splits")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.data_dir is not None:
        names = sorted(p.parent.name for p in args.data_dir.glob("*/data.parquet"))
    else:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(HF_DATASET, repo_type="dataset", revision=HF_REVISION)
        names = sorted({f.split("/")[0] for f in files if f.endswith("/data.parquet")})

    tables = {}
    for name in names:
        X, y, task = load_table(name, data_dir=args.data_dir)
        folds = fold_table(y, task)
        folds.to_parquet(args.out / f"{name}.parquet", index=False)
        sha = hashlib.sha256((args.data_dir / name / "data.parquet").read_bytes()).hexdigest() if args.data_dir else None
        tables[name] = {
            "task": task,
            "n_rows": int(len(y)),
            "test_rows_per_outer_fold": folds["outer_fold"].value_counts().sort_index().astype(int).tolist(),
            "data_parquet_sha256": sha,
        }
        print(f"{name:<55} {task:<10} {len(y):>6} rows")

    manifest = {
        "dataset": HF_DATASET,
        "revision": HF_REVISION,
        "seed": SEED,
        "outer_folds": N_OUTER,
        "inner_folds": N_INNER,
        "scikit_learn": sklearn.__version__,
        "tables": tables,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(tables)} tables written to {args.out}")


if __name__ == "__main__":
    main()
