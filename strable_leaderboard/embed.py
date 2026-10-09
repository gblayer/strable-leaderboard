"""Fill the language-model embedding cache, one table at a time, before any model runs.

    python -m strable_leaderboard.embed --list --cache-dir cache          # missing "embed <encoder> <table>" lines
    python -m strable_leaderboard.embed --encoder llama-3.1-8b --table <table> --cache-dir cache --device cuda

Every numeric-only model reads ``<cache>/embeddings/<encoder>/<table>.parquet``. Computing it
inside the model tasks would make several tasks (3 folds, several models) write the same file at
once, so the runner submits one array of these tasks first, and the model tasks then only read
the cache (read-only inside the sandbox).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import list_tables, load_table
from .encoders import ENCODERS, embed_table

LM_ENCODERS = [e for e, spec in ENCODERS.items() if spec["checkpoint"]]


def missing(cache_dir: Path, encoders=LM_ENCODERS, tables=None) -> list[str]:
    """Task lines ``embed <encoder> <table>`` for every table not yet in the cache."""
    tables = tables if tables is not None else list_tables()
    return [f"embed {e} {t}" for e in encoders for t in tables
            if not (Path(cache_dir) / "embeddings" / e / f"{t}.parquet").exists()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="print the missing tasks")
    parser.add_argument("--encoder", choices=LM_ENCODERS)
    parser.add_argument("--table")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(missing(args.cache_dir)))
        return 0
    if not (args.encoder and args.table):
        parser.error("give --list, or --encoder and --table")
    X, _, _ = load_table(args.table, data_dir=args.cache_dir / "data")
    embeddings, seconds = embed_table(X, args.encoder, args.table, args.cache_dir, args.device)
    print(f"{args.encoder} {args.table}: {len(embeddings)} strings, {seconds:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
