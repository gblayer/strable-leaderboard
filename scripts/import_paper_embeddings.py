"""Fill the embedding cache with the paper's extracted embeddings. For the maintainer only.

    python scripts/import_paper_embeddings.py <paper repository> --cache-dir cache

The paper's code keeps, per language model and table, ``data/llm_embeding/<model>/<model>|<table>.parquet``
(a ``name`` column, then every dimension) and ``data/llm_embed_time/<model>/<model>|<table>.npy``
(the extraction seconds). This writes them where ``strable_leaderboard.embed`` writes its own,
``<cache>/embeddings/<encoder>/<table>.parquet`` and ``.time``, with the dimensions the encoder
uses (30 for first-30, all for PCA-30). The embed step then finds every table, and a run's time
counts the paper's extraction time. Tables already in the cache are left as they are.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strable_leaderboard.data import list_tables  # noqa: E402
from strable_leaderboard.encoders import ENCODERS, N_DIMS  # noqa: E402

PAPER_MODELS = {"qwen-3-8b-first30": "llm-qwen3-8b", "llama-3.1-8b-first30": "llm-llama-3.1-8b",
                "minilm-l6-v2-pca30": "llm-all-MiniLM-L6-v2"}


def import_table(paper: Path, cache: Path, encoder: str, table: str) -> str:
    model = PAPER_MODELS[encoder]
    source = paper / "data" / "llm_embeding" / model / f"{model}|{table}.parquet"
    source_time = paper / "data" / "llm_embed_time" / model / f"{model}|{table}.npy"
    folder = cache / "embeddings" / encoder
    path, time_path = folder / f"{table}.parquet", folder / f"{table}.time"
    if path.exists():
        return "cached"
    if not (source.exists() and source_time.exists()):
        return "missing"
    seconds = np.load(source_time)
    if seconds.shape != ():
        return "missing"
    first30 = ENCODERS[encoder]["reduction"] == "first-30"
    embeddings = pd.read_parquet(source, columns=["name"] + [f"X{i}" for i in range(N_DIMS)] if first30 else None)
    folder.mkdir(parents=True, exist_ok=True)
    time_path.write_text(f"{float(seconds):.4f}")
    tmp = path.with_suffix(f".tmp{os.getpid()}")  # the parquet file marks the table as done
    embeddings.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return "imported"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", type=Path, help="the paper's repository")
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    missing = []
    for encoder in PAPER_MODELS:
        counts = {}
        for table in list_tables():
            status = import_table(args.paper, args.cache_dir, encoder, table)
            counts[status] = counts.get(status, 0) + 1
            if status == "missing":
                missing.append(f"{encoder} {table}")
        print(encoder, counts)
    print("\n".join(f"missing: {m}" for m in missing))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
