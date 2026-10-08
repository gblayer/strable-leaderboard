"""String encoders for models that only take numbers.

Every numeric-only model is paired with three encoders. Each string column becomes 30 numbers:

| id             | Encoder          | Checkpoint                              | Reduction               |
|----------------|------------------|-----------------------------------------|-------------------------|
| tfidf          | Tf-Idf           | none, character 3-4 grams               | truncated SVD           |
| minilm-l6-v2   | all-MiniLM-L6-v2 | sentence-transformers/all-MiniLM-L6-v2  | first 30 embedding dims |
| llama-3.1-8b   | LLaMA-3.1-8B     | meta-llama/Llama-3.1-8B                 | first 30 embedding dims |

The two language models embed every distinct string of a table once (``embed_table``), as in
the paper; the embeddings and their extraction time are cached per table. Rows imported from
the paper name other encoders too, such as 30-component PCA of the same models; their ids and
display names are in ``scripts/paper_selection.yaml``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 1234
N_DIMS = 30

ENCODERS = {
    "tfidf": {"checkpoint": None, "reduction": "svd-30"},
    "minilm-l6-v2": {"checkpoint": "sentence-transformers/all-MiniLM-L6-v2", "reduction": "first-30"},
    "llama-3.1-8b": {"checkpoint": "meta-llama/Llama-3.1-8B", "reduction": "first-30"},
}
NAMES = {"none": "", "tfidf": "Tf-Idf", "minilm-l6-v2": "MiniLM-L6-v2", "llama-3.1-8b": "LLaMA-3.1-8B"}


def tfidf_encoder(seed: int = SEED):
    """The paper's Tf-Idf encoder: skrub's TableVectorizer with a 30-component StringEncoder.

    Columns with fewer than 40 distinct values are one-hot encoded, dates are expanded,
    and numbers go through a SquashingScaler.
    """
    from skrub import SquashingScaler, StringEncoder, TableVectorizer

    return TableVectorizer(high_cardinality=StringEncoder(n_components=N_DIMS, random_state=seed), numeric=SquashingScaler())


def string_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("string", "category")]


def embed_table(X: pd.DataFrame, encoder: str, table: str, cache_dir: Path, device: str = "cuda") -> tuple[pd.DataFrame, float]:
    """Embeddings of every distinct string of the table, and the seconds it took.

    Cached as ``<cache_dir>/embeddings/<encoder>/<table>.parquet`` with a ``name`` column and
    one column per dimension, plus ``<table>.time``. Mirrors the paper's extraction script:
    sentence-transformers, no normalization, batch 32 (16 for the 8B model).
    """
    folder = Path(cache_dir) / "embeddings" / encoder
    path, time_path = folder / f"{table}.parquet", folder / f"{table}.time"
    if path.exists():
        return pd.read_parquet(path), float(time_path.read_text())
    from sentence_transformers import SentenceTransformer

    checkpoint = ENCODERS[encoder]["checkpoint"]
    weights = Path(cache_dir) / "weights" / encoder
    model = SentenceTransformer(str(weights) if weights.exists() else checkpoint, device=device)
    if "llama" in encoder:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    strings = pd.unique(pd.concat([X[c].dropna().astype(str) for c in string_columns(X)], ignore_index=True))
    t0 = time.perf_counter()
    vectors = model.encode(np.asarray(strings), convert_to_numpy=True, batch_size=16 if "8b" in encoder else 32,
                           normalize_embeddings=False)
    seconds = time.perf_counter() - t0
    table_embeddings = pd.DataFrame(vectors[:, :N_DIMS], columns=[f"X{i}" for i in range(N_DIMS)])
    table_embeddings.insert(0, "name", strings)
    folder.mkdir(parents=True, exist_ok=True)
    table_embeddings.to_parquet(path, index=False)
    time_path.write_text(f"{seconds:.4f}")
    return table_embeddings, seconds


def first_dims_encoder(embeddings: pd.DataFrame):
    """TableVectorizer that replaces every string column by the first 30 embedding dimensions.

    Missing strings stay missing. Numbers go through a SquashingScaler, as in the paper.
    """
    from sklearn.preprocessing import FunctionTransformer
    from skrub import SquashingScaler, TableVectorizer

    lookup = embeddings.set_index("name")

    def replace(column: pd.DataFrame) -> pd.DataFrame:
        name = column.columns[0]
        values = column[name]
        out = lookup.reindex(values.astype(str).where(values.notna(), None)).to_numpy(dtype=float)
        out[values.isna().to_numpy()] = np.nan
        return pd.DataFrame(out, index=column.index).add_prefix(f"{name}_")

    return TableVectorizer(cardinality_threshold=0, high_cardinality=FunctionTransformer(replace), numeric=SquashingScaler())


def warm_up(encoder: str) -> None:
    """Import what ``encode`` needs, so that imports are not timed."""
    import skrub  # noqa: F401

    if encoder != "tfidf":
        import sentence_transformers  # noqa: F401


def encode(X_train, X_test, encoder: str, table: str, cache_dir=None, device: str = "cuda"):
    """Numeric train and test frames for one encoder, and the embedding seconds for the whole table.

    The encoder is fitted on the training rows only. The embedding time is 0 for Tf-Idf; for the
    language models it is the cached extraction time of the table, measured once.
    """
    if encoder == "tfidf":
        enc, seconds = tfidf_encoder(), 0.0
    else:
        embeddings, seconds = embed_table(pd.concat([X_train, X_test]), encoder, table, Path(cache_dir), device)
        enc = first_dims_encoder(embeddings)
    return enc.fit_transform(X_train), enc.transform(X_test), seconds
