"""String encoders for models that only take numbers.

Every numeric-only model is paired with four encoders, as in the paper. Each string column becomes 30 numbers:

| id                   | Encoder          | Checkpoint                              | Reduction                       |
|----------------------|------------------|-----------------------------------------|---------------------------------|
| tfidf                | Tf-Idf           | none, character 3-4 grams               | truncated SVD                   |
| qwen-3-8b-first30    | Qwen-3-8B        | Qwen/Qwen3-Embedding-8B                 | first 30 embedding dims         |
| llama-3.1-8b-first30 | LLaMA-3.1-8B     | meta-llama/Llama-3.1-8B                 | first 30 embedding dims         |
| minilm-l6-v2-pca30   | all-MiniLM-L6-v2 | sentence-transformers/all-MiniLM-L6-v2  | PCA to 30, fitted per column    |

The language models embed every distinct string of a table once (``embed_table``); the embeddings
and their extraction time are cached per table. The ids are those of the paper's rows in
``scripts/paper_selection.yaml``, so a new model's rows sit next to the paper's.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

SEED = 1234
N_DIMS = 30

# model: the folder of the weights in <cache>/weights/; revision: the paper's checkpoint.
ENCODERS = {
    "tfidf": {"checkpoint": None, "reduction": "svd-30"},
    "qwen-3-8b-first30": {"model": "qwen-3-8b", "checkpoint": "Qwen/Qwen3-Embedding-8B",
                          "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "reduction": "first-30"},
    "llama-3.1-8b-first30": {"model": "llama-3.1-8b", "checkpoint": "meta-llama/Llama-3.1-8B",
                             "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b", "reduction": "first-30"},
    "minilm-l6-v2-pca30": {"model": "minilm-l6-v2", "checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
                           "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "reduction": "pca-30"},
}
NAMES = {"none": "", "tfidf": "Tf-Idf", "qwen-3-8b-first30": "Qwen-3-8B · first-30",
         "llama-3.1-8b-first30": "LLaMA-3.1-8B · first-30", "minilm-l6-v2-pca30": "MiniLM-L6-v2 · PCA-30"}


def cleaner():
    """The paper's first TableVectorizer: it parses the table (dates become numbers) and
    leaves the strings as they are, for the encoder that follows to scale every number."""
    from skrub import TableVectorizer

    return TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")


def tfidf_encoder(seed: int = SEED):
    """The paper's Tf-Idf encoder (prepare_tabvec): the cleaner, then skrub's TableVectorizer
    with a 30-component StringEncoder.

    Columns with fewer than 40 distinct values are one-hot encoded, and numbers, dates included,
    go through a SquashingScaler.
    """
    from skrub import SquashingScaler, StringEncoder, TableVectorizer

    return make_pipeline(cleaner(), TableVectorizer(high_cardinality=StringEncoder(n_components=N_DIMS, random_state=seed),
                                                    numeric=SquashingScaler()))


def string_columns(X: pd.DataFrame) -> list[str]:
    """The string columns as skrub sees them; columns it parses as dates are not strings."""
    return list(cleaner().fit(X).kind_to_columns_["high_cardinality"])


def embed_table(X: pd.DataFrame, encoder: str, table: str, cache_dir: Path, device: str = "cuda") -> tuple[pd.DataFrame, float]:
    """Embeddings of every distinct string of the table, and the seconds it took.

    Cached as ``<cache_dir>/embeddings/<encoder>/<table>.parquet`` with a ``name`` column and
    one column per kept dimension (30 for first-30, all for PCA), plus ``<table>.time``. Mirrors
    the paper's extraction script: every cell of the string columns as text, missing ones as
    "nan", distinct values in column order, sentence-transformers with the checkpoint's own
    modules and no prompt, no extra normalization, batch 32 (16 for the 8B models; halved while
    a batch of the longest texts does not fit in GPU memory).
    """
    folder = Path(cache_dir) / "embeddings" / encoder
    path, time_path = folder / f"{table}.parquet", folder / f"{table}.time"
    if path.exists():
        return pd.read_parquet(path), float(time_path.read_text())
    import torch
    from sentence_transformers import SentenceTransformer

    spec = ENCODERS[encoder]
    weights = Path(cache_dir) / "weights" / spec["model"]
    model = SentenceTransformer(str(weights) if weights.exists() else spec["checkpoint"], device=device)
    if "llama" in encoder:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    X = X.fillna(value=np.nan)
    columns = string_columns(X)
    batch_size = 16 if "8b" in encoder else 32
    while True:
        t0 = time.perf_counter()
        strings = pd.unique(pd.concat([X[c].astype(str) for c in columns], ignore_index=True))
        try:
            vectors = model.encode(np.asarray(strings, dtype=object), convert_to_numpy=True, batch_size=batch_size,
                                   normalize_embeddings=False)
            break
        except torch.OutOfMemoryError:  # the longest texts of a table may not fit: smaller batches, same embeddings
            if batch_size == 1:
                raise
            batch_size //= 2
            torch.cuda.empty_cache()
            print(f"{encoder} {table}: out of GPU memory, batch size {batch_size}", file=sys.stderr)
    if spec["reduction"] == "first-30":
        vectors = vectors[:, :N_DIMS]
    table_embeddings = pd.DataFrame(vectors, columns=[f"X{i}" for i in range(vectors.shape[1])])
    table_embeddings.insert(0, "name", strings)
    seconds = time.perf_counter() - t0
    folder.mkdir(parents=True, exist_ok=True)
    # The parquet file marks the table as done, so it is written last and renamed into place:
    # a reader never sees half a file. scripts/run_array.sh fills this cache in its own step,
    # before any model runs, so tasks never race to write the same table.
    time_path.write_text(f"{seconds:.4f}")
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    table_embeddings.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return table_embeddings, seconds


def _lookup(embeddings: pd.DataFrame):
    """The cached embedding of each cell of a one-column frame (NaN rows where none is cached)."""

    def lookup(column: pd.DataFrame) -> np.ndarray:
        df = pd.DataFrame(column)
        df.columns = ["name"]
        return np.array(df.merge(how="left", right=embeddings, on="name").drop(columns="name"))

    return lookup


class FirstDims(TransformerMixin, BaseEstimator):
    """One string column -> its first 30 embedding dimensions; missing cells stay missing."""

    def __init__(self, lookup=None, n_components: int = N_DIMS):
        self.lookup = lookup
        self.n_components = n_components

    def fit(self, column, y=None):
        return self

    def transform(self, column):
        name = column.columns[0]
        mask = column[name].isnull().to_numpy()
        out = np.zeros((len(column), self.n_components))
        out[~mask] = self.lookup(column)[~mask, : self.n_components]
        out[mask] = np.nan
        return pd.DataFrame(out).add_prefix(f"{name}_")


class PCADims(TransformerMixin, BaseEstimator):
    """One string column -> 30 PCA components of its embeddings, fitted on the training rows.

    The PCA (after a mean imputer, for strings with no cached embedding) is fitted on the
    non-missing cells only; missing cells stay missing. A column with fewer non-missing training
    cells than components keeps the first 30 dimensions instead.
    """

    def __init__(self, lookup=None, n_components: int = N_DIMS, random_state: int = SEED):
        self.lookup = lookup
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, column, y=None):
        self.fit_transform(column)
        return self

    def fit_transform(self, column, y=None):
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        self.pipeline_ = None
        if (~column[column.columns[0]].isnull().to_numpy()).sum() >= self.n_components:
            self.pipeline_ = Pipeline([("impute", SimpleImputer(copy=False)),
                                       ("pca", PCA(n_components=self.n_components, copy=False, random_state=self.random_state))])
        return self._encode(column, fit=True)

    def transform(self, column):
        return self._encode(column, fit=False)

    def _encode(self, column, fit: bool):
        name = column.columns[0]
        mask = column[name].isnull().to_numpy()
        vectors = self.lookup(column)[~mask]
        if self.pipeline_ is None:
            vectors = vectors[:, : self.n_components]
        else:  # the training rows come from fit_transform, as in the paper
            vectors = self.pipeline_.fit_transform(vectors) if fit else self.pipeline_.transform(vectors)
        out = np.zeros((len(column), self.n_components))
        out[~mask] = vectors
        out[mask] = np.nan
        return pd.DataFrame(out).add_prefix(f"{name}_")


def embedding_encoder(embeddings: pd.DataFrame, reduction: str):
    """TableVectorizer that replaces every string column by 30 numbers from its embeddings.

    Numbers, and the features skrub extracts from dates, go through a SquashingScaler, as in
    the paper (prepare_llm and prepare_llm_no_pca_mrl in its src/encoding.py).
    """
    from skrub import SquashingScaler, TableVectorizer

    per_column = (FirstDims if reduction == "first-30" else PCADims)(_lookup(embeddings))
    return TableVectorizer(cardinality_threshold=0, high_cardinality=per_column, numeric=SquashingScaler())


def warm_up(encoder: str) -> None:
    """Import what ``encode`` needs, so that imports are not timed."""
    import skrub  # noqa: F401
    import sklearn.decomposition  # noqa: F401


def encode(X_train, X_test, encoder: str, table: str, cache_dir=None, device: str = "cuda"):
    """Numeric train and test frames for one encoder, and the seconds the encoding costs.

    The encoder, the paper's cleaner first, is fitted on the training rows only. The seconds are
    the table's language-model extraction time, measured once and cached (0 for Tf-Idf), plus
    fitting and applying the encoder here; reading the cache is not counted.
    """
    if encoder == "tfidf":
        t0, seconds = time.perf_counter(), 0.0
        enc = tfidf_encoder()
    else:
        embeddings, seconds = embed_table(pd.concat([X_train, X_test]), encoder, table, Path(cache_dir), device)
        t0 = time.perf_counter()
        enc = make_pipeline(cleaner(), embedding_encoder(embeddings, ENCODERS[encoder]["reduction"]))
    X_train, X_test = enc.fit_transform(X_train), enc.transform(X_test)
    return X_train, X_test, seconds + time.perf_counter() - t0
