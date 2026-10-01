"""String encoders for models that only take numbers.

The leaderboard pairs every numeric-only model with three encoders. Each string column is
reduced to 30 numbers:

| Encoder            | Checkpoint                              | Reduction to 30 dims    |
|--------------------|-----------------------------------------|-------------------------|
| Tf-Idf             | none (character 3-4 grams)              | truncated SVD           |
| all-MiniLM-L6-v2   | sentence-transformers/all-MiniLM-L6-v2  | first 30 embedding dims |
| LLaMA-3.1-8B       | meta-llama/Llama-3.1-8B                 | first 30 embedding dims |

Only Tf-Idf is implemented here, for the smoke test. The language-model encoders belong to
the runner, which computes each table's embeddings once and caches them for every model.
"""

from __future__ import annotations

ENCODERS = {
    "tfidf": {"checkpoint": None, "reduction": "svd-30"},
    "minilm-l6-v2": {"checkpoint": "sentence-transformers/all-MiniLM-L6-v2", "reduction": "first-30"},
    "llama-3.1-8b": {"checkpoint": "meta-llama/Llama-3.1-8B", "reduction": "first-30"},
}


def tfidf_encoder(seed: int = 1234):
    """The paper's Tf-Idf encoder: skrub's TableVectorizer with a 30-component StringEncoder.

    Columns with fewer than 40 distinct values are one-hot encoded, dates are expanded,
    and numbers are passed through a SquashingScaler, as in the paper.
    """
    from skrub import SquashingScaler, StringEncoder, TableVectorizer

    return TableVectorizer(
        high_cardinality=StringEncoder(n_components=30, random_state=seed),
        numeric=SquashingScaler(),
    )
