"""Load STRABLE tables at the pinned Hugging Face revision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import HF_DATASET, HF_REVISION

# config.json task names -> leaderboard task names
TASKS = {
    "regression": "regression",
    "b-classification": "binary",
    "m-classification": "multiclass",
}

MANIFEST = Path(__file__).resolve().parent.parent / "splits" / "manifest.json"


def list_tables() -> list[str]:
    """Names of the 108 tables, read from splits/manifest.json."""
    return sorted(json.loads(MANIFEST.read_text())["tables"])


def load_table(name: str, data_dir: str | Path | None = None, revision: str = HF_REVISION):
    """Return ``(X, y, task)`` for one table.

    With ``data_dir``, read ``<data_dir>/<name>/data.parquet`` and ``config.json`` from disk,
    for example a local ``snapshot_download`` of the dataset. Otherwise download both files
    from Hugging Face at the pinned revision. Rows keep the order of ``data.parquet``, which
    the cross-validation splits rely on. Classification targets are label-encoded to
    0..K-1 in the sorted order of the original labels, as in the paper.
    """
    if data_dir is not None:
        folder = Path(data_dir) / name
        table_path, config_path = folder / "data.parquet", folder / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        kwargs = dict(repo_id=HF_DATASET, repo_type="dataset", revision=revision)
        table_path = hf_hub_download(filename=f"{name}/data.parquet", **kwargs)
        config_path = hf_hub_download(filename=f"{name}/config.json", **kwargs)
    config = json.loads(Path(config_path).read_text())
    data = pd.read_parquet(table_path)
    target, task = config["target_name"], TASKS[config["task"]]
    y = data[target].to_numpy()
    if task != "regression":
        y = np.unique(y, return_inverse=True)[1]
    return data.drop(columns=target), y, task
