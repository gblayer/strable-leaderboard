# Cross-validation folds

One file per table: `<table>.parquet`, with one row per row of that table's `data.parquet`, in the same order. The folds reproduce the paper's protocol exactly. See the [paper](https://arxiv.org/abs/2605.12292).

| Column | Meaning |
|---|---|
| `outer_fold` | Outer fold (0, 1 or 2) in which the row is a test row |
| `inner_fold_0`, `inner_fold_1`, `inner_fold_2` | Inside outer training set k, the inner fold (0 to 7) in which the row is a validation row. -1 when the row is in outer test fold k |

`manifest.json` records the dataset revision, the seed, the scikit-learn version and, per table, the row count, the test rows per outer fold, and the SHA-256 of the `data.parquet` the folds were computed on.

Use them in Python:

```python
import pandas as pd
from strable_leaderboard.splits import splits_from_table

folds = pd.read_parquet("splits/yelp_business.parquet")
train_idx, test_idx, inner = splits_from_table(folds, outer=0)
fit_idx, val_idx = inner[0]  # the early-stopping split, relative to train_idx
```

Or from any language: `outer_fold == k` gives the test rows of fold k.

Regenerate them with:

```
python scripts/export_splits.py --data-dir /path/to/STRABLE-benchmark
```
