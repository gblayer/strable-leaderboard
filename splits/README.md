# Cross-validation folds

The paper's folds, one file per table: `<table>.parquet`. Row i of the file describes row i of the table's `data.parquet` on Hugging Face. The board's folds are used for everything. See the [paper](https://arxiv.org/abs/2605.12292).

| Column | Meaning |
|---|---|
| `outer_fold` | Which of the 3 test folds the row belongs to: 0, 1 or 2. Every row is a test row exactly once. |
| `inner_fold_0`, `inner_fold_1`, `inner_fold_2` | When outer fold k is the test set, the other rows are split into 8 validation groups (the paper uses them for tuning and early stopping). `inner_fold_k` is the row's group, 0 to 7, or -1 when the row is in test fold k. |

`manifest.json` records the dataset revision (the Hugging Face commit of the data), the seed, the scikit-learn version the folds were computed with and, per table, the row count, the number of test rows in each outer fold, and the SHA-256 of `data.parquet`, to detect a change in the data.

In Python:

```python
import pandas as pd
from strable_leaderboard.splits import splits_from_table

folds = pd.read_parquet("splits/yelp_business.parquet")
train_idx, test_idx, inner = splits_from_table(folds, outer=0)
fit_idx, val_idx = inner[0]  # the early-stopping split, relative to train_idx
```
