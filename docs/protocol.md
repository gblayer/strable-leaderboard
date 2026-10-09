# Evaluation protocol

Everything a result depends on. A change to anything on this page is a minor version bump (v0.1 to v0.2), and every pipeline is re-run.

## Data

- **Dataset:** [inria-soda/STRABLE-benchmark](https://huggingface.co/datasets/inria-soda/STRABLE-benchmark) on Hugging Face, pinned to revision `335c0a77595d4a9ddd86f46dc9c3b414f63a09a8` (2026-06-11). The pin lives in `strable_leaderboard/__init__.py`.
- **Tables:** 108. Each folder has `data.parquet` and `config.json` with `target_name` and `task`.

| Task | Tables | Metric |
|---|---|---|
| Regression | 76 | R² |
| Binary classification | 13 | ROC-AUC |
| Multiclass classification | 19 | ROC-AUC, one-vs-rest, macro average |

- **Targets:** classification labels are encoded to 0..K-1 in the sorted order of the original labels, as in the paper. Regression targets are used as stored: 61 of the 76 were transformed by the paper's skewness-minimizing step, and R² is computed on that scale.

## Cross-validation

The folds are exactly the paper's. The code is in `strable_leaderboard/splits.py`, and the precomputed folds for every table are in `splits/`.

**Outer loop**: evaluation.
- 3 folds, one repetition, seed 1234, on the row order of `data.parquet`.
- Regression: `RepeatedKFold(n_splits=3, n_repeats=1, random_state=1234)`.
- Classification: `RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1234)`.
- Each row is a test row exactly once. With the 75,000-row cap, training sets hold at most 50,000 rows.

**Inner loop**: validation, inside each outer training set.
- 8 folds, same classes and seed.
- Default configurations use it only for early stopping: the first inner fold is the validation set, and the model trains on the other 7/8. `StrableModel.validation_split(y)` returns that split.
- Tuned variants, when added, will search hyperparameters over the 8 inner folds.

**Files**: `splits/<table>.parquet` has one row per table row, in the order of `data.parquet`.

| Column | Meaning |
|---|---|
| `outer_fold` | The outer fold (0, 1 or 2) in which the row is a test row |
| `inner_fold_0`, `inner_fold_1`, `inner_fold_2` | Inside outer training set k, the inner fold (0 to 7) in which the row is a validation row. -1 when the row is in outer test fold k |

`splits/manifest.json` records the dataset revision, the seed, the scikit-learn version and, per table, the row count and the SHA-256 of `data.parquet`. `strable_leaderboard.splits.splits_from_table` rebuilds index arrays from a file.

## Scores

1. **Per fold:** R² on the test fold for regression. ROC-AUC from predicted probabilities for classification: the positive-class column for binary tasks, one-vs-rest macro average for multiclass tasks.
2. **Per table:** the mean over the 3 outer folds, raw: a negative R² stays negative, as in the paper.
3. **Per pipeline:** the mean of the table scores over the tables of the chosen view.

| View | Tables averaged |
|---|---|
| Both (default) | All 108, R² for regression and AUC for classification |
| Regression | The 76 regression tables |
| Classification | The 32 classification tables |

There is no per-table normalization, so a pipeline's score never changes when other pipelines join. A pipeline must cover every table and fold of a view to be ranked in it.

This is the only score on the board. Elo, improvability and normalized scores are relative to the field and change whenever a model joins, so they are not shown.

## Time

One time per run, the total, in seconds per 1,000 rows of the table: string embedding plus training plus prediction, as in the paper's Pareto plot.

It is measured by the runner's trusted parent process, never by the model's code (`strable_leaderboard/run.py`):

- The parent encodes the strings itself. Encoding time = the table's language-model extraction time, measured once per table and encoder when the cache is filled and added to every fold and every model that uses it (0 for Tf-Idf), plus the time to fit and apply the encoder on this fold. Reading the cache is not counted.
- The model runs in a child process in the sandbox. Two warm-up children run first: each starts the sandbox and Python, reads the inputs, imports the model, fits on 64 rows (at least one per class) and predicts 8, which covers imports, CUDA start-up and weight loading. The first one only warms the file caches; the wall-clock time of the second is the start-up overhead O.
- The timed child does the same start-up, fits on all training rows and predicts the test rows. Its wall-clock time is T.
- `total_s = encoding + max(T − O, 0.001)`. The difference is a measurement: for models that fit in well under a second, the noise of a process start dominates it.
- Per table: the mean over folds. Per pipeline: the median over tables.
- Fit and predict are not timed apart: a split would have to come from timestamps written by the model's own code. The board's fit and predict views therefore stay empty for these rows, as for rows imported from the paper.

## Pareto front

For the selected time, sort pipelines by time. A pipeline is Pareto-optimal when its score is strictly higher than that of every faster pipeline. The front is computed from the results, in the browser by `site/index.html` and in Python by `strable_leaderboard.metrics.pareto_front`; nobody places a row on it by hand. API models are shown but excluded from the computation, since their time includes the network. Open, closed and anonymous rows are all eligible.

The top three rows of the current order, Pareto, score or time, carry a red border in the table and a ring in the plot.

## Pipelines

**End-to-end models** read raw strings themselves, such as ConTextTab, TabSTAR or CatBoost with native categorical handling. They enter as they are.

**Numeric-only models**, such as TabPFN-2.5 or TabICLv2, are paired with three string encoders. Each string column becomes 30 numbers:

| Encoder | Checkpoint | Reduction to 30 dimensions |
|---|---|---|
| Tf-Idf | none, character 3-4 grams | Truncated SVD |
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | First 30 embedding dimensions |
| LLaMA-3.1-8B | `meta-llama/Llama-3.1-8B` | First 30 embedding dimensions |

- Three encoders cover the range: the cheapest and strongest baseline, a small sentence encoder, a large decoder. The paper's E5-small-v2 and Qwen-3-8B behave like their neighbours and are left out to keep a new model at 972 runs instead of 1,620.
- The reduction is fixed per encoder and never picked per pipeline from test scores. The paper finds that decoder embeddings lose accuracy under PCA and that encoder-only embeddings are indifferent, so every language model keeps its first 30 dimensions. Full embeddings are not an option: 384 or 4,096 numbers per string column exceed what the tabular models take.
- The Tf-Idf pipeline uses skrub's `TableVectorizer`, as in the paper: columns with fewer than 40 distinct values are one-hot encoded, dates are expanded, numbers go through a `SquashingScaler`. Language-model pipelines send every string column to the encoder.
- Each cell is embedded on its own, and missing strings stay missing.

## Configurations

Default configurations only: every model runs with the hyperparameters it ships with. Early stopping may use the protocol's validation split. Tuned and tuned-plus-ensembled variants are planned for later versions.

## Hardware

Each result records what it ran on. GPU rows: one GPU of the reference type, fixed at launch, with 32 CPU cores and 128 GB of RAM. CPU rows: the same node without the GPU. API rows: the vendor's servers. A model too large for the reference node runs on a larger one, and the row says so.

## Result records

One file per model, `<model>.csv`, one row per encoder, table and outer fold. The files stay private in the gitignored `results/` folder of the clone on the cluster; the board publishes only the aggregates in `site/results.json`.

`model, encoder, table, task, fold, score, fit_s, predict_s, total_s, n_train, n_test, hardware, package, checkpoint_revision, protocol_version, date`

- `encoder` is `none` for end-to-end models. `score` is the raw fold score.
- Every row fills `total_s` only; `fit_s` and `predict_s` stay empty. Runs by the runner have `hardware` the GPU model or `cpu`.
- Rows imported from the paper (`scripts/import_paper_results.py`) have `hardware: paper`, since the paper ran on mixed hardware. They may name encoders outside the three above.
- Rows are default configurations; a `variant` column is added when tuned runs come. `python -m strable_leaderboard.check_results` validates a file.

## Differences from the paper

- The paper times one total per fold in the model's process, with the embedding time measured once per table and added to every fold. The leaderboard also times one total, from outside the sandboxed model process, minus the measured start-up overhead.
- The paper evaluates about 445 pipelines. The leaderboard opens with a launch set, chosen around mid-November 2026, and grows.
- Still to decide: for multiclass tasks, the paper re-applies a softmax to `predict_proba` before computing AUC. The leaderboard plans to use the probabilities as returned.
