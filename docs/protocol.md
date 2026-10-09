# Evaluation protocol

Everything a result depends on. A change to anything on this page is a minor version bump (v0.1 to v0.2), and every pipeline is re-run.

## Data

- **Dataset:** [inria-soda/STRABLE-benchmark](https://huggingface.co/datasets/inria-soda/STRABLE-benchmark) on Hugging Face, pinned to revision `335c0a77595d4a9ddd86f46dc9c3b414f63a09a8` (2026-06-11). The pin lives in `strable_leaderboard/__init__.py`.
- **Tables:** 108. Each folder has `data.parquet` and `config.json` with `target_name` and `task`.

| Task | Tables | Metric |
|---|---|---|
| Regression | 76 | R² |
| Binary classification | 13 | ROC-AUC |
| Multiclass classification | 19 | ROC-AUC of the softmaxed probabilities, one-vs-rest, macro average |

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
- Default configurations use it only for early stopping: the first inner fold is the validation set, and the model trains on the other 7/8, as the paper's `run_inference` (`src/inference.py`) does for XGBoost and CatBoost without a search. `StrableModel.validation_split(y)` returns that split.
- Tuned variants search hyperparameters over the 8 inner folds (see Configurations).
- Learners with their own internal validation in the paper (RealMLP and TabM through pytabkit, TabSTAR, Mambular) did not use these inner folds.

**Files**: `splits/<table>.parquet` has one row per table row, in the order of `data.parquet`.

| Column | Meaning |
|---|---|
| `outer_fold` | The outer fold (0, 1 or 2) in which the row is a test row |
| `inner_fold_0`, `inner_fold_1`, `inner_fold_2` | Inside outer training set k, the inner fold (0 to 7) in which the row is a validation row. -1 when the row is in outer test fold k |

Checked on 2026-10-09 for all 108 tables: the outer and inner folds in `splits/` equal, row by row, those the paper's code (`set_split_cv` and the inner `cv` of `script_evaluate.py`) generates on the paper's processed data, and each `data.parquet` at the pinned revision is byte-identical to the paper's.

`splits/manifest.json` records the dataset revision, the seed, the scikit-learn version and, per table, the row count and the SHA-256 of `data.parquet`. `strable_leaderboard.splits.splits_from_table` rebuilds index arrays from a file.

## Scores

1. **Per fold:** R² on the test fold for regression. ROC-AUC from predicted probabilities for classification: the positive-class column for binary tasks; for multiclass tasks, a softmax over each row of `predict_proba`, then the one-vs-rest macro average. The softmax is the paper's (`reshape_pred_output` in its `src/utils_evaluation.py`), kept so that scores stay comparable with the paper's.
2. **Per pipeline:** the median of the fold scores (R² for regression, ROC-AUC for classification) over all fold rows of the view, raw, exactly as in the paper's Figure 4a, "Median score (R² & AUC)": every (table, outer fold) pair is one row, and a negative R² stays negative.

| View | Fold rows |
|---|---|
| Both (default) | All 108 tables × 3 folds = 324, R² and AUC pooled |
| Regression | The 76 regression tables × 3 = 228 |
| Classification | The 32 classification tables × 3 = 96 |

The Regression and Classification views are the same metric on a subset of rows. There is no per-table normalization, so a pipeline's score never changes when other pipelines join. A pipeline is shown in every view where it has rows. When tables or folds are missing, the row carries the tag **INCOMPLETE**; the page shows no counts, and `site/results.json` keeps the fold rows present and expected.

This is the only score on the board. Elo, improvability and normalized scores are relative to the field and change whenever a model joins, so they are not shown.

## Time

One time, the total run time per 1,000 samples, as in the paper's Figure 4a:

- **Per fold:** encoding plus training plus prediction seconds, divided by the table's total rows, times 1,000. For tuned variants it also includes the hyperparameter search.
- **Per pipeline:** the median over the fold rows of the view, like the score.
- The plot shows it on a log x axis, "Total run time per 1K samples (s), log scale". There are no separate fit or predict views.

Runs are timed by the runner's trusted parent process, never by the model's code (`strable_leaderboard/run.py`):

- The parent encodes the strings itself. Language-model embeddings are computed once per table and encoder, for every distinct string of the table, then cached and reused by every fold and every model. Encoding time = the table's extraction time, measured once when the cache is filled and added to every fold of every pipeline that uses it (0 for Tf-Idf), plus the time to fit the encoder on this fold's training rows and apply it to the training and test rows. Reading the cache is not counted.
- The model runs in a child process in the sandbox. Two warm-up children run first: each starts the sandbox and Python, reads the inputs, imports the model, fits on 64 rows (at least one per class) and predicts 8, which covers imports, CUDA start-up and weight loading. The first one only warms the file caches; the wall-clock time of the second is the start-up overhead O.
- The timed child does the same start-up, fits on all training rows and predicts the test rows. Its wall-clock time is T.
- `total_s = encoding + max(T − O, 0.001)`. The difference is a measurement: for models that fit in well under a second, the noise of a process start dominates it.
- Fit and predict are not timed apart: a split would have to come from timestamps written by the model's own code. `fit_s` and `predict_s` stay empty.

## Pareto front

Sort pipelines by the total run time per 1,000 samples, the only time on the board. A pipeline is Pareto-optimal when its score is strictly higher than that of every faster pipeline. The front is computed from the results, in the browser by `site/index.html` and in Python by `strable_leaderboard.metrics.pareto_front`; nobody places a row on it by hand. Every pipeline is eligible, tuned and incomplete ones included, as in the paper's Figure 4a.

The top three rows of the current order, Pareto, score or time, carry a red border in the table. The plot does not depend on the order.

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

**Other encoders.** Pipelines can also use the paper's other encoders, under ids and names that say what was done (`scripts/paper_selection.yaml`): the language models and FastText went through a PCA to 30 components fitted on the training rows, the paper's default, so they appear as, for example, `minilm-l6-v2-pca30`, "MiniLM-L6-v2 · PCA-30". `-ss-pca30` adds a StandardScaler before the PCA, `-first30` keeps the first 30 dimensions without PCA. TargetEncoder and Tarte keep their names. Tarte is skrub's `TableVectorizer` with string columns passed through, then tarte_ai's `TARTE_TablePreprocessor`; the learner receives TARTE's 768-dimensional row embeddings (paper Appendix D.1 and Table D.2). Only the paper's Tf-Idf is the leaderboard's `tfidf`; a PCA row never carries the id of a first-30 encoder.

## Configurations

Default configurations: every model runs with the hyperparameters it ships with. Early stopping may use the protocol's validation split. Each result row has a `variant`: `default`, or `tuned` for a hyperparameter search over the inner folds. Tuned rows (XGBoost, ExtraTrees and CatBoost so far) carry a **tuned** tag. Tuning is a random search of 100 configurations (99 sampled plus the default, seed 1234) over the 8 inner folds of each outer training set, in the search space of the paper's Table D.1, "Hyperparameter search space". For CatBoost (`run_gbdt_param_search` in the paper's `src/param_search.py`), each configuration trains on 7/8 with early stopping on the held-out inner fold; the best configuration by mean inner score is then refitted on 7/8, with early stopping on the inner fold where it scored highest. Tuned-plus-ensembled variants are planned for later versions.

## Hardware

Each result records what it ran on. GPU rows: one GPU of the reference type, fixed at launch, with 32 CPU cores and 128 GB of RAM. CPU rows: the same node without the GPU. A model too large for the reference node runs on a larger one, and the row says so.

## Result records

One file per model, `<model>.csv`, one row per encoder, variant, table and outer fold, in the gitignored `results/` folder of the clone on the cluster. `scripts/build_site_data.py` publishes from them the aggregates in `site/results.json` and `site/results.parquet`, and every fold row with a score in `site/results_per_fold.parquet` and `site/results_per_fold.csv`: `id, encoder, learner, variant, table, task, fold, score, total_s, n_rows, time_per_1k_s, leakage` (`leakage` is `possible` when the review found that the model's pre-training data may include web tables, see `docs/automation.md`, section 6). Its `--check` recomputes the medians of `results.json` from the per-fold file.

`model, encoder, variant, table, task, fold, score, fit_s, predict_s, total_s, n_train, n_test, hardware, package, checkpoint_revision, protocol_version, date`

- `encoder` is `none` for end-to-end models. `score` is the raw fold score.
- Runs by the runner fill `total_s` only (see Time); `fit_s` and `predict_s` stay empty. Rows that fill `fit_s` and `predict_s` instead count their sum. `hardware` is the GPU model or `cpu`.
- `variant` is `default` or `tuned`; the runner copies it from the model card.
- A row without a score is an error for `check_results`, and only rows with a score count towards a view's fold rows.
- `python -m strable_leaderboard.check_results` validates a file and prints its coverage per view.
