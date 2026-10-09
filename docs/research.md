# Research notes

Collected from 29 September to 1 October 2026 while planning the leaderboard.

## TabArena

[Space](https://huggingface.co/spaces/TabArena/leaderboard) and [repository](https://github.com/autogluon/tabarena), read at leaderboard version v0.1.9.3.

- **Site:** a Gradio app on a free CPU Space. Every view is precomputed and committed as CSV files and self-contained HTML figures.
- **Metrics:** Elo (pairwise, RandomForest pinned at 1000, 200-round bootstrap), normalized score (best = 1, median = 0), improvability, average and harmonic rank, median train and predict time per 1,000 rows.
- **Pareto front:** a 2-D skyline drawn as a step line with an "Optimal" arrow, one view per time.
- **Filters:** task, dataset size, imbalance, imputed models, TabArena-Lite, commercial use. Pools: open, using LLMs, closed APIs. Families: foundation model, neural network, tree-based, baseline, system. Variants: default, tuned, tuned plus ensembled.
- **Submissions:** authors run TabArena-Lite and open a pull request with an AutoGluon wrapper and a search space; maintainers re-run the full benchmark on their own hardware and update the board in batches. "TabArena is not a benchmarking service."
- **Compute:** 1 hour per configuration, 8 CPUs, 32 GB, untimed warm-up. A CatBoost re-run on different CPUs gave the same Elo but fit times 1.5 to 1.6 times higher: time columns need fixed hardware.
- **Protocol:** 51 datasets, 3-fold CV repeated 3 or 10 times, 8-fold inner bagging, 200 random configurations per model, missing results imputed with a default RandomForest.

## The STRABLE paper and code

[Paper](https://arxiv.org/abs/2605.12292), [code](https://github.com/soda-inria/strable), [data](https://huggingface.co/datasets/inria-soda/STRABLE-benchmark).

- **Corpus:** 108 tables (76 regression, 13 binary, 19 multiclass) from 33 sources in 8 fields. Median table 7.7K rows and 18 columns, median string 17 characters. Tables capped at 75,000 rows. 61 regression targets carry a skewness-minimizing transform.
- **Splits:** outer 3-fold, `RepeatedKFold` or `RepeatedStratifiedKFold`, `random_state=1234`, one repetition. Inner 8-fold with the same settings. Labels are label-encoded before splitting.
- **Metrics:** R² and ROC-AUC (binary on the positive-class probability, multiclass one-vs-rest macro), averaged over folds. The Pareto plot uses medians over table-fold pairs.
- **Encoders:** Tf-Idf through skrub's `StringEncoder` (char_wb 3-4 grams, SVD to 30, low-cardinality columns one-hot), TargetEncoder, and 41 language models embedded per cell and reduced to 30 dimensions per column by PCA, scaling plus PCA, or the first 30 dimensions. Decoders lose accuracy under plain PCA:

| Encoder | PCA | First 30 dims |
|---|---|---|
| LLaMA-3.1-8B | 0.6245 | 0.6975 |
| Qwen-3-8B | 0.6385 | 0.6958 |

Mean over five learners, read from Figure 2.

- **Pareto frontier, Table E.10:** TargetEncoder + Ridge 0.6593 at 0.09 s/1K rows; TargetEncoder + ExtraTrees 0.7290 at 0.17; Tf-Idf + ExtraTrees 0.7407 at 0.97; E5-small-v2 + ExtraTrees 0.7422 at 2.5; Tf-Idf + TabICLv2 0.7799 at 3.0; Tf-Idf + TabPFN-2.5 0.7891 at 6.7.
- **Runtime:** mixed hardware (V100, A100, A40 GPUs, several CPUs), 842 compute-days in total. LLM embedding time is measured once per table and added to every fold.
- **Known issues in the paper code at commit 0d5720f:** `src/encoding.py` line 90 has an extra indent, so the `elif encode_method == "tabstar"` branch is a SyntaxError; `estim_configs` lacks TabICL, TabM, RealMLP, TabSTAR, ConTextTab and Mambular; `data_list_wide` has 9 old names and misses 6 tables of the Hugging Face dataset.

## Other leaderboards

- **Open LLM Leaderboard / HF template:** a requests dataset, a backend that polls it and runs evaluations, a results dataset the frontend reads. Retired in March 2025.
- **MTEB:** submitters run the evaluation and open pull requests with result files; CI compares with reference models; the job that runs contributor code has read-only permissions.
- **[RamanBench](https://github.com/ml-lab-htw/RamanBench):** submissions go through a coding agent. "Clone RamanBench, then tell your coding agent: Add my model to RamanBench, test it, and run it across the benchmark." The workflow lives in `.claude/agents/model-agent.md`. This repository copies the pattern.
- **[Impermanent](https://impermanent.timecopilot.dev/):** one static page on GitHub Pages behind a CDN, Chart.js from jsDelivr. Much faster than a Gradio Space.
- **Automated runs on your own compute:** pull, not push. A worker inside the network fetches approved jobs and posts results back. GitHub advises against self-hosted runners on public repositories.
