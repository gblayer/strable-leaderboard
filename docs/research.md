# Research notes

Collected from 29 September to 1 October 2026 while planning the leaderboard.

## TabArena

Sources: [the Space](https://huggingface.co/spaces/TabArena/leaderboard) and [the repository](https://github.com/autogluon/tabarena), read at leaderboard version v0.1.9.3.

**Site**
- A Gradio app on a free CPU Space.
- All results are precomputed and committed to the Space as CSV files and self-contained HTML figures. There is one precomputed cell for every combination of entrant pool, task, dataset subset and imputation setting.

**Metrics**
- Elo: pairwise, default RandomForest pinned at 1000, 200-round bootstrap for the confidence intervals.
- Normalized score: best model = 1, median model = 0.
- Improvability: mean over datasets of the relative gap to the best model.
- Average rank and harmonic rank.
- Median train time and median predict time per 1,000 rows.

**Pareto front:** a 2-D skyline. Sort by time, and keep a point only if it beats every faster point. Drawn as a step line with an "Optimal" arrow. There are separate views for train time and predict time.

**Filters**
- Task: classification, regression, binary, multiclass.
- Dataset size and target imbalance.
- Imputed models, TabArena-Lite, commercial use.
- Entrant pools: open systems, systems using LLMs, closed APIs.

**Families:** foundation model, neural network, tree-based, baseline, system. **Variants:** default, tuned, tuned plus ensembled.

**Submissions**
- The authors run TabArena-Lite themselves and open a pull request with an AutoGluon model wrapper and a search space.
- Maintainers re-run the full benchmark on their own hardware. "TabArena is not a benchmarking service."
- The original authors sign off, which earns a "verified" check mark.
- Maintainers update the leaderboard in batches, by hand, a few days after a merge.

**Compute**
- The paper runs used CPU M6i.2xlarge and GPU L40S machines.
- Current runs use a GCP SLURM cluster with RTX PRO 6000 GPUs.
- Budget: 1 hour per configuration, 8 CPUs, 32 GB. An untimed warm-up runs before timing.
- A CatBoost re-run on different CPUs gave the same Elo but fit times 1.5 to 1.6 times higher. Runtime columns need fixed hardware.

**Protocol**
- 51 datasets, 3-fold CV repeated 3 or 10 times, 8-fold inner bagging.
- 200 random configurations per model.
- Missing results are imputed with a default RandomForest.

## The STRABLE paper and code

[Paper](https://arxiv.org/abs/2605.12292), [code](https://github.com/soda-inria/strable), [data](https://huggingface.co/datasets/inria-soda/STRABLE-benchmark).

**Corpus**
- 108 tables: 76 regression, 13 binary, 19 multiclass.
- 33 sources across 8 fields.
- The median table has 7.7K rows and 18 columns. The median string is 17 characters long.
- Tables are capped at 75,000 rows.
- 61 regression targets carry a skewness-minimizing transform.

**Splits**
- Outer: 3-fold, `RepeatedKFold` or `RepeatedStratifiedKFold` with `random_state=1234`, one repetition.
- Inner: 8-fold with the same settings.
- Classification labels are label-encoded before splitting.

**Metrics**
- R² for regression. ROC-AUC for classification: binary on the positive-class probability, multiclass one-vs-rest macro.
- Scores are averaged over folds.
- The Pareto plot uses medians over table-fold pairs of score and run time per 1,000 rows.

**Encoders**
- Tf-Idf: skrub `StringEncoder`, char_wb 3-4 grams, SVD to 30. Low-cardinality columns are one-hot encoded.
- TargetEncoder.
- 41 language models, embedded per cell and reduced to 30 dimensions per column by PCA, standard scaling then PCA, or the first 30 dimensions.
- Decoder embeddings lose accuracy under plain PCA:

| Encoder | PCA | First 30 dims |
|---|---|---|
| LLaMA-3.1-8B | 0.6245 | 0.6975 |
| Qwen-3-8B | 0.6385 | 0.6958 |

Mean over five learners, read from Figure 2.

**Pareto frontier, Table E.8**

| Pipeline | Score | Seconds per 1K rows |
|---|---|---|
| TargetEncoder + Ridge | 0.6593 | 0.0924 |
| TargetEncoder + ExtraTrees | 0.7290 | 0.1695 |
| Tf-Idf + ExtraTrees | 0.7407 | 0.9679 |
| E5-small-v2 + ExtraTrees | 0.7422 | 2.5152 |
| Tf-Idf + TabICLv2 | 0.7799 | 2.9988 |
| Tf-Idf + TabPFN-2.5 | 0.7891 | 6.6700 |

**Runtime in the paper**
- Mixed hardware: V100, A100 and A40 GPUs, and several AMD and Intel CPUs.
- 842 days of compute in total.
- LLM embedding time is measured once per table and added to every fold.

**Known issues in soda-inria/strable, as of commit 0d5720f**
- `src/encoding.py` line 90 has an extra indent, so the `elif encode_method == "tabstar"` branch is a SyntaxError. Every evaluate script imports this module.
- `estim_configs` has no entry for TabICL, TabM, RealMLP, TabSTAR, ConTextTab or Mambular.
- `data_list_wide` doesn't match the Hugging Face dataset: 9 names are old or renamed, and 6 tables are missing.

## Other leaderboards

- **Hugging Face leaderboard template, and the Open LLM Leaderboard:** a requests dataset, a backend that polls it and runs evaluations, and a results dataset that the frontend reads. The Open LLM Leaderboard re-ran every model on its own cluster and was retired in March 2025.
- **MTEB:**
  - Submitters run the evaluation themselves and open pull requests with result files on a results repository.
  - CI compares them with reference models, and a sync job pushes the results to a Hugging Face dataset.
  - The job that runs contributor code has read-only permissions.
- **[RamanBench](https://github.com/ml-lab-htw/RamanBench):** submissions go through a coding agent. "Clone RamanBench, then tell your coding agent: Add my model to RamanBench, test it, and run it across the benchmark." The workflow lives in `.claude/agents/model-agent.md`. The README also has an issues link and named contacts. This repository copies that pattern.
- **[Impermanent](https://impermanent.timecopilot.dev/):** one static HTML page on GitHub Pages behind a CDN, with Chart.js from jsDelivr. That is why it loads much faster than a Gradio Space.
- **Automated runs on your own compute:** the safe pattern is pull, not push. A worker inside the network fetches approved jobs and posts results back. GitHub's guidance is to avoid self-hosted runners on public repositories.
