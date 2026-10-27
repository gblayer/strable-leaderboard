# Hardware

Each result row records the hardware it ran on, in its `hardware` column; `site/results.json` gives each pipeline's `hw`, the GPUs of its rows. The page's Compute dropdown groups the pipelines by it:

| Entry | Pipelines | What the rows record |
|---|---|---|
| A40, A100, H100, … | run by the board, or imported with their GPU | the GPU's name, e.g. `NVIDIA A40` |
| A100 + A40 | rows spread over two GPU types | both names |
| Shared cluster | the STRABLE paper's pipelines | `paper`: no GPU per row. As the paper's Appendix D.1 says, its jobs were scheduled across NVIDIA V100, A100 and A40 GPUs with AMD EPYC and Intel Xeon CPUs (32 cores and one GPU per job), not pinned to one GPU type |
| API | models that run on their provider's servers | `api` |

For a pipeline with a language-model encoder, the GPU shown is the one the learner ran on. The encoder's extraction, measured once per table, may have run on another GPU (almost always an A100 or an A40).

## Does the GPU type change the results?

It changes the run time, not the score. The test below ran two board pipelines with the board's own runner on two GPU types, on the same tasks.

- Pipelines: TabPFN-3.5 + Tf-Idf and Kumo Tabular + Tf-Idf.
- Tasks: 20 tables (13 regression, 7 classification, from the smallest to the largest), outer fold 0.
- GPUs: NVIDIA A40 (48 GB) and NVIDIA A100-PCIE-40GB. H100: pending. The V100s of the paper's runs were not tested.
- Score: R² for regression, ROC-AUC for classification. Difference: A100 − A40, per table.
- Time: total run time per 1,000 samples (encoding, training, prediction). Ratio: A100 ÷ A40, per table; below 1 means faster on the A100.
- Test: the paired Wilcoxon signed-rank test on the 20 differences (on the log of the times). A p-value above 0.05 means no difference was detected.

| Pipeline | Median score difference | Largest \|difference\| | p (score) | Median time ratio | p (time) |
|---|---|---|---|---|---|
| TabPFN-3.5 + Tf-Idf | +0.00006 | 0.0018 | 0.33 | 0.74 | 0.000004 |
| Kumo Tabular + Tf-Idf | 0.00000 | 0.0010 | 0.42 | 0.61 | 0.000002 |

The same pipelines, run twice on the A40 (the board's rows and the test's rows), as a baseline:

| Pipeline | Median score difference | Largest \|difference\| | Median time ratio | p (time) |
|---|---|---|---|---|
| TabPFN-3.5 + Tf-Idf | 0 (19 of 20 tables identical) | 0.0000 | 0.96 | 0.06 |
| Kumo Tabular + Tf-Idf | 0.00000 | 0.0001 | 0.99 | 0.06 |

How to read it:

- Scores: no difference was detected (p = 0.33 and 0.42). The A100 and A40 scores differ more than two runs on the same GPU (at most 0.0018 against 0.0001), but the differences are small and go in no consistent direction.
- Times: at the median table, the A100 took 26% (TabPFN-3.5) and 39% (Kumo Tabular) less time than the A40. It was faster on 39 of the 40 tasks.
- So the hardware moves a pipeline along the time axis of the plot, not along the score axis. The Compute dropdown keeps pipelines that ran on the same kind of machine, so that their times compare fairly; it does not change their scores.
- 20 tables and one fold: a small effect cannot be excluded.
