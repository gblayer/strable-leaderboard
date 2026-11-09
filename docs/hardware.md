# Hardware

Each result row records the hardware it ran on, in its `hardware` column; `site/results.json` gives each pipeline's `hw`, the GPUs of its rows. The page's Compute dropdown groups the pipelines by it:

| Entry | Pipelines | What the rows record |
|---|---|---|
| A40, A100, H100, … | run by the board, or imported with their GPU | the GPU's name, e.g. `NVIDIA A40` |
| A100 + A40 | rows spread over two GPU types | both names |
| Shared cluster | the STRABLE paper's pipelines | `paper`: no GPU per row. As the paper's Appendix D.1 says, its jobs were scheduled across NVIDIA V100, A100 and A40 GPUs with AMD EPYC and Intel Xeon CPUs (32 cores and one GPU per job), not pinned to one GPU type |
| API | models that run on their provider's servers. The board is a third party to these runs and cannot verify them: it shows the time and hardware the provider reports. ChakraTab: YHat Labs states an NVIDIA RTX PRO 6000 GPU and 12 vCPUs | `api` |

For a pipeline with a language-model encoder, the GPU shown is the one the learner ran on. The encoder's extraction, measured once per table, may have run on another GPU (almost always an A100 or an A40).

## Does the GPU type change the results?

It changes the run time, not the score. The test below ran two board pipelines with the board's own runner on three GPU types, on the same tasks.

- Pipelines: TabPFN-3.5 + Tf-Idf and Kumo Tabular + Tf-Idf.
- Tasks: 20 tables (13 regression, 7 classification, from the smallest to the largest), outer fold 0.
- GPUs: NVIDIA A40 (48 GB), NVIDIA A100-PCIE-40GB and NVIDIA H100 NVL (94 GB). The V100s of the paper's runs were not tested.
- Score: R² for regression, ROC-AUC for classification. Difference: the other GPU − A40, per table.
- Time: total run time per 1,000 samples (encoding, training, prediction). Ratio: the other GPU ÷ A40, per table; below 1 means faster than on the A40.
- Test: the paired Wilcoxon signed-rank test on the 20 differences (on the log of the times). A p-value above 0.05 means no difference was detected.

| Pipeline | GPU | Median score difference | Largest \|difference\| | p (score) | Median time ratio | p (time) |
|---|---|---|---|---|---|---|
| TabPFN-3.5 + Tf-Idf | A100 | +0.00006 | 0.0018 | 0.33 | 0.74 | 0.000004 |
| TabPFN-3.5 + Tf-Idf | H100 | +0.00005 | 0.0018 | 0.33 | 0.42 | 0.014 |
| Kumo Tabular + Tf-Idf | A100 | 0.00000 | 0.0010 | 0.42 | 0.61 | 0.000002 |
| Kumo Tabular + Tf-Idf | H100 | 0.00000 | 0.0010 | 0.50 | 0.32 | 0.006 |

The H100 rows ran with a later version of the shared environment, in which two data-loading packages (`huggingface-hub`, `pyarrow`) were updated; the models' packages were the same.

The same pipelines, run twice on the A40 (the board's rows and the test's rows), as a baseline:

| Pipeline | Median score difference | Largest \|difference\| | Median time ratio | p (time) |
|---|---|---|---|---|
| TabPFN-3.5 + Tf-Idf | 0 (19 of 20 tables identical) | 0.0000 | 0.96 | 0.06 |
| Kumo Tabular + Tf-Idf | 0.00000 | 0.0001 | 0.99 | 0.06 |

How to read it:

- Scores: no difference was detected (p from 0.33 to 0.50). The A100 and H100 scores differ from the A40 ones more than two runs on the same GPU (at most 0.0018 against 0.0001), but the differences are small and go in no consistent direction.
- Times: at the median table, the A100 took 26% (TabPFN-3.5) and 39% (Kumo Tabular) less time than the A40, and the H100 58% and 68% less. The A100 was faster on 39 of the 40 tasks, the H100 on 32; on the other 8 the H100 was slower, up to 14 times on one table.
- So the hardware moves a pipeline along the time axis of the plot, not along the score axis. The Compute dropdown keeps pipelines that ran on the same kind of machine, so that their times compare fairly; it does not change their scores.
- 20 tables and one fold: a small effect cannot be excluded.
