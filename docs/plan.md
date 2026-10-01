# Plan

## Goal

A public leaderboard at https://gblayer.github.io/strable-leaderboard/ that ranks pipelines on STRABLE's 108 tables with raw strings and updates itself when a model is added. The board shows one accuracy number per pipeline, its fit and predict time per 1,000 rows, and lists the Pareto-optimal pipelines first.

New models come out all the time, so the board does not list the paper's ~445 pipelines. It opens with a launch set chosen around mid-November 2026 and grows with each submission.

## How it works

1. **Submit.** A pull request adds `models/<name>/`, by hand or through a coding agent. Closed and anonymous models use a private channel. See [submitting.md](submitting.md).
2. **Check and review.** GitHub runs the smoke test. Claude reviews the diff twice and posts a recommendation. A maintainer merges. See [automation.md](automation.md).
3. **Run.** The runner finds merged pipelines without results and submits a job array on the cluster.
4. **Publish.** The runner opens a results pull request. Claude checks it, a maintainer merges, the site rebuilds and the patch version goes up.

## What the board shows

- **Score:** one number, the mean over the 108 tables of R² for regression and AUC for classification, R² floored at 0 per table. Views for regression or classification alone. See [protocol.md](protocol.md).
- **Time:** fit plus predict by default, or either alone: fit is embedding plus training, predict is embedding plus prediction, per 1,000 rows. The choice drives the plot and the Pareto order. The top three rows of any order are marked.
- **Rows:** modular pipelines, one of three string encoders plus a numeric learner, and end-to-end models that read strings themselves. Tags for end-to-end, anonymous and closed rows. Hovering a row shows whether it used the GPU. The table has a fixed height and scrolls, so every pipeline is always in it.
- Version history, a suggestion panel, a Refresh button next to the version chip, and a guided demo that shows where things are; the key under the controls says how to read them.

Not shown: Elo, improvability, normalized scores, families, filters, a hardware column. Each of them either changes when other models join or adds a column without adding knowledge.

## Hosting

| Part | Where |
|---|---|
| This repository | GitHub, gblayer/strable-leaderboard |
| Site | GitHub Pages, https://gblayer.github.io/strable-leaderboard/. No domain to pay for |
| Paper code | soda-inria/strable, frozen with the paper |
| Data | Hugging Face, inria-soda/STRABLE-benchmark, pinned revision |
| Runs | The maintainers' cluster: one GPU type for GPU rows, the same nodes without GPU for CPU rows |
| Suggestions | A free Tally form that emails the maintainer, plus GitHub issues |

## Launch, v0.1.0

1. Pick the launch set from the paper: the baselines Ridge, ExtraTrees, XGBoost, CatBoost, the foundation models TabPFN-2.5 and TabICLv2 with the three encoders, the end-to-end models ConTextTab and TabSTAR, and the models released since, such as Kumo Tabular. Decided around mid-November 2026.
2. Run those pipelines once on the reference node, so every row has comparable times, and check that the scores match the paper.
3. Publish as v0.1.0. From then on every merged results pull request adds one patch version: v0.1.1, v0.1.2, and so on. Nothing before the launch carries a version.

A submission never triggers a re-run of the others: each pipeline's rows are stored once in `results/`, and the board recomputes the ranking and the Pareto front from all stored rows. Embeddings are cached per encoder, table and fold, so a new numeric-only model costs only its own 972 fit-and-predict runs.

## Later

- Tuned and tuned-plus-ensembled variants, as extra rows. The result records already carry a `variant` field.
- A per-table view.
- More string encoders, if asked for.

## Open questions

- The GPU type of the reference node.
- Whether the cluster offers a container runtime, a service account and cron on the login node.
- When the repository goes public.
- The Balimo font licence.
- Multiclass AUC: keep the paper's extra softmax, or use probabilities as returned.
