# Plan

## Goal

A public leaderboard at strable.ai that ranks pipelines on STRABLE's 108 tables with raw strings and updates itself when a model is added. One score, two times, Pareto-optimal pipelines first.

New models come out all the time, so the board does not list the paper's ~445 pipelines. It opens with a launch set chosen around mid-November 2026 and grows with each submission.

## How it works

1. **Submit.** A pull request adds `models/<name>/`, by hand or through a coding agent. Closed, anonymous and API models use a private channel. See [submitting.md](submitting.md).
2. **Check and review.** GitHub runs the smoke test. Claude reviews the diff twice and posts a recommendation. A maintainer merges. See [automation.md](automation.md).
3. **Run.** The runner finds merged pipelines without results and submits a job array on the cluster.
4. **Publish.** The runner opens a results pull request. Claude checks it, a maintainer merges, the site rebuilds and the patch version goes up.

## What the board shows

- **Score:** one number, the mean over the 108 tables of R² for regression and AUC for classification, R² floored at 0 per table. Views for regression or classification alone. See [protocol.md](protocol.md).
- **Time:** fit, meaning embedding plus training, and predict, meaning embedding plus prediction, per 1,000 rows. A switch picks which one drives the plot and the Pareto order.
- **Rows:** modular pipelines, one of five string encoders plus a numeric learner, and end-to-end models that read strings themselves. Tags for end-to-end, anonymous, closed and API rows. Each row says what it ran on.
- Version history, a suggestion panel, a refresh button and a guided demo.

Not shown: Elo, improvability, normalized scores, families, filters. Each of them either changes when other models join or adds a column without adding knowledge.

## Hosting

| Part | Where |
|---|---|
| This repository | GitHub, gblayer/strable-leaderboard |
| Paper code | soda-inria/strable, frozen with the paper |
| Site | GitHub Pages with the domain strable.ai |
| Data | Hugging Face, inria-soda/STRABLE-benchmark, pinned revision |
| Runs | The maintainers' cluster: one GPU type for GPU rows, the same nodes without GPU for CPU rows |
| Suggestions | A free Tally form that emails the maintainer, plus GitHub issues |

## Launch, v0.1.0

1. Import the paper's per-fold results for the launch set.
2. Re-run those pipelines once on the reference node, so every row has comparable times.
3. Check that the re-run scores match the paper.
4. Publish.

## Later

- Tuned and tuned-plus-ensembled variants, as extra rows. The result records already carry a `variant` field.
- A per-table view.
- More string encoders, if asked for.

## Open questions

- The GPU type of the reference node.
- Whether the cluster offers a container runtime, a service account and cron on the login node.
- Where API models run, since compute nodes have no internet.
- When the repository goes public.
- Registering strable.ai, and the Balimo font licence.
- Multiclass AUC: keep the paper's extra softmax, or use probabilities as returned.
