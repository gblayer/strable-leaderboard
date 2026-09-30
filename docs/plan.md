# Plan

## Goal

A public leaderboard at strable.ai that ranks pipelines on STRABLE's 108 tables with raw strings, and updates itself when a model is added. The MVP stays simple: pipeline, score, fit time and predict time, with Pareto-optimal pipelines first.

The leaderboard doesn't list all ~445 pipelines from the paper, because new models come out all the time. It starts from the paper's latest best results for a launch set, chosen around mid-November 2026, and grows with each new model.

## How it works

1. **Submit.** A contributor opens a pull request with one folder, `models/<name>/`, or asks their coding agent to do it. Anonymous and closed-source models go through a private channel instead. See [submitting.md](submitting.md).
2. **Check.** GitHub runs the smoke test on three small tables, and Claude reviews the diff against the checklist. Claude merges pull requests from trusted accounts. External submissions get Claude's `claude-approved` label and a one-click merge. See [automation.md](automation.md).
3. **Run.** A runner on Margaret's login node finds merged pipelines without results, prepares environments and checkpoints, and submits a job array on parietal. See [cluster.md](cluster.md).
4. **Publish.** The runner opens a results pull request. Claude checks and merges it, the site rebuilds on GitHub Pages, and the version goes up by one patch.

Rankings and the Pareto front are computed in the browser from `results.json`, so they refresh with every results merge.

## What the board shows

- **Score:** by default the mean over all 108 tables of R² for regression and AUC for classification, with R² floored at 0 per table. Regression-only and classification-only views too. See [protocol.md](protocol.md).
- **Time:** fit time, meaning embedding plus training, and predict time, meaning embedding plus prediction, both per 1,000 rows. A switch picks which time drives the plot and the Pareto order.
- **Pipelines:** modular pipelines pair one of five string encoders with a numeric-only learner. End-to-end models read raw strings themselves.
- **Labels:** family and access filters, and tags for end-to-end, anonymous and API entries. Each row says where it ran.
- **Other parts:** version history, a suggestion panel, and a guided demo.

## Where things are hosted

| Part | Where |
|---|---|
| This repository | GitHub, gblayer/strable-leaderboard. Private for now, moved to an organization later if wanted |
| Code to reproduce the paper | Stays in soda-inria/strable, frozen with the paper |
| Site | GitHub Pages with the custom domain strable.ai |
| Data | Hugging Face, inria-soda/STRABLE-benchmark, pinned revision |
| Runs | Margaret, parietal partition. Jean Zay only for models too large for Margaret |
| Suggestions | A free Tally form emailing gioia.blayer@inria.fr, plus GitHub issues |

## Seeding v0.1 from the paper

1. Import the paper's per-fold results for the launch set.
2. Re-run the same pipelines once on the reference node, so fit and predict times are comparable across all entries.
3. Check that the re-run scores match the paper. This confirms the leaderboard code reproduces it.
4. Publish v0.1.0.

## Metrics roadmap

| Metric | Status |
|---|---|
| Mean score, R² and AUC | v0.1 headline |
| Elo | Planned: pairwise wins across tables, with confidence intervals. Relative, so it changes when models join |
| Improvability | Planned: how far a model is from the best model on each table, averaged. Also relative |

## Other ideas from TabArena, not in v0.1

- **Variants:** default, tuned, tuned plus ensembled. v0.1 is default only. The result records keep a variant field so tuned runs can be added later.
- **Per-table results:** a view to browse one table.
- **More string encoders**, if the community asks.

## Open questions

- Which exact GPU the reference nodes have, A100 or A40.
- Whether Margaret offers Apptainer, a service account, and cron on the login node.
- Where API models run, since compute nodes have no internet.
- The private submission channel for anonymous models.
- When the repository goes public, which also enables free branch protection.
- Registering strable.ai, and the Balimo font licence.
- For multiclass AUC: keep the paper's extra softmax or use probabilities as returned.
