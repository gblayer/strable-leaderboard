# Decisions

Taken by Gioia Blayer while planning the leaderboard with Claude, 29 September to 1 October 2026.

## Product

| Date | Decision |
|---|---|
| 09-29 | A standalone leaderboard, with a structure inspired by TabArena |
| 09-29 | The MVP shows pipeline, score and runtime, with Pareto-optimal pipelines first. Users can sort by score or time |
| 09-30 | Versions follow v0.MINOR.PATCH: a new model is a patch, a protocol change is a minor version. No "seasons" |
| 09-30 | Default configurations only for the MVP |
| 09-30 | Anyone can submit at launch, from academia or industry, open or closed source |
| 09-30 | The launch set is chosen around mid-November 2026, not all ~445 paper pipelines |
| 10-01 | The repository starts from the paper's latest best results and keeps integrating new models |
| 10-01 | Claude reviews and merges submissions. The current workflows let Claude merge results and trusted pull requests itself. External submissions get Claude's approval label and a one-click merge, because of prompt-injection risk. Full auto-merge is available if accepted, see `automation/README.md` |
| 10-01 | A maintainer can hand an issue to Claude with the label `claude` |

## Scores and time

| Date | Decision |
|---|---|
| 09-30 | The default task view is "Both": R² and AUC averaged over all 108 tables |
| 10-01 | Per table, the mean over folds, then an R² below 0 counts as 0 |
| 10-01 | No per-table normalization. Elo and improvability are candidates for later |
| 10-01 | Two times per pipeline: fit (embedding plus training) and predict (embedding plus prediction), per 1,000 rows |
| 10-01 | Each row shows the hardware it ran on |

## Models

| Date | Decision |
|---|---|
| 09-30 | Models that read raw strings natively are always included as they are |
| 09-30 | Numeric-only models get five encoders: Tf-Idf, E5-small-v2, all-MiniLM-L6-v2, LLaMA-3.1-8B, Qwen-3-8B |
| 10-01 | Decoders, meaning LLaMA and Qwen, use their first 30 embedding dimensions. E5 and MiniLM use PCA to 30, and Tf-Idf uses SVD to 30 |
| 09-30 | API models are allowed with a tag. They stay off the Pareto front, their version is pinned, the vendor pays and agrees not to train on the data |
| 10-01 | Anonymous models are allowed. The submitter doesn't have to identify, the model runs on our hardware, and it appears under a codename such as "jumping-carrot" with an anonymous tag. It stays on the Pareto front, and the name is never revealed |

## Site

| Date | Decision |
|---|---|
| 09-30 | Pantone palette: Mother of Pearl, Mystic Navy, Red Inferno, Pine Grove. Balimo font |
| 09-30 | Headline "STRABLE", then "STRing-tABLE" and the pronunciation *stray-buhl* |
| 09-30 | "Version history" instead of "changelog", with the version at the top |
| 09-30 | A panel asks visitors which feature they would like to see |
| 10-01 | No waitlist page. The ticker stays |
| 10-01 | The site says STRABLE was accepted to the NeurIPS 2026 Evaluation and Datasets track. It doesn't say every pipeline runs on the same hardware |
| 10-01 | The Paper link goes to arXiv. "Submit a pipeline" goes to this repository, not to soda-inria/strable |
| 10-01 | Suggestions go through a free tool that emails Gioia, or through GitHub issues |
| 10-01 | The site is a static page on GitHub Pages, like the Impermanent leaderboard |

## Infrastructure

| Date | Decision |
|---|---|
| 09-30 | Runs happen on Margaret, on the parietal partition where Gioia has priority. Jean Zay only for models too large for Margaret |
| 10-01 | The repository lives under Gioia's GitHub account for now, not in an organization |
| 10-01 | The dataset licence audit is dropped: STRABLE was accepted to NeurIPS and ships a Croissant file |
