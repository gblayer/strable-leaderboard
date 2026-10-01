# Decisions

Taken by Gioia Blayer while planning the leaderboard with Claude, 29 September to 1 October 2026.

## Product

| Date | Decision |
|---|---|
| 09-29 | A standalone leaderboard, inspired by TabArena, in its own repository |
| 09-29 | The MVP shows pipeline, score and times, Pareto-optimal pipelines first. Users can sort by score or time |
| 09-30 | Versions follow v0.MINOR.PATCH: a new model is a patch, a protocol change a minor version. No "seasons" |
| 09-30 | Default configurations only. Tuned and tuned-plus-ensembled variants later, as extra rows |
| 09-30 | Anyone can submit, from academia or industry, open or closed source |
| 09-30 | The launch set is chosen around mid-November 2026, not all ~445 paper pipelines |
| 10-01 | Nothing merges automatically. Claude reviews every pull request twice and posts a short recommendation; a maintainer merges |
| 10-01 | A maintainer can hand an issue to Claude with the label `claude` |
| 10-01 | The repository names no cluster or institution; hardware is described as GPU, CPU and memory |

## Scores and time

| Date | Decision |
|---|---|
| 09-30 | The default task view is "Both": R² and AUC averaged over all 108 tables |
| 10-01 | Per table, the mean over folds, then an R² below 0 counts as 0 |
| 10-01 | One score, no normalization. Elo and improvability are not shown |
| 10-01 | Two times per pipeline: fit (embedding plus training) and predict (embedding plus prediction), per 1,000 rows. The default view is their sum |
| 10-01 | No hardware column: hovering a row shows whether it used the GPU; API rows carry a tag |
| 10-01 | The top three rows of the current order carry a red border |

## Models

| Date | Decision |
|---|---|
| 09-30 | Models that read raw strings natively are always included as they are |
| 09-30 | Numeric-only models get five encoders: Tf-Idf, E5-small-v2, all-MiniLM-L6-v2, LLaMA-3.1-8B, Qwen-3-8B |
| 10-01 | Three encoders instead of five: Tf-Idf (SVD to 30), all-MiniLM-L6-v2 and LLaMA-3.1-8B (first 30 embedding dimensions). E5-small-v2 and Qwen-3-8B dropped to keep a new model at 972 runs |
| 09-30 | API models are allowed, kept off the Pareto front, version pinned, the vendor pays and agrees not to train on the data |
| 10-01 | Anonymous models are allowed: run on our hardware, codename like "jumping-carrot" (an action verb and a fruit or vegetable) with an anonymous tag, on the Pareto front, name never revealed |
| 10-01 | The model card has no family field |
| 10-01 | Access is open, closed or api. Anonymous is closed with the name hidden |
| 10-01 | The card carries `variant: default`; tuned and tuned-ensembled rows come later from their own folders |

## Site

| Date | Decision |
|---|---|
| 09-30 | Pantone palette: Mother of Pearl, Mystic Navy, Red Inferno, Pine Grove. Balimo font |
| 09-30 | Headline "STRABLE", then "STRing-tABLE" and the pronunciation *stray-buhl* |
| 09-30 | "Version history" instead of "changelog", with the version at the top |
| 09-30 | A panel asks visitors which feature they would like to see |
| 10-01 | No waitlist page. The ticker stays |
| 10-01 | The lede says STRABLE was accepted to the NeurIPS 2026 Evaluation and Datasets track and that the board does not list all ~445 paper pipelines. Nothing else |
| 10-01 | The Paper link goes to arXiv. "Submit a pipeline" goes to this repository |
| 10-01 | Suggestions go through a free Tally form that emails Gioia, or through GitHub issues |
| 10-01 | A static page on GitHub Pages, like the Impermanent leaderboard |
| 10-01 | No family or access filters. Tags on the rows instead. A Refresh button under the table |
| 10-01 | The hero art uses invented company names only |
| 10-01 | The README carries the site's palette and type through an SVG banner; the leaderboard URL goes in the README and the repository's About box |

## Infrastructure

| Date | Decision |
|---|---|
| 09-30 | Runs happen on the maintainers' cluster, on nodes where the runner has priority. Larger models run on a larger node and the row says so |
| 10-01 | The repository lives under Gioia's GitHub account, not in an organization |
| 10-01 | The dataset licence audit is dropped: STRABLE was accepted to NeurIPS and ships a Croissant file. Every table and column stays as published |
