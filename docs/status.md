# Status, 8 October 2026

What exists, what was decided, what is left. Written for Gioia at the end of the first build phase (29 September to 4 October 2026).

## Where everything is

| | |
|---|---|
| Repository | https://github.com/gblayer/strable-leaderboard, private, branch `main` (8 commits) |
| Live page | https://gblayer.github.io/strable-leaderboard/ (GitHub Pages, public even though the repository is private, placeholder numbers until the launch) |
| Design preview, same page | https://claude.ai/artifact/VeySGPLA7yTSFYkoqwD4Cr |
| Local clone | `~/Documents/Documents - Work/strable-leaderboard` |
| Open branch | `add-extratrees`: the first real submission, waiting for its pull request at https://github.com/gblayer/strable-leaderboard/pull/new/add-extratrees |
| Tests | `pytest`: 15 pass |

## What was built

**The page** (`site/index.html`). Pantone palette (Mother of Pearl, Mystic Navy, Red Inferno, Pine Grove), Outfit as the stand-in for Balimo, IBM Plex Mono for numbers. Header: STRABLE, STRing-tABLE, /ˈstreɪ.bəl/, *stray-buhl*, the NeurIPS lede, version chip, Version history link, build date, Refresh button. Ticker. Controls: Task (Both, Regression · R², Classification · AUC), Sort (★ Pareto, Score, Time). A fixed-height scrolling table with every pipeline, tags for end-to-end, anonymous and closed rows, a star on the Pareto front, one red frame around the top three; the plot with the Pareto staircase from the x axis to the right edge, labels on the front, a "Better" arrow, rings around the top three; the version history with the citation line; a suggestion panel (Tally, or a GitHub issue); a six-step Demo in the navigation. No key, no filters, no footer, no hardware column, no API rows. The page reads `site/results.json` when it opens and on Refresh; without the file it shows placeholders and says so.

**The protocol** (`docs/protocol.md`, `strable_leaderboard/`). The paper's exact folds for all 108 tables, exported to `splits/` with a manifest (dataset revision pinned). Scores exactly as in the paper: raw R² and ROC-AUC per fold, then the median over all fold rows as in Figure 4a (changed 8 October from the mean over tables), no flooring, no normalization. One time, the total run time per 1,000 samples (median over fold rows), as in Figure 4a; no fit or predict views (8 October). Three encoders for numeric-only models: Tf-Idf (SVD to 30), all-MiniLM-L6-v2 and LLaMA-3.1-8B (first 30 embedding dimensions). Default configurations only; `variant` field reserved.

**The model contract** (`strable_leaderboard/model.py`, `models/_template/`, `models/ridge/`). `class Model(StrableModel)` with `fit`, `predict`, `predict_proba`, a `handles_strings` flag, a model card (`model.yaml`: name, version, tasks, variant, hardware, package pin, checkpoint repo + commit hash, licence, `trained_on_strable`) and pinned `requirements.txt`. A smoke test runs one fold of three small tables.

**Review automation** (`.github/workflows/`, `.claude/skills/`). Checks: the smoke test on every model pull request and a validation of `site/results.json`, on GitHub's machines without secrets. Claude review: two independent passes that post a short verdict (MERGE, FIX or REJECT) and a label; nothing merges by itself. Claude on issues labelled `claude`. Pages deployment on every push touching `site/`. Skills: `add-model` (for contributors' coding agents), `review-submission`, `run-model` (for Claude Code on the cluster).

**The runner** (`strable_leaderboard/run.py`, `scripts/`). One task = one model × encoder × table × fold, timed with an untimed warm-up, writing one row. `prepare.py` downloads the dataset, checkpoints and builds the environment; `make_tasks.py` lists what is missing; `run_array.sh` is the SLURM array, requeue-safe; `gather_results.py` assembles `results/<model>.csv`; `check_results.py` validates it; `build_site_data.py` writes `site/results.json` and validates it with `--check`; `import_paper_results.py` converts the paper's `result_comparison` CSVs (scores and total time, hardware "paper"). Language-model embeddings are computed once per table and cached, as in the paper. Ran end to end locally with Ridge + Tf-Idf.

**Storage.** Per-fold CSVs and `versions.yaml` live in `results/`, gitignored, so they stay on the cluster. GitHub holds only `site/results.json`, the aggregates the page shows (what TabArena also publishes).

**README and docs.** README with an SVG banner in the site's palette (lettering as outlines, so GitHub renders the typeface), badges, the page address, a star-and-cite line, a closing Citation section with the paper's BibTeX, `AGENTS.md`. Docs: plan, protocol, submitting (with the pull-request diagram and the fork explanation), automation, site, decisions (dated), research.

## Decisions, in short

Full list with dates in `docs/decisions.md`. The ones that shaped everything: one score, computed as in the paper; three encoders; default configurations; anyone can submit an open model; no API models; Claude reviews twice, the maintainer merges; results private, aggregates public; GitHub Pages, no paid domain; no cluster or institution named in the repository; the board opens with the paper's Figure 3 and 4a pipelines and grows from there.

## Done by Gioia on GitHub

Labels `claude`, `claude-approved`, `needs-maintainer`; Pages switched on (source: GitHub Actions); the About box; the `CLAUDE_CODE_OAUTH_TOKEN` secret, generated on the cluster.

## Next

1. Open the pull request for `add-extratrees` and watch the Checks and the two Claude reviews; merge if they say MERGE.
2. On the cluster: set `STRABLE_CACHE`, run `scripts/import_paper_results.py` on the paper's CSVs, add a line to `results/versions.yaml`, build `site/results.json`, open the results pull request: the first real board.
3. A `models/<name>/` folder for each learner of the launch set (ExtraTrees done; XGBoost, CatBoost, TabPFN-2.5, TabICLv2, ConTextTab, TabSTAR, TabM, RealMLP, Kumo Tabular), so they can be re-run with fit and predict timed apart.
4. Check the reference GPU with `nvidia-smi` on the node where the runs will happen, and write the type into the page's hover text.
5. Decide: supersede old model versions as TabArena now does, or keep one row per version; a "verified by the authors" tag; a LICENSE file for the repository (Apache-2.0 or MIT).
6. When the repository goes public: branch protection on `main`, a GoatCounter account if visits should be counted, a stars badge.
