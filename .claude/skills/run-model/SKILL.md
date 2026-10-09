---
name: run-model
description: Run a merged model on the cluster and publish its results on the STRABLE leaderboard, by hand or by checking on the dispatcher. Prepares the shared environment and cache, submits the smoke, embedding and model arrays, gathers and checks the rows, rebuilds site/results.json and opens the pull request. Use on the cluster when asked to run, benchmark or publish a model, or why the dispatcher stopped.
---

# Run a model and publish its results

You are in the clone of strable-leaderboard on the cluster's login node, on `main`. Read `docs/automation.md`, section 4, first. `cache/` holds the shared environment (`cache/env`), the dataset, the checkpoints, the embeddings, the sandbox image and the dispatcher's state; `results/` holds the per-fold CSVs and `results/versions.yaml`; `private/runner.env` holds the cluster's partition flags. All three are gitignored and never leave the cluster. Always run Python as `PYTHONNOUSERSITE=1 cache/env/bin/python`.

## Normally: the dispatcher

`scripts/dispatch.py` runs from cron every 15 minutes and does everything below. To see where a model is: `cat cache/dispatch/<name>.json` (step, job, reason) and `tail cache/dispatch/cron.log`. A failed model stays failed until `rm cache/dispatch/<name>.json`; the next pass retries from the start. One pass by hand: `PYTHONNOUSERSITE=1 cache/env/bin/python scripts/dispatch.py`.

## By hand

1. `git pull`, then check that `models/<name>/` is merged and `results/<name>.csv` is absent or incomplete.
2. `python scripts/prepare.py models/<name>` (internet: the login node). It stops if the model's pins differ from the shared environment.
3. Smoke test on the reference hardware: `echo "smoke models/<name>" > smoke.txt && bash scripts/run_array.sh smoke.txt`; the verdict is in `cache/dispatch/smoke/<name>.rc` and `.log`.
4. Numeric-only models: `python -m strable_leaderboard.embed --list --cache-dir cache > embed.txt`; if not empty, `bash scripts/run_array.sh embed.txt` and wait for it before step 5.
5. `python scripts/make_tasks.py models/<name> > tasks.txt` (972 for a numeric-only model, 324 for an end-to-end one), then `bash scripts/run_array.sh tasks.txt`. Watch with `squeue -u $USER`; logs in `logs/`. Resubmit the same file after a preemption: finished tasks are skipped.
6. `python scripts/gather_results.py runs results`, then `python -m strable_leaderboard.check_results results/<name>.csv`. Fix or re-run anything it flags.
7. One line at the top of `results/versions.yaml` with the next patch version, today's date and what changed, e.g. `- {version: v0.1.3, date: 2026-11-20, note: Added TabPFN 2.5}`. Then, in a worktree of `origin/main` on a branch `results/<name>`: `python scripts/build_site_data.py --results <clone>/results`, commit `site/results.json`, push, open the pull request titled `Results: <note> (<version>)`. Checks, the reviews and the gate take it from there.
8. After the merge: `git tag <version> <merge commit> && git push origin <version>`.

Never commit `results/`, `private/`, `cache/`, `runs/` or the paper's files, and never write the cluster's name into the repository.
