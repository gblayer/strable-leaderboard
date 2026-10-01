---
name: run-model
description: Run a merged model on the cluster and publish its results on the STRABLE leaderboard. Prepares the cache, submits the SLURM job array, gathers and checks the rows, rebuilds site/results.json and opens the pull request. Use on the cluster when asked to run, benchmark or publish a model.
---

# Run a model and publish its results

You are in a clone of strable-leaderboard on the cluster's login node. `STRABLE_CACHE` is the shared cache (dataset, checkpoints, embeddings, environments) and `STRABLE_RESULTS` the private results folder, a clone of the private repository. Read `docs/automation.md`, section 3, first.

1. `git pull`, then check that `models/<name>/` is merged and `$STRABLE_RESULTS/<name>.csv` is absent or incomplete.
2. `python scripts/prepare.py models/<name> --cache-dir $STRABLE_CACHE` (needs internet: the login node).
3. `python scripts/make_tasks.py models/<name> > tasks.txt`; report how many tasks are missing (972 for a numeric-only model, 324 for an end-to-end one).
4. `sbatch --array=0-$(( $(wc -l < tasks.txt) - 1 ))%16 scripts/run_array.sh tasks.txt`. Watch with `squeue -u $USER`; look at `logs/` for failures. Resubmit the same line after a preemption: finished tasks are skipped.
5. When the array is done: `python scripts/gather_results.py runs $STRABLE_RESULTS`, then `python -m strable_leaderboard.check_results $STRABLE_RESULTS/<name>.csv`. Fix or re-run anything it flags; coverage must be complete for the views the model supports.
6. Commit and push the private results folder (`cd $STRABLE_RESULTS && git add . && git commit -m "<name>" && git push`).
7. Add one line at the top of `$STRABLE_RESULTS/versions.yaml` with the next patch version, today's date and what changed, then `python scripts/build_site_data.py --results $STRABLE_RESULTS`.
8. In the public clone: `git checkout -b results/<name>`, commit `site/results.json`, push, open the pull request titled `Results: <display name>`, body = the check output and the task count. Claude reviews it; the maintainer merges; GitHub Pages redeploys and the page shows the new row.

Never commit a results CSV or the paper's files to the public repository.
