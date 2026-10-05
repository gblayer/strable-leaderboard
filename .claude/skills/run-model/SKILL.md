---
name: run-model
description: Run a merged model on the cluster and publish its results on the STRABLE leaderboard. Prepares the cache, submits the SLURM job array, gathers and checks the rows, rebuilds site/results.json and opens the pull request. Use on the cluster when asked to run, benchmark or publish a model.
---

# Run a model and publish its results

You are in the clone of strable-leaderboard on the cluster's login node. `STRABLE_CACHE` is the shared cache (dataset, checkpoints, embeddings, environments). `results/` holds the per-fold CSVs and `results/versions.yaml`; it is gitignored and never leaves the cluster. Read `docs/automation.md`, section 3, first.

1. `git pull`, then check that `models/<name>/` is merged and `results/<name>.csv` is absent or incomplete.
2. `python scripts/prepare.py models/<name> --cache-dir $STRABLE_CACHE` (needs internet: the login node).
3. `python scripts/make_tasks.py models/<name> > tasks.txt`; report how many tasks are missing (972 for a numeric-only model, 324 for an end-to-end one).
4. `sbatch --array=0-$(( $(wc -l < tasks.txt) - 1 ))%16 scripts/run_array.sh tasks.txt`. Watch with `squeue -u $USER`; look at `logs/` for failures. Resubmit the same line after a preemption: finished tasks are skipped.
5. When the array is done: `python scripts/gather_results.py runs results`, then `python -m strable_leaderboard.check_results results/<name>.csv`. Fix or re-run anything it flags; coverage must be complete for the views the model supports.
6. Add one line at the top of `results/versions.yaml` with the next patch version, today's date and what changed, for example `- {version: v0.1.3, date: 2026-11-20, note: Added TabPFN-2.5}`. Then `python scripts/build_site_data.py --results results`.
7. `git checkout -b results/<name>`, commit `site/results.json`, push, and open the pull request titled `Results: <display name>` with the check output and the task count in the body. Claude reviews it; the maintainer merges; GitHub Pages redeploys and the page shows the new row.
8. After the merge, on `main`: `git tag v0.1.3 && git push origin v0.1.3`, so the version a paper cites points at an exact commit.

Never commit `results/`, `private/`, `runs/` or the paper's files.
