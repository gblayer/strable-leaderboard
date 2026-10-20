---
name: run-model
description: Run a merged model on the cluster and publish its results on the STRABLE leaderboard, by hand or by checking on the dispatcher. Prepares the shared environment and cache, submits the smoke, embedding and model arrays, gathers and checks the rows, rebuilds site/results.json and opens the pull request. Use on the cluster when asked to run, benchmark or publish a model, or why the dispatcher stopped.
---

# Run a model and publish its results

You are in the clone of strable-leaderboard on the cluster's login node. Read `docs/automation.md`, sections 4 and 5, first; the dispatcher's steps are in the docstring of `scripts/dispatch.py`, the jobs in `scripts/run_array.sh`, the sandbox in `strable_leaderboard/run.py` and `scripts/sandbox.sh`. `cache/` holds the shared environment (`cache/env`), the dataset, the checkpoints, the embeddings, the sandbox image and the dispatcher's state; `results/` holds the per-fold CSVs and `results/versions.yaml`; `private/runner.env` holds the cluster's partition flags. All three are gitignored and never leave the cluster. Always run Python as `PYTHONNOUSERSITE=1 cache/env/bin/python`.

## Normally: the dispatcher

`scripts/dispatch.py` runs from cron every 15 minutes and does everything below. To see where a model is: `cat cache/dispatch/<name>.json` (step, job, reason) and `tail cache/dispatch/cron.log`. A failed model stays failed until `rm cache/dispatch/<name>.json`; the next pass retries from the start. One pass by hand: `PYTHONNOUSERSITE=1 cache/env/bin/python scripts/dispatch.py`.

## By hand

Never run from the clone's working copy, which may hold unmerged edits: use a checkout of `origin/main`, as the dispatcher does.

1. `git fetch origin && git worktree add --detach cache/manual origin/main && cd cache/manual`, then `export STRABLE_CACHE=<clone>/cache STRABLE_RUNS=<clone>/runs STRABLE_LOGS=<clone>/cache/logs STRABLE_RUNNER_ENV=<clone>/private/runner.env`. Check that `models/<name>/` is there and `<clone>/results/<name>.csv` is absent or incomplete.
2. `python scripts/prepare.py models/<name>` (internet: the login node). It stops if the model's pins differ from the shared environment or the checkpoint is above 20 GB.
3. Smoke test on the reference hardware: `echo "smoke models/<name>" > smoke.txt && bash scripts/run_array.sh smoke.txt`; the verdict is in `$STRABLE_CACHE/dispatch/smoke/<name>.rc` and `.log`.
4. Numeric-only models: `python -m strable_leaderboard.embed --list --cache-dir $STRABLE_CACHE > embed.txt`; if not empty, `bash scripts/run_array.sh embed.txt` and wait for it before step 5.
5. `STRABLE_RESULTS=<clone>/results python scripts/make_tasks.py models/<name> > tasks.txt` (1296 for a numeric-only model, 324 for an end-to-end one), then `bash scripts/run_array.sh tasks.txt`. Watch with `squeue -u $USER`; logs in `$STRABLE_LOGS`. Resubmit the same file after a preemption: finished tasks are skipped.
6. `python scripts/gather_results.py <clone>/runs <clone>/results <name>`, then `python -m strable_leaderboard.check_results <clone>/results/<name>.csv`. Fix or re-run anything it flags.
7. One line at the top of `<clone>/results/versions.yaml` with the next patch version, today's date and what changed, e.g. `- {version: v0.1.3, date: 2026-11-20, note: Added TabPFN 2.5}`. Then, in a worktree of `origin/main` on a branch `results/<name>`: write `site/flags.json` (`{}` if absent), with `"<name>": {"overlap": true, "evidence": "https://github.com/gblayer/strable-leaderboard/pull/<number>"}` if the maintainer gave the model's pull request the label `pretraining-overlap`; `python scripts/build_site_data.py --results <clone>/results --models <the models of the published site/results.json, the paper's files, <name>>`; commit the files `RESULTS_FILES` of `strable_leaderboard/__init__.py` names, push, open the pull request titled `Results: <note> (<version>)`. Checks, the review and the gate take it from there.
8. After the merge: `git tag <version> <merge commit> && git push origin <version>`, and `git worktree remove cache/manual`.

## An API model

A model with `hardware: api` (docs/automation.md, section 4, API models) needs its base URL and key on the cluster, never in git: `install -d -m 700 private/api_keys`, then `private/api_keys/<name>.env` with the two lines `STRABLE_API_URL=https://api.yhatlabs.com` (the provider's base URL) and `STRABLE_API_KEY=...`, `chmod 600`. Write the key without leaving it in the shell history, as for the token below. The order: audit `python scripts/audit_api.py models/test-<name> --data-dir $STRABLE_CACHE/data` (key file `private/api_keys/test-<name>.env`; any FAIL, ask the provider before going on), then the test run of `models/test-<name>/` (3 tasks), then the real folders. The dispatcher starts each only once its pull request has the label `run-api`. By hand, `export STRABLE_API_KEYS=<clone>/private/api_keys` before step 3.

## A new package in the shared environment

A merged pull request that adds a line to `env/requirements.in`: run `bash scripts/lock_env.sh` (the paper's full environment, `env/paper-freeze.txt`, as constraints, so no existing version moves; a pin changed in `env/requirements.in` overrides it), commit the new `env/requirements.txt`, then `python scripts/prepare.py` rebuilds `cache/env`. Until then `prepare.py` stops on the model. Another version of a package already there is the maintainer's decision alone: it moves the whole board.

## Setup, once

1. Secret `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` (uses the Claude subscription), or `ANTHROPIC_API_KEY`.
2. Labels `claude`, `claude-approved`, `needs-maintainer`, `pretraining-overlap`, `run-api`.
3. Each workflow sets its own token permissions in its `permissions` block; nothing to change in the settings.
4. On the cluster, in the clone: `python scripts/prepare.py` (shared env, data, encoders, sandbox image), `private/runner.env` with the partition flags, and the cron line of `scripts/dispatch.py`.
5. The dispatcher's token, on the cluster. On github.com: Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token. Resource owner `gblayer`; Repository access: *Only select repositories*, `gblayer/strable-leaderboard` only; Repository permissions: Contents read and write, Pull requests read and write, Issues read and write, Metadata read-only (set by default); nothing else, in particular not Workflows; expiration at most one year (put a reminder in the calendar). Store it, readable by you only, without leaving it in the shell history:
   ```
   install -d -m 700 ~/.config/strable && (umask 077; read -rs -p "token: " t && printf '%s\n' "$t" > ~/.config/strable/github_token)
   ```
   `STRABLE_GH_TOKEN_FILE` points elsewhere if needed. The dispatcher refuses to start when the file is missing or readable by anyone else (mode other than 0600). It reaches git through a credential helper that reads the file, set in the dispatcher's environment only, which also switches off every other helper (`credential.helper=store` included).
6. Branch protection on `main`, requiring the "Checks" workflow. Do not require approving reviews: the gate merges with `GITHUB_TOKEN`, which cannot approve.

Never commit `results/`, `private/`, `cache/`, `runs/` or the paper's files, and never write the cluster's name into the repository.
