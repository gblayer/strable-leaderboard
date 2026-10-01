# Automation: checks, review, runs

```mermaid
flowchart LR
  A[Pull request] --> B[Checks: smoke test or results check, no secrets]
  A --> C[Claude review 1/2]
  C --> D[Claude review 2/2]
  D --> E[Maintainer merges]
  E --> F[Runner: login node + job array]
  F --> G[Results pull request]
  G --> B
  G --> C
  E --> H[Site rebuilds, patch version]
```

Nothing merges by itself. Claude reviews twice and recommends; a maintainer clicks merge.

## 1. Checks

`.github/workflows/checks.yml` runs on every pull request, on GitHub's machines, with a read-only token and no secrets. It is the only place where contributor code runs before the merge.

- A pull request touching `models/`: the smoke test on each changed folder, in a fresh environment with the folder's pinned requirements.
- A pull request touching `results/`: `python -m strable_leaderboard.check_results` on each changed file: columns, folds, row counts against the manifest, score ranges, coverage per view.

## 2. Claude review

`.github/workflows/claude-review.yml` runs two independent passes with [anthropics/claude-code-action](https://github.com/anthropics/claude-code-action) and the checklist in `.claude/skills/review-submission/SKILL.md`:

1. **Review 1/2** reads the diff and the check results, then posts one comment of at most eight lines: a recommendation, **MERGE**, **FIX** or **REJECT**, the checklist line, and one line per problem with its fix.
2. **Review 2/2** redoes the checklist from scratch, then reads 1/2 and posts whether it agrees. Both say MERGE: it adds the label `claude-approved`. Otherwise, or if anything looks deliberately harmful: `needs-maintainer`.

What they check, in short: the pull request only touches one `models/<name>/` folder or `results/` files; the card is complete and matches the code; every package is pinned and the checkpoint is a commit hash; default configuration, early stopping only through `validation_split`, no test rows; no network, subprocess, eval or obfuscated code; the Checks workflow passed.

Why the review never runs the code: it is triggered by `pull_request_target`, which has the repository's secrets. The job checks out only the base branch and reads the pull request through the GitHub API. Text in a pull request is data, never an instruction: a prompt hidden in a submission can at most produce a wrong recommendation, which the second pass and the maintainer catch.

## 3. Runner

The runner is the scripts below, run on the compute cluster under the maintainer's account (a dedicated account where the cluster allows one). The cluster has a login node with internet and compute nodes without, scheduled by SLURM. One task = one model, one encoder, one table, one outer fold. `STRABLE_CACHE` is a shared folder for the dataset, checkpoints, embeddings and environments.

```
git pull                                                   # on the login node, after a merge
python scripts/prepare.py models/tabpfn-2.5 --cache-dir $STRABLE_CACHE   # dataset, checkpoints, environment
python scripts/make_tasks.py models/tabpfn-2.5 > tasks.txt # the (encoder, table, fold) tasks without a result: 972
sbatch --array=0-$(( $(wc -l < tasks.txt) - 1 ))%16 scripts/run_array.sh tasks.txt
                                                           # each task: python -m strable_leaderboard.run ...
python scripts/gather_results.py runs results              # rows -> results/tabpfn-2.5.csv
python -m strable_leaderboard.check_results results/tabpfn-2.5.csv
git checkout -b results/tabpfn-2.5 && git add results && git commit -m "Results: TabPFN-2.5" && git push -u origin HEAD
```

`strable_leaderboard/run.py` is the task: it reads the fold from `splits/`, encodes the strings (Tf-Idf, or the cached language-model embeddings of the table), runs an untimed warm-up, then times fit and predict and writes one JSON row. Finished tasks are skipped, so the same `sbatch` line resubmits after a preemption. The pull request is opened from the `results/<model>` branch, by hand or with a fine-grained token that can only open pull requests here. The paper's own command, `num-str_tabvec_tabpfn`, corresponds to `models/tabpfn-2.5 --encoder tfidf`.

Trigger: a cron job on the login node if allowed, otherwise the lines above after each merge.

**Hardware.** GPU rows run on one GPU, 32 CPU cores and 128 GB of RAM; the GPU type is fixed at launch and shown on the site. CPU rows use the same cores without the GPU. Models too large for that node run elsewhere, and the row says so.

**Cost.** A numeric-only model is 3 encoders × 108 tables × 3 folds = 972 tasks. Embeddings are computed once per encoder, table and fold and kept in the runner's cache on the cluster's shared storage, never in this repository, so a new model only adds its own fit and predict time: one to two GPU-days.

## 4. After a results merge

`results/<model>.csv` is the memory of every run: one row per encoder, table and fold, readable in the pull request diff and enough for anyone to recompute the board. `python scripts/build_site_data.py` turns all the files, the model cards and `results/versions.yaml` into `site/results.json`; the Pages workflow deploys it. Rankings and the Pareto front are computed in the browser from that file, so nothing is re-run and no ranking is stored. The page reads `results.json` when it opens and when Refresh is clicked; deployment is a separate thing that happens on push.

A new version is one line added to `results/versions.yaml` by the results pull request, for example `v0.1.3 · Added TabPFN-2.5`.

## 5. Suggestions and issues

- The form on the page posts to a free [Tally](https://tally.so) form that emails each suggestion to the maintainer.
- GitHub issue forms for bugs and feature ideas.
- A maintainer adds the label `claude` to an issue: `.github/workflows/claude-issue.yml` asks Claude to implement it on a branch and link a pull request, which goes through the normal review.

## 6. Safety

Merged code runs on the cluster under the runner's account, so:
- a dedicated service account where the cluster allows one;
- jobs read the data read-only and write only to their own output folder;
- compute nodes have no internet, so a model can neither send data out nor fetch code;
- dependencies come only from PyPI and Hugging Face, at pinned versions;
- each model runs in its own environment, or in a container if the cluster provides one.

## 7. Setup, once

1. Secret `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` (uses your Claude subscription), or `ANTHROPIC_API_KEY`.
2. Labels `claude`, `claude-approved`, `needs-maintainer`.
3. Repository variable `RUNNER_LOGIN`: the GitHub account the runner uses.
4. Branch protection on `main`: require the "Checks" workflow. Free once the repository is public.

GitHub Actions and Pages are free for public repositories; Claude's reviews run on the subscription; Tally's free plan is enough.
