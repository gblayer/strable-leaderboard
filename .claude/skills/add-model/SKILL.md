---
name: add-model
description: Add a model to the STRABLE leaderboard. Creates models/<name>/ from the template in a fork, wraps the model in the StrableModel contract, fills EVERY model card field from the model's package, docs and paper, pins dependencies to the shared environment, runs the smoke test and opens a pull request. Use when someone asks to add, submit, integrate or benchmark a model on the STRABLE leaderboard.
---

# Add a model to the STRABLE leaderboard

Goal: one pull request from the user's fork adding `models/<name>/`, passing the smoke test and `.claude/skills/review-submission/SKILL.md`, so that it can merge automatically. Read `docs/submitting.md`, `strable_leaderboard/model.py` and `env/requirements.txt` first.

## 0. Fork

The user cannot push to `gblayer/strable-leaderboard`. If the clone's `origin` is that repository, fork it (`gh repo fork gblayer/strable-leaderboard --remote`, or ask the user to click *Fork* on GitHub and add the fork as a remote), then work on a branch `add-<name>`. Explain this in one sentence if the user does not know forks: the fork is their own copy on GitHub, the pull request goes from it to the original.

## 1. Facts: find every one yourself

Fill the card without asking the user, from the model's sources, in this order: the package's metadata on PyPI (`pip index versions <pkg>`, `pip show`, the project page), its README and documentation, the Hugging Face model card and its commit history, the paper. Ask only for what none of them gives, and say where you looked.

| Card field | Where it comes from |
|---|---|
| `name`, `version` | The model's own name and release. One folder per model version; the version lives here |
| `handles_strings` | Does `fit` accept raw string columns? Check the docs or try it. Yes: `true` (end-to-end). No: `false` |
| `tasks` | Regression, binary, multiclass: what the package supports |
| `variant` | Always `default` |
| `access` | `open`. Closed, anonymous or API-only models stop here: point to `docs/submitting.md`, "Open, closed, anonymous" |
| `hardware` | `gpu` if it needs or is meant for a GPU, else `cpu` |
| `package` | `name==version`, the version in `env/requirements.txt` if the package is there |
| `checkpoint` | Hugging Face `repo` and the full 40-character commit hash of the weights (`huggingface_hub.HfApi().model_info(repo).sha`, or the repository's commit list), never `main`. `null` if there are no weights |
| `licence` | Code and weights licences, from the repository and the model card |
| `trained_on_strable` | STRABLE tables in its pre-training or tuning data, from the paper's data section; `[]` if none. Names from `splits/manifest.json` |
| `code`, `paper` | Repository and paper URLs (optional, but fill them when they exist) |

Never leave a template value (`My Model`, `my-model==1.0.0`, `0123456789abcdef`, ...): the smoke test fails on it.

## 2. Folder

`cp -r models/_template models/<name>`: lowercase letters, digits, dots and hyphens, one folder per model version. Never one of the names reserved for the paper's learners: `ridge`, `xgboost`, `extratrees`, `catboost`, `tabpfn-2.5`, `tabicl-v2`, `realmlp`, `tabm`, `mambular`, `contexttab`, `tabstar`, `tarte`; the gate refuses them.

## 3. `model.py`

`class Model(StrableModel)`:
- Set `handles_strings`.
- `fit(X, y)`: the default configuration; pass `self.seed`, `self.device`, `self.n_jobs`; for classification set `self.classes_ = np.unique(y)`.
- **No folds or search of your own** (`models/ridge` keeps RidgeCV's alpha search: the paper's baseline, the only exception): never `train_test_split`, `KFold`, `cross_val_score`, `GridSearchCV` or a package option that holds out its own validation set. Early stopping only through `fit_idx, val_idx = self.validation_split(y)`, the paper's first inner fold. If the package splits internally by default, turn that off or pass the protocol's split.
- **No seed of your own**: `random_state=self.seed` (1234), never a number; no `np.random.seed(0)` or `torch.manual_seed(0)`.
- `predict(X)`: values or labels. `predict_proba(X)`: shape `(n_rows, n_classes)` in the order of `self.classes_`, rows summing to 1.
- Weights load from `self.weights_dir`. No downloads in `fit` or `predict`: the runner runs the model without network. Point the package's cache setting at `self.weights_dir` if it downloads on first use.
- No reading of `splits/` or the tables, no writes outside a temporary folder, no subprocess, no environment variables beyond device settings.
- Keep it short: a translation between the contract and the package.

`models/ridge/model.py` is a complete numeric-only example.

## 4. `requirements.txt`: the shared environment

Every model runs in one shared environment, `env/requirements.txt` (the paper's versions). List the packages `model.py` imports, each as `name==version` with exactly the version in `env/requirements.txt`; not `strable-leaderboard` itself. If the model needs a package that is missing, ADD one line `name==version` for it to `env/requirements.in` and touch nothing else in `env/`; never change an existing pin, Checks fails on it. Tell the user that the pull request will then wait for the maintainer, who relocks the environment. If it needs another version of a package already there, stop and tell the user to open an issue.

## 5. Smoke test

Python 3.12 or later:

```
pip install -c env/requirements.txt -e ".[smoke]" -r models/<name>/requirements.txt
python -m strable_leaderboard.smoke models/<name> --device cpu
```

It checks the card, the pins and the code, then runs three small tables. Its search for own folds and seeds is a regular-expression hint; the review is the real check, so follow step 3 whatever it says. Fix every FAIL line and run it again until it passes. A GPU model whose package cannot run on CPU: run `--static` and say in the pull request that the full smoke test needs a GPU; the maintainers run it on theirs after the merge.

## 6. Pull request

Push `add-<name>` to the fork and open the pull request against `gblayer/strable-leaderboard`, `main`: title `Add <Name> <version>`, only `models/<name>/` in the diff (plus the added lines of `env/requirements.in` when step 4 required them), `.github/pull_request_template.md` filled, the smoke test output pasted. Then tell the user what happens next: Checks and two Claude reviews; if all pass it merges by itself and the maintainers run the full benchmark; otherwise the maintainer looks at it.
