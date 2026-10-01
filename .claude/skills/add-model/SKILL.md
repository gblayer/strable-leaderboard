---
name: add-model
description: Add a model to the STRABLE leaderboard. Creates models/<name>/ from the template, wraps the model in the StrableModel contract, fills the model card, pins dependencies, runs the smoke test and opens a pull request. Use when someone asks to add, submit, integrate or benchmark a model on the STRABLE leaderboard.
---

# Add a model to the STRABLE leaderboard

Goal: one pull request adding `models/<name>/`, passing the smoke test and `.claude/skills/review-submission/SKILL.md`. Read `docs/submitting.md` and `strable_leaderboard/model.py` first.

## 1. Facts

Take them from the user's repository, package docs or model card; ask only for what you cannot find.

- Name and version. One folder per version: `tabpfn-3.5`, not `tabpfn`.
- The pip package and its exact version.
- The weights: a Hugging Face repository and a commit hash, or none.
- Does it read raw string columns itself? Yes: end-to-end, `handles_strings: true`. No: it gets numbers from the five encoders.
- Tasks, gpu or cpu, GPU memory, licence, code and paper links.
- STRABLE tables in its pre-training or tuning data, for `trained_on_strable`.

Closed, anonymous or API model: stop here and point the user to `docs/submitting.md`, section "Open, closed, anonymous, API". They do not go through a public pull request.

## 2. Folder

`cp -r models/_template models/<name>`. Lowercase letters, digits, dots and hyphens.

## 3. `model.py`

`class Model(StrableModel)`:
- Set `handles_strings`.
- `fit(X, y)`: the default configuration; pass `self.seed`, `self.device`, `self.n_jobs`; for classification set `self.classes_ = np.unique(y)`; early stopping only through `fit_idx, val_idx = self.validation_split(y)`.
- `predict(X)`: values or labels. `predict_proba(X)`: shape `(n_rows, n_classes)` in the order of `self.classes_`, rows summing to 1.
- Weights load from `self.weights_dir`. No downloads in `fit` or `predict`: point the package's cache setting at `self.weights_dir` if it downloads on first use.
- No hyperparameter search, no reading of `splits/` or the tables, no writes outside a temporary folder.
- Keep it short: a translation between the contract and the package.

`models/ridge/model.py` is a complete numeric-only example.

## 4. `model.yaml` and `requirements.txt`

Every field of `models/_template/model.yaml` unless marked optional. `requirements.txt` pins every package with `==`, the model's package included; not `strable-leaderboard` itself.

## 5. Smoke test

```
pip install -e ".[dev]" -r models/<name>/requirements.txt
python -m strable_leaderboard.smoke models/<name>
```

Fix every FAIL line. A GPU-only model on a machine without a GPU: say so in the pull request rather than forcing CPU.

## 6. Pull request

Branch `add-<name>`, title `Add <Name> <version>`, only `models/<name>/` in the diff. Fill `.github/pull_request_template.md` and paste the smoke test output. The maintainers run the full benchmark after the merge.
