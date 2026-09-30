---
name: add-model
description: Add a new model to the STRABLE leaderboard. Creates models/<name>/ from the template, wraps the model in the StrableModel contract, fills the model card, pins dependencies, runs the smoke test and opens a pull request. Use when someone asks to add, submit, integrate or benchmark a model or pipeline on the STRABLE leaderboard.
---

# Add a model to the STRABLE leaderboard

You are working in a clone of the strable-leaderboard repository. The goal is a pull request that adds one folder, `models/<name>/`, which passes the smoke test and the review checklist in `.claude/skills/review-submission/SKILL.md`. Read `docs/submitting.md` and `strable_leaderboard/model.py` before writing code.

## 1. Collect the facts

Find these in the user's repository, package docs or model card. Ask the user only for what you can't find:

- Display name and version. Each version gets its own folder, for example `tabpfn-3.5`, not `tabpfn`.
- The pip package and its exact version.
- Where the weights live: a Hugging Face repository and a commit hash, never a branch name. Or null if the model has no weights.
- Whether the model reads raw string columns itself. If yes it is end-to-end, `handles_strings: true`. If it only takes numbers, the leaderboard pairs it with its five string encoders.
- Supported tasks: regression, binary, multiclass.
- GPU or CPU, and the GPU memory it needs.
- Family: foundation-model, tree-based, neural-network, linear or other.
- Access: open-source, closed-source or api.
- Licence, code and paper links.
- Whether it was trained or tuned on any STRABLE table. Look at the training data description, and list any overlap in `trained_on_strable`.

If the model is closed-source or must stay anonymous, stop here. It doesn't go through a public pull request. Point the user to the "Anonymous models" section of `docs/submitting.md`.

## 2. Create the folder

```
cp -r models/_template models/<name>
```

Folder names use lowercase letters, digits, dots and hyphens.

## 3. Write `model.py`

Implement `class Model(StrableModel)`:

- Set `handles_strings`.
- `fit(X, y)`:
  - Build the estimator with its **default** configuration.
  - Pass `self.seed`, `self.device` and `self.n_jobs` through.
  - Classification targets are integers 0..K-1. Set `self.classes_ = np.unique(y)`.
  - If the model needs a validation set for early stopping, use `fit_idx, val_idx = self.validation_split(y)`, and nothing else.
- `predict(X)`: values for regression, labels for classification.
- `predict_proba(X)`: shape `(n_rows, n_classes)`, columns in the order of `self.classes_`, rows summing to 1.
- Load weights from `self.weights_dir`. Jobs have no internet, so no downloads inside `fit` or `predict`. If the package downloads weights on first use, point its cache or path setting at `self.weights_dir`.
- No hyperparameter search. Never read `splits/` or the tables yourself. No writes outside a temporary folder.
- Keep it short. The wrapper translates between the contract and the package, nothing more.

`models/ridge/model.py` is a complete example of a numeric-only model.

## 4. Fill `model.yaml` and `requirements.txt`

- Every field in `models/_template/model.yaml` is required unless marked optional.
- `requirements.txt` pins exact versions with `==`, including the model's package. Don't list `strable-leaderboard` itself.

## 5. Run the smoke test

```
pip install -e ".[dev]"
pip install -r models/<name>/requirements.txt
python -m strable_leaderboard.smoke models/<name>
```

It runs outer fold 0 of one small table per task. Fix every FAIL line. For a GPU-only model on a machine without a GPU, say so in the pull request instead of forcing CPU.

## 6. Open the pull request

- Branch name: `add-<name>`. The pull request only touches `models/<name>/`.
- Fill the template in `.github/pull_request_template.md`, and paste the smoke test output.
- Title: `Add <display name> <version>`.

After the merge, the runner computes the full results on the reference hardware and the leaderboard refreshes. The contributor doesn't run the full benchmark.
