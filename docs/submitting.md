# Submitting a model

One pull request adds one folder, `models/<name>/`:

| File | What it holds |
|---|---|
| `model.py` | `class Model(StrableModel)` with `fit`, `predict` and `predict_proba` |
| `model.yaml` | The model card, below. Every field filled |
| `requirements.txt` | Your model's package and its version: `tabpfn==9.0.0`. The model's own version goes in `model.yaml` |

`models/_template/` is the starting point, `models/ridge/` a complete example.

**Folder names.** One folder per model version, named with lowercase letters, digits, dots and hyphens (`kumo-tabular-1.0`); the version itself lives in `model.yaml`. Folder names already used by models on the board are taken: `ridge`, `xgboost`, `extratrees`, `catboost`, `tabpfn-2.5`, `tabicl-v2`, `realmlp`, `tabm`, `mambular`, `contexttab`, `tabstar`, `tarte`, `causilo`.

**One shared environment.** All models run in one Python environment, [`env/requirements.txt`](../env/requirements.txt), so that results stay comparable. In your `requirements.txt`, write your model's package version, as listed there. If your model needs a package that is missing, ask for it in the pull request text and add one line `name==version` to [`env/requirements.in`](../env/requirements.in); never change a line that is already there. A package that is not on PyPI is pinned to one commit of its git repository, `name @ git+https://<host>/<repository>@<full 40-character commit hash>`, never a branch or a tag; the card's `package` field holds the same line. The maintainer decides on any other version and updates the environment.

## Step by step

You need a GitHub account.

1. **Fork.** On https://github.com/gblayer/strable-leaderboard, click *Fork*. GitHub makes your own copy of the repository, `github.com/<you>/strable-leaderboard`.
2. **Clone your fork**: `git clone https://github.com/<you>/strable-leaderboard && cd strable-leaderboard`, then `git checkout -b add-<name>`.
3. **Add the folder**: `cp -r models/_template models/<name>`, then fill the three files. You can also ask your agent: see [Ask your agent](../README.md#ask-your-agent).
4. **Test** with Python 3.12 or later:
   ```
   pip install -c env/requirements.txt -e ".[smoke]" -r models/<name>/requirements.txt
   python -m strable_leaderboard.smoke models/<name>
   ```
   It fails on a missing or invalid card field, on a version that differs from the shared environment, on wrong predictions, and on code that looks like it makes its own folds or fixes its own seed (a hint only: the review reads the code). Fix every FAIL line. The test runs on CPU by default; add `--device cuda` to use your GPU. If your model cannot run on CPU and you have no GPU, run `--static` and say so in the pull request.
5. **Push and open the pull request**: `git push -u origin add-<name>`, then on GitHub open a pull request from your branch to `gblayer/strable-leaderboard`, `main`, titled `Add <Name> <version>`, with the template filled and the smoke test output pasted. Only `models/<name>/` changes, plus the lines you add to `env/requirements.in`.

## What happens next

- Automatic checks run, and Claude reviews the pull request twice. One comment on the pull request gives both review verdicts and what the model's documents say about its pre-training data.
- If the checks and both reviews pass and only your folder changes, the pull request merges by itself. Otherwise, for example when it adds a package, the maintainer looks at it.
- After the merge, we run the model on all 108 tables × 3 folds. A comment on your pull request gives a rough estimate of when the results will be ready. Then the board updates.
- If the model's pre-training data may overlap with STRABLE tables, the maintainer adds the label `pretraining-overlap`, which puts a small ∩ next to the model's name on the board.

Details: [automation.md](automation.md).

## The model contract

Every model implements the class `StrableModel` of `strable_leaderboard/model.py`, so that all models are trained, timed and scored the same way:

- `end_to_end = True`: the model reads the raw table itself, text columns included, and runs alone. `False`: the model takes numbers only. The board turns the text into numbers with each of its four string encoders, as in the paper (Tf-Idf; the first 30 embedding dimensions of Qwen-3-8B and of LLaMA-3.1-8B; MiniLM-L6-v2 embeddings reduced to 30 by PCA) and runs the model once per encoder. A model runs one way only.
- `fit(X, y)` trains on the training rows of one fold. `X` is a pandas DataFrame: numbers only for a numeric-only model (missing cells stay NaN), the raw table for an end-to-end one. `y` is a NumPy array. Classification targets are integers 0..K-1; the model sets `self.classes_`.
- `predict(X)` returns values or labels. `predict_proba(X)` returns one column per class, in the order of `self.classes_`.
- **The board's folds are used for everything.** Never split, cross-validate or search hyperparameters yourself. Early stopping may only use `fit_idx, val_idx = self.validation_split(y)`; turn off a package's own validation split.
- **No seed of your own.** Pass `self.seed` wherever a seed goes, never a number.
- `self.device` and `self.n_jobs` are given; the runner already caps threads to `self.n_jobs`.
- Use the package's default settings; options that only lift size limits (such as TabPFN's `ignore_pretraining_limits`) are fine. Never read the tables, the split files or the test rows yourself.

## The model card

Every field is required unless marked optional. The smoke test (`strable_leaderboard/card.py`) fails on a missing field, an unknown field, a value outside the allowed ones, a checkpoint revision that is not a full 40-character commit hash, or a placeholder left from the template. The card of `models/tabpfn-3.5/`, with comments:

```yaml
name: TabPFN-3.5
version: "v3.5-20260909"
end_to_end: false               # true: reads the raw table itself. false: numbers only, run with each string encoder
tasks: [regression, binary, multiclass]
variant: default                # the package's default configuration
hardware: gpu                   # gpu | cpu

package: tabpfn==9.0.0          # the code, same version as requirements.txt
checkpoint:                     # the weights, or null
  repo: Prior-Labs/tabpfn_3_5
  revision: 06bf2ba35c80a92a3b9abb436b99cf49e7a0365e   # the full 40-character commit hash
  files: [tabpfn-v3.5-20260909.safetensors]     # optional: only the files the model loads
licence: Prior Labs Non-Commercial (TabPFN-3.5 License v1.0)
pretraining_data: synthetic     # synthetic | real | mixed | none (no pre-training) | not disclosed

code: https://github.com/PriorLabs/TabPFN      # optional
# paper: https://arxiv.org/abs/<id>            # optional
```

| Field | Meaning |
|---|---|
| `end_to_end` | `true`: one row on the board. `false`: one row per string encoder |
| `tasks` | The tasks the model supports: `regression`, `binary`, `multiclass` |
| `variant` | Always `default`: the package's default configuration |
| `licence` | The licence of the code and weights (the field is spelled `licence`) |
| `hardware` | `gpu` or `cpu`: what the board runs the model on, shown when hovering its row. `api` is for a model that runs on its provider's servers, which only the maintainer adds ([automation.md](automation.md#4-the-run), API models); it then needs an `api` block, `auth` and `concurrency` |
| `pretraining_data` | What the model was pre-trained on, as its paper or model card says: `synthetic` (generated data only), `real` (real tables), `mixed` (both), `none` (no pre-training) or `not disclosed` |
| `code`, `paper` | Links to the code and the paper. Optional, but fill them when they exist: the review reads them |

Why a commit hash in `checkpoint.revision`: a Hugging Face repository is a git repository, and vendors update weights in place. The hash pins the exact weights that were benchmarked, so the row can be reproduced later; a branch name like `main` would silently point to newer weights. To get the hash: on the Hugging Face repository, open *Files and versions*, then *History*, and copy the full hash of the commit you tested.

## Package, weights, checkpoint

- The **package** is the code: its version in `requirements.txt` and in the card's `package` field.
- The **weights** are what the model learned in pre-training. A **checkpoint** is the file holding them, on Hugging Face. Here the two words mean the same file, pinned by its repository and commit hash in `model.yaml`.
- The repository stores neither, only the pins. The board downloads the checkpoint before the run. Load the weights from `self.weights_dir`. It is the Hugging Face repository at the pinned revision: its file `<file>` is at `self.weights_dir / "<file>"`. If a package downloads its weights on first use, give it that path or point its cache setting at `self.weights_dir`.
- The smoke test downloads the checkpoint the same way, into `cache/weights/<name>/` of your clone (`--weights-dir` gives another folder).
- A repository with several checkpoints: list the files the model loads in `checkpoint.files` (names or patterns such as `*.ckpt`); only those are downloaded. A checkpoint above 20 GB needs the maintainer's approval.

## Versions

Each model version is its own folder and its own row: `models/tabpfn-3/` and `models/tabpfn-3.5/`, each with its `version` in `model.yaml` and its own checkpoint hash, so a new release never changes an older row. If a new checkpoint loads with the same package, the new folder can copy `model.py` and change only `model.yaml`. A change to a folder already on `main` goes to the maintainer.

## Examples

**Numeric-only: TabPFN-3.5** ([`models/tabpfn-3.5/`](../models/tabpfn-3.5/)). `model.py` sets `end_to_end = False` and loads its checkpoint file from `self.weights_dir`; `model.yaml` is the card above. The board shows four rows, one per string encoder.

**A package that is not in the shared environment: Kumo Tabular.** Its package, `structured-data-models`, is not in `env/requirements.txt`. The pull request adds the line `structured-data-models==<version>` to `env/requirements.in` and asks for it in its text; the maintainer decides and updates the environment. The card has `package: structured-data-models==<version>`, `checkpoint: {repo: nvidia/Kumo-Tabular, revision: <commit hash>}` and the weights' licence, OpenMDW-1.1.

**End-to-end: a model that reads strings itself.** Same steps with `end_to_end = True`. `fit` receives the raw DataFrame, with its string, numeric and datetime columns. The board shows one row.
