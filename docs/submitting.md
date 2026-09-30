# Submitting a model

You submit code, not results: we run every model on our hardware so that scores and times are comparable.

The fastest way: clone this repository and tell your coding agent *"Add my model to the STRABLE leaderboard, test it, and open a pull request."* It follows `.claude/skills/add-model/SKILL.md`. The rest of this page is the same process by hand.

## What goes into a pull request

One folder, `models/<name>/`, copied from `models/_template/`:

| File | What it is | Why |
|---|---|---|
| `model.py` | A class `Model(StrableModel)` with `fit`, `predict` and `predict_proba` | So the runner can call every model the same way |
| `model.yaml` | The model card: name, version, family, access, package, checkpoint, licence | So the leaderboard can label the row, and anyone can find the exact code and weights |
| `requirements.txt` | Exact package versions, for example `tabpfn==6.0.6` | So a run can be reproduced months later |

This is not a Docker container. A container would bundle a whole operating system, would be heavy to review, and Margaret runs jobs through SLURM without Docker. A pinned Python environment is enough for almost every tabular model. If a model ever needs system libraries, an Apptainer image is the fallback, if Margaret supports it.

`models/ridge/` is a complete, working example.

## The contract

`strable_leaderboard/model.py` defines it. In short:

- `handles_strings = True`: the model receives the raw table and is listed as **end-to-end**. `False`: it receives numbers from one of the five string encoders, and the runner pairs it with all five.
- `fit(X, y)` trains on the outer training rows. Classification targets arrive as integers 0..K-1, and the model sets `self.classes_`.
- `predict(X)` returns values or labels. `predict_proba(X)` returns one column per class.
- Use `self.seed`, `self.device` and `self.n_jobs`. Load weights from `self.weights_dir`: jobs have no internet.
- Default configuration only. Early stopping may use `self.validation_split(y)`.
- Never read the tables or the split files yourself, and never touch the test rows during `fit`.

## Package, weights and checkpoints

- **The package** is the code: the model architecture, the preprocessing, the inference loop. You install it with pip, for example `tabpfn` from PyPI.
- **The weights** are the numbers the model learned during pre-training. Without them, a foundation model is an empty network.
- **A checkpoint** is the file that stores the weights, for example `tabpfn-v2.5-regressor.ckpt` or a `.safetensors` file on Hugging Face. Some checkpoints also store training state, which inference ignores. For the leaderboard, "weights" and "checkpoint" mean the same file.

A run needs both, pinned: the package to an exact version in `requirements.txt`, the checkpoint to a commit hash in `model.yaml`. Many packages download their checkpoint on first use. On Margaret, the runner downloads it on the login node before the job starts, because compute nodes have no internet.

## New versions of a model

Every version is its own entry, and older entries stay on the board.

Example: TabPFN-2.5 is on the board, and Prior Labs wants to add TabPFN-3.5 and TabPFN-4.
1. They open a pull request with `models/tabpfn-3.5/` and `models/tabpfn-4/`. Each pins its own package version and checkpoint.
2. If a new checkpoint loads with the same package, the folder can reuse the same `model.py`, and only `model.yaml` changes.
3. TabPFN-2.5 keeps its results. Its environment is pinned, so a newer `tabpfn` release doesn't change it.
4. The runner only computes what is missing: the new model with each of the five encoders on every table and fold. Embeddings are cached, so only fit and predict run.

## Model card fields

See `models/_template/model.yaml`. The leaderboard uses:

| Field | Values | Shown as |
|---|---|---|
| `family` | foundation-model, tree-based, neural-network, linear, other | The Family filter |
| `access` | open-source, closed-source, api | The Access filter and tags |
| `handles_strings` | true or false | The end-to-end tag |
| `hardware` | gpu or cpu | The "Runs on" column |
| `trained_on_strable` | tables the model trained on, or [] | A contamination note |

## Anonymous models

A model can appear under a random codename such as **jumping-carrot**, with an *anonymous* tag. It stays on the Pareto front, and its real name is never revealed. The submitter doesn't have to say who they are.

Anonymous and closed-source code can't go through a public pull request. Instead:
1. The submitter gives read access to a private Hugging Face repository holding the package and the checkpoint, and sends the model card through the private submission form.
2. The entry lives in a private repository, `strable-leaderboard-private`, that only the maintainers can read.
3. `strable_leaderboard.codenames.codename()` draws the name. The mapping from codename to model stays in the private repository.
4. The runner treats it like any other model, on the same hardware.

## API models

Models reached over the internet, such as a hosted embedding API:
- They are tagged *API* and kept off the Pareto front, because their time includes the network.
- The model version must be pinned.
- The vendor pays for the calls and agrees not to train on the data.
- Compute nodes have no internet, so API calls run from a machine with access. Where exactly is still to decide.

## Before you open the pull request

```
pip install -e ".[dev]"
python -m strable_leaderboard.smoke models/<name>
```

The smoke test runs your model on one small table per task, outer fold 0. It checks shapes, probabilities and classes, not scores.
