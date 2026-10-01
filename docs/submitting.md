# Submitting a model

You submit code, we run it. One pull request adds one folder, `models/<name>/`:

| File | What it holds |
|---|---|
| `model.py` | `class Model(StrableModel)` with `fit`, `predict` and `predict_proba` |
| `model.yaml` | The model card, below |
| `requirements.txt` | Exact versions: `tabpfn==6.0.6` |

`models/_template/` is the starting point, `models/ridge/` a complete example.

## Three steps

1. `cp -r models/_template models/<name>` and fill the three files. The folder name carries the version: `tabpfn-2.5`.
2. Test: `pip install -e ".[dev]" -r models/<name>/requirements.txt && python -m strable_leaderboard.smoke models/<name>`
3. Open the pull request, title `Add <Name> <version>`.

Or clone the repository and tell your coding agent: *"Add my model to the STRABLE leaderboard, test it, and open a pull request."* It follows `.claude/skills/add-model/SKILL.md`.

After that: Claude reviews the pull request and posts a verdict, a maintainer merges, the runner computes all 108 tables, and the board updates. See [automation.md](automation.md).

## The model contract

The contract is the interface every model implements, so that the runner can train, time and score all models the same way. It is the class `StrableModel` in `strable_leaderboard/model.py`:

- `handles_strings = True`: `fit` and `predict` receive the raw table, and the row is tagged **end-to-end**. `False`: they receive a numeric table from one of the five string encoders, and the runner pairs the model with all five.
- `fit(X, y)` trains on the outer training rows. Classification targets are integers 0..K-1; the model sets `self.classes_`.
- `predict(X)` returns values or labels. `predict_proba(X)` returns one column per class, in the order of `self.classes_`.
- `self.seed`, `self.device`, `self.n_jobs` are given. Weights load from `self.weights_dir`: there is no network during a run.
- Default configuration. Early stopping may use `self.validation_split(y)`, the protocol's first inner fold.
- Never read the tables, the split files or the test rows yourself.

## The model card

```yaml
name: TabPFN
version: "2.5"
handles_strings: false          # false: five encoder rows. true: end-to-end
tasks: [regression, binary, multiclass]
access: open                    # open | closed | api
hardware: gpu                   # gpu | cpu
gpu_memory_gb: 24               # optional

package: tabpfn==6.0.6          # the code, same pin as requirements.txt
checkpoint:                     # the weights, or null
  repo: Prior-Labs/TabPFN-2.5
  revision: 0123456789abcdef    # commit hash
licence: Prior Labs licence
trained_on_strable: []          # STRABLE tables seen in pre-training or tuning

code: https://github.com/PriorLabs/TabPFN      # optional
paper: https://arxiv.org/abs/0000.00000        # optional
```

What the board derives from it:

| Field | On the board |
|---|---|
| `handles_strings` | The end-to-end tag, or one row per encoder |
| `access` | `api`: an API row, off the Pareto front. `closed`: a closed tag |
| `hardware` | The "Runs on" column |
| `trained_on_strable` | A contamination note on the row |

There is no family field. Whether a model is a tree ensemble or a foundation model changes nothing in how it runs or ranks, and its own page says so.

**Variants.** Every row is the default configuration. Tuned and tuned-plus-ensembled variants are a property of a run, not of the model, so they need no card field: when they come, the runner records `variant` in the result rows and the board shows them as separate rows.

## Package, weights, checkpoint

- The **package** is the code, installed with pip and pinned in `requirements.txt`.
- The **weights** are what the model learned in pre-training. A **checkpoint** is the file holding them, on Hugging Face. For the leaderboard the two words mean the same file, pinned by its repository and commit hash in `model.yaml`.
- The repository stores neither: only the pins. The runner downloads the checkpoint into a cache before the job starts, and jobs run offline. If a package downloads its weights on first use, point its cache setting at `self.weights_dir`.

## Versions

Each version is its own folder and its own row, forever: `models/tabpfn-2.5/`, `models/tabpfn-3/`, `models/tabpfn-3.5/`. Each pins its own package version and checkpoint hash, so a new release never changes an older row. If a new checkpoint loads with the same package, the new folder can copy `model.py` and change only `model.yaml`. The runner computes only what is missing.

## Open, closed, anonymous, API

| | Code and weights | Runs on | Pareto front | How to submit |
|---|---|---|---|---|
| Open | Public | Our hardware | Yes | Public pull request |
| Closed | Private, shared with us | Our hardware | Yes, with a *closed* tag | Private channel |
| Anonymous | Private, shared with us | Our hardware | Yes, under a codename with an *anonymous* tag | Private channel |
| API | Never leave the vendor | The vendor's servers | No: the time includes the network | Private channel |

Closed and anonymous are the same case; anonymous also hides the name. Closed, anonymous and API models do not fit a public pull request. Instead:

1. The submitter puts the package and the checkpoint in a private Hugging Face repository (Settings → *Make private*) and adds the runner account as a read collaborator (Settings → *Collaborators*). A private repository is visible only to its owner and its collaborators.
2. The submitter emails the model card and `model.py` to the maintainer. For an API model, the card says how the runner authenticates, the vendor pays for the calls and agrees not to train on the data, and the API version is pinned.
3. The folder goes into the private repository `strable-leaderboard-private`, with the same layout as `models/`. Anonymous entries get a codename from `strable_leaderboard.codenames` and `anonymous: true` in the card; the codename-to-model mapping never leaves that repository.
4. Results are published here like any other, under the name or the codename.

## Examples

**Open, numeric-only: Kumo Tabular.** NVIDIA's in-context model takes numerical and categorical columns; text needs preprocessing, so on STRABLE it is paired with the five encoders, like TabPFN and TabICL.
1. `cp -r models/_template models/kumo-tabular-1.0`
2. `model.py`: `handles_strings = False`; `fit` builds `sdm.models.KumoTabular` on `self.device` and stores the context; `predict_proba` returns the class probabilities in the order of `self.classes_`.
3. `model.yaml`: `access: open`, `hardware: gpu`, `package: structured-data-models==<version>`, `checkpoint: {repo: nvidia/Kumo-Tabular, revision: <commit hash>}`, `licence: OpenMDW-1.1 weights, Apache-2.0 code`.
4. Smoke test on a GPU machine, then the pull request `Add Kumo Tabular 1.0`.

**Open, end-to-end: a model that reads strings itself.** Same steps with `handles_strings = True`; `fit` receives the raw DataFrame with its string, numeric and datetime columns. The board shows one row with an end-to-end tag.

**Closed or anonymous.** The private channel above. On the board the row looks like any other, plus its tag.

**API.** The private channel above. The row shows approximate times, and the plot draws it as a hollow square, off the front.
