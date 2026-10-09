## Model

- **Name and version:**
- **End-to-end** (reads raw strings) **or numeric-only:**
- **Package:** `name==version`
- **Checkpoint:** Hugging Face repository and commit hash, or none
- **Trained or tuned on any STRABLE table?** No / Yes, which:

## Checklist

- [ ] Only `models/<name>/` changes (opened from a fork, branch `add-<name>`), plus at most added lines in `env/requirements.in`
- [ ] Every field of `model.yaml` is filled, and the checkpoint revision is a full commit hash
- [ ] `requirements.txt` pins each package with `==` at the version of `env/requirements.txt`, the shared environment; a missing package is added as one line to `env/requirements.in`, no existing pin changed
- [ ] Default configuration, with no hyperparameter search
- [ ] No split or cross-validation of its own: early stopping only through `self.validation_split(y)`; every seed is `self.seed`
- [ ] No network access in `fit` or `predict`: weights load from `self.weights_dir`
- [ ] The smoke test passes locally (`--static` for a GPU model on a machine without GPU)

## Smoke test output

```
paste the output of: python -m strable_leaderboard.smoke models/<name>
```
