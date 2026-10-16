## Model

- **Name and version:**
- **end_to_end:** true or false
- **Package:** `name==version`
- **Checkpoint:** Hugging Face repository and commit hash, or none
- **Pre-training data** (`pretraining_data` in `model.yaml`), with a link to where the model's documents say it:
- **New package**, only if you add a line to `env/requirements.in`: `name==version`, and why the model needs it:

## Checklist

- [ ] Only `models/<name>/` changes (opened from a fork, branch `add-<name>`), plus at most added lines in `env/requirements.in`
- [ ] Every field of `model.yaml` is filled, and the checkpoint revision is a full commit hash
- [ ] `requirements.txt` gives the model's version, as in `env/requirements.txt`
- [ ] The board's folds are used for everything: early stopping only through `self.validation_split(y)`; every seed is `self.seed`
- [ ] The smoke test passes locally (`--static` for a GPU model on a machine without GPU)

## Smoke test output

```
paste the output of: python -m strable_leaderboard.smoke models/<name>
```
