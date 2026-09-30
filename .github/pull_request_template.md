## Model

- **Name and version:**
- **End-to-end** (reads raw strings) **or numeric-only:**
- **Package:** `name==version`
- **Checkpoint:** Hugging Face repository and commit hash, or none
- **Trained or tuned on any STRABLE table?** No / Yes, which:

## Checklist

- [ ] Only `models/<name>/` changes
- [ ] `model.yaml` is complete, and the checkpoint revision is a commit hash
- [ ] `requirements.txt` pins every package with `==`
- [ ] Default configuration, with no hyperparameter search
- [ ] No network access in `fit` or `predict`: weights load from `self.weights_dir`
- [ ] The smoke test passes locally

## Smoke test output

```
paste the output of: python -m strable_leaderboard.smoke models/<name>
```
