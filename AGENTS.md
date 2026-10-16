# For coding agents

## Adding a model (contributors)

Follow `.claude/skills/add-model/SKILL.md`, with `docs/submitting.md` and `strable_leaderboard/model.py`. In short:

- One pull request, from a fork, adding one folder `models/<name>/`: `model.py`, `model.yaml`, `requirements.txt`. One folder per model version; the version lives in `model.yaml`. Some folder names are taken (list in `docs/submitting.md`).
- Fill every field of `model.yaml` yourself from the model's package, documentation and paper, and run the smoke test before opening the pull request.
- Every model runs in one shared environment, `env/requirements.txt`. In `requirements.txt`, write the model's version. A missing package is asked for in the pull request text and added as one line to `env/requirements.in`; a line already there never changes.
- The board's folds are used for everything: early stopping uses `self.validation_split(y)`, the seed is `self.seed`.

Maintainer tasks (reviews, runs, results): `docs/automation.md`, section 6.

Keep code and prose short and plain. Never commit `results/`, `runs/`, data, weights, `cache/` or `private/`.
