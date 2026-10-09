# For coding agents

## Adding a model (contributors)

Follow `.claude/skills/add-model/SKILL.md`, with `docs/submitting.md` and `strable_leaderboard/model.py`. In short:

- One pull request, from a fork, adding one folder `models/<name>/`: `model.py`, `model.yaml`, `requirements.txt`. One folder per model version; the version lives in `model.yaml`. The folder names of the paper's learners are reserved (list in `docs/submitting.md`).
- Fill every field of `model.yaml` yourself from the model's package, documentation and paper, and run the smoke test before opening the pull request.
- Every model runs in one shared environment, `env/requirements.txt`; a model never brings its own. A missing package is ADDED as one line to `env/requirements.in`; an existing pin never changes.
- Models never make their own folds, searches or seeds: the runner gives the paper's outer folds, early stopping uses `self.validation_split(y)`, the seed is `self.seed`.

## Maintainers

- Reviewing a pull request: `.claude/skills/review-submission/SKILL.md`.
- Running a model on the cluster and publishing its results: `.claude/skills/run-model/SKILL.md`, with `docs/automation.md`. Normally the dispatcher (`scripts/dispatch.py`) does it.
- Any change to scores, folds, encoders or times: `docs/protocol.md`; every decision taken so far is in `docs/decisions.md` and holds unless the maintainer changes it.

Keep code and prose short and plain. Never commit per-fold results, data, weights, `cache/` or `private/` to this repository, and never name the cluster in it.
