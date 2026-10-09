# For coding agents

## Adding a model (contributors)

Follow `.claude/skills/add-model/SKILL.md`, with `docs/submitting.md` and `strable_leaderboard/model.py`. In short:

- Open models only (public code and weights): no closed, anonymous or API-only models.
- One pull request, from a fork, adding one folder `models/<name>/`: `model.py`, `model.yaml`, `requirements.txt`. One folder per model version; the version lives in `model.yaml`. The folder names of the paper's learners are reserved (list in `docs/submitting.md`).
- Fill every field of `model.yaml` yourself from the model's package, documentation and paper, and run the smoke test before opening the pull request.
- Every model runs in one shared environment, `env/requirements.txt`; a model never brings its own. A missing package is ADDED as one line to `env/requirements.in`; an existing pin never changes.
- Models never make their own folds, searches or seeds: the runner gives the paper's outer folds, early stopping uses `self.validation_split(y)`, the seed is `self.seed`.

## Maintainers

- Reviewing a pull request: `.claude/skills/review-submission/SKILL.md`.
- Running a model on the cluster and publishing its results: `.claude/skills/run-model/SKILL.md`, with `docs/automation.md`. Normally the dispatcher (`scripts/dispatch.py`) does it.
- Scores, folds, encoders and times follow the paper (https://arxiv.org/abs/2605.12292); only the maintainer changes them.

Keep code and prose short and plain. Results reach this repository only as the results files of `site/` (`RESULTS_FILES`: the aggregates and the per-fold rows `scripts/build_site_data.py` writes, and the leakage flags); never commit `results/`, `runs/`, data, weights, `cache/` or `private/`, and never name the cluster in it.
