# For coding agents

Read [README.md](README.md), then the document for the task:

- Adding a model: `.claude/skills/add-model/SKILL.md`, with `docs/submitting.md` and `strable_leaderboard/model.py`. Fill every field of `model.yaml` yourself from the model's package, documentation and paper, and run the smoke test before opening the pull request.
- Reviewing a pull request: `.claude/skills/review-submission/SKILL.md`.
- Running a model on the cluster and publishing its results: `.claude/skills/run-model/SKILL.md`, with `docs/automation.md`. Normally the dispatcher (`scripts/dispatch.py`) does it.
- Any change to scores, folds, encoders or times: `docs/protocol.md`; every decision taken so far is in `docs/decisions.md` and holds unless the maintainer changes it.

Every model runs in one shared environment pinned in `env/requirements.txt`; a model never brings its own. Models never make their own folds or seeds: the runner gives the paper's outer folds, early stopping uses `self.validation_split(y)`, the seed is `self.seed`.

Keep code and prose short and plain. Never commit per-fold results, data, weights, `cache/` or `private/` to this repository, and never name the cluster in it.
