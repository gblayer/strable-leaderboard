# For coding agents

Read [README.md](README.md), then the document for the task:

- Adding a model: `.claude/skills/add-model/SKILL.md`, with `docs/submitting.md` and `strable_leaderboard/model.py`.
- Reviewing a pull request: `.claude/skills/review-submission/SKILL.md`.
- Running a model on the cluster and publishing its results: `.claude/skills/run-model/SKILL.md`, with `docs/automation.md`.
- Any change to scores, folds, encoders or times: `docs/protocol.md`; every decision taken so far is in `docs/decisions.md` and holds unless the maintainer changes it.

Keep code and prose short and plain. Never commit per-fold results, data or weights to this repository.
