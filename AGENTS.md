# For coding agents

Read [README.md](README.md), then the document for the task:

- Adding a model: `.claude/skills/add-model/SKILL.md`, with `docs/submitting.md` and `strable_leaderboard/model.py`.
- Reviewing a pull request: `.claude/skills/review-submission/SKILL.md`.
- Running a model on the cluster and publishing its results: `.claude/skills/run-model/SKILL.md`, with `docs/automation.md`.
- Scores, folds, encoders and times follow the paper (https://arxiv.org/abs/2605.12292); only the maintainer changes them.

Keep code and prose short and plain. Never commit `results/`, `private/`, data or weights to this repository, and never name the cluster or the institution in it: describe hardware as GPU, CPU and memory.
