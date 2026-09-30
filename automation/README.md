# Claude workflows, ready to enable

These workflows are templates. They live here, not in `.github/workflows/`, so they don't run before their secrets exist. See [docs/automation.md](../docs/automation.md) for how the pieces fit together.

| Workflow | Trigger | What Claude does |
|---|---|---|
| `claude-review-submissions.yml` | A pull request that touches `models/` | Reviews the diff against `.claude/skills/review-submission/SKILL.md`, then comments. It adds `claude-approved` when everything passes. It never merges |
| `claude-review-results.yml` | A results pull request from the runner account | Checks the results and merges |
| `claude-issue.yml` | A maintainer adds the label `claude` to an issue | Implements the change on a branch and links a pull request |

## Enable them

1. Make the repository public, which also makes branch protection and Actions free.
2. In Claude Code, run `/install-github-app`, or run `claude setup-token` and store the token as the repository secret `CLAUDE_CODE_OAUTH_TOKEN`. This uses your Claude subscription. An API key in `ANTHROPIC_API_KEY` works too.
3. Create the labels `claude`, `claude-approved` and `needs-maintainer`.
4. Set the repository variable `RUNNER_LOGIN` to the account the cluster runner uses.
5. Protect `main`: require the "Smoke test" check before merging.
6. Copy the three files into `.github/workflows/`.

## Why Claude doesn't merge external submissions

The action runs only for people with write access unless `allowed_non_write_users` is set. Its security guide warns that doing so is risky: keep permissions minimal. A prompt hidden in a pull request could otherwise talk Claude into merging code that then runs on Margaret.

So Claude reviews every submission, and a maintainer merges the ones labelled `claude-approved` with one click.

Fully automatic merging means adding `contents: write` and `Bash(gh pr merge:*)` to `claude-review-submissions.yml`. Only do that together with the runner safeguards in docs/automation.md.
