# STRABLE leaderboard

The living leaderboard of [STRABLE](https://arxiv.org/abs/2605.12292), the benchmark of 108 real-world tables with raw strings: 76 regression and 32 classification tasks. Accepted to the NeurIPS 2026 Evaluation and Datasets track. The board will live at **strable.ai**.

**Status: pre-launch.** Protocol, cross-validation folds, the model contract with its smoke test, the review automation and the site design are here. The first results come with the launch set in November 2026.

| | |
|---|---|
| Paper | https://arxiv.org/abs/2605.12292 |
| Data | https://huggingface.co/datasets/inria-soda/STRABLE-benchmark |
| Paper code | https://github.com/soda-inria/strable |

## Submit a model

Clone this repository, then tell your coding agent:

```
Add my model to the STRABLE leaderboard, test it, and open a pull request.
```

It follows [.claude/skills/add-model/SKILL.md](.claude/skills/add-model/SKILL.md). By hand: [docs/submitting.md](docs/submitting.md).

You submit code, not results. After the merge we run the model on all 108 tables and the board updates.

## Issues and ideas

[Open an issue](../../issues/new/choose) for a bug, a wrong score, a table problem or a feature.

## Layout

| Path | |
|---|---|
| [docs/plan.md](docs/plan.md) | How the leaderboard works and what comes next |
| [docs/protocol.md](docs/protocol.md) | Data, cross-validation, score, time, encoders, hardware, result records |
| [docs/submitting.md](docs/submitting.md) | Submitting: the contract, the model card, versions, open/closed/anonymous/API, examples |
| [docs/automation.md](docs/automation.md) | Checks, Claude review, the runner, issues, safety, setup |
| [docs/site.md](docs/site.md) | Hosting and design of strable.ai |
| [docs/decisions.md](docs/decisions.md) | Every decision, with its date |
| [docs/research.md](docs/research.md) | Notes on TabArena, the paper and other leaderboards |
| [splits/](splits/) | Outer and inner cross-validation folds of all 108 tables |
| [strable_leaderboard/](strable_leaderboard/) | Splits, metrics, model contract, smoke test, results check |
| [models/](models/) | One folder per model: `_template/` to start from, `ridge/` as a working example |
| [site/index.html](site/index.html) | Design of strable.ai |
| [.github/workflows/](.github/workflows/) | Checks, Claude review, Claude on issues |

## Develop

```
pip install -e ".[dev]"
pytest
python -m strable_leaderboard.smoke models/ridge
```

## Contact

Gioia Blayer, gioia.blayer@inria.fr
