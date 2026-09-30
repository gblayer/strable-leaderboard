# STRABLE leaderboard

A living leaderboard for [STRABLE](https://arxiv.org/abs/2605.12292), the benchmark of 108 real-world tables with raw strings: names, codes, free text and dates. STRABLE was accepted to the NeurIPS 2026 Evaluation and Datasets track. The leaderboard will live at **strable.ai**.

**Status: design phase.** This repository holds the plan, the evaluation protocol, the cross-validation folds of every table, the model contract with a smoke test, and the design of the site. Nothing runs automatically yet.

| Looking for | Go to |
|---|---|
| Paper | https://arxiv.org/abs/2605.12292 |
| Data | https://huggingface.co/datasets/inria-soda/STRABLE-benchmark |
| Code to reproduce the paper | https://github.com/soda-inria/strable |

## Submit a new model

Clone this repository, then tell your coding agent:

```
cd strable-leaderboard
Add my model to the STRABLE leaderboard, test it, and open a pull request.
```

The agent follows [`.claude/skills/add-model/SKILL.md`](.claude/skills/add-model/SKILL.md). To do it by hand, read [docs/submitting.md](docs/submitting.md).

You don't run the benchmark yourself. Once your pull request is merged, we run the model on all 108 tables on our hardware and the leaderboard refreshes.

## Report issues and suggest features

Found a bug, a mislabeled table or a wrong metric? Want a feature? [Open an issue](../../issues/new/choose).

## What is in this repository

| Path | What it is |
|---|---|
| [docs/plan.md](docs/plan.md) | How the leaderboard works, end to end, and what comes next |
| [docs/protocol.md](docs/protocol.md) | Data, cross-validation, metrics, time, encoders, hardware |
| [docs/submitting.md](docs/submitting.md) | How to add a model, package versus checkpoint, anonymous and API models |
| [docs/automation.md](docs/automation.md) | Claude review and merge, issues, the cluster runner, safety |
| [docs/site.md](docs/site.md) | Hosting and design of strable.ai |
| [docs/decisions.md](docs/decisions.md) | Every decision taken so far, with its date |
| [docs/cluster.md](docs/cluster.md) | Facts about Inria's Margaret cluster |
| [docs/research.md](docs/research.md) | Notes on TabArena, the STRABLE paper and code, and other leaderboards |
| [splits/](splits/) | Outer and inner cross-validation folds for all 108 tables |
| [strable_leaderboard/](strable_leaderboard/) | Splits, metrics, model contract, smoke test |
| [models/](models/) | One folder per model: `_template/` to start, `ridge/` as a working example |
| [site/index.html](site/index.html) | Design preview of strable.ai |

## Quick start

```
pip install -e ".[dev]"
pytest
python -m strable_leaderboard.smoke models/ridge
```

## Contact

Gioia Blayer, gioia.blayer@inria.fr
