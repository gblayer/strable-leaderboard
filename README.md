<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/banner-dark.svg">
  <img alt="STRABLE leaderboard: the living leaderboard of 108 real-world tables with raw strings" src="docs/banner-light.svg" width="100%">
</picture>

[![Leaderboard](https://img.shields.io/badge/leaderboard-gblayer.github.io%2Fstrable--leaderboard-4E0000?style=flat-square&labelColor=13273F)](https://gblayer.github.io/strable-leaderboard/)
[![Paper](https://img.shields.io/badge/paper-arXiv%202605.12292-223631?style=flat-square&labelColor=13273F)](https://arxiv.org/abs/2605.12292)
[![Data](https://img.shields.io/badge/data-Hugging%20Face-223631?style=flat-square&labelColor=13273F)](https://huggingface.co/datasets/inria-soda/STRABLE-benchmark)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-Evaluation%20and%20Datasets%20track-4E0000?style=flat-square&labelColor=13273F)](#) <!-- TODO: link to the NeurIPS 2026 video presentation -->

The living leaderboard of [STRABLE](https://arxiv.org/abs/2605.12292), the benchmark of 108 real-world tables with raw strings: 76 regression and 32 classification tasks. One score and one run time per pipeline, Pareto-optimal pipelines first.

**Leaderboard:** https://gblayer.github.io/strable-leaderboard/

⭐ If you find STRABLE useful, star the repository and cite the paper ([BibTeX](#citation)) 🙂

When you report a position on the board, give the version and date shown at the top of the page (for example v0.1.2, 2026-10-14): the board changes as models are added.

## Submit a model

<!-- The page's "Submit a model" button links to this heading (#submit-a-model): keep its text. -->

One pull request adds one folder, `models/<name>/`, with three files: `model.py` (your model), `model.yaml` (its card) and `requirements.txt` (its version).

1. **Fork** the repository on GitHub and clone your fork.
2. **Add one folder**, `models/<name>/`, starting from [models/_template/](models/_template/), and fill its three files. One folder per model version, named with lowercase letters, digits, dots and hyphens. Folder names already used by the paper's models (`ridge`, `xgboost`, `extratrees`, `catboost`, `tabpfn-2.5`, `tabicl-v2`, `realmlp`, `tabm`, `mambular`, `contexttab`, `tabstar`, `tarte`) and `causilo` are taken.
3. **Run the smoke test**, as in [Check your model locally](#check-your-model-locally).
4. **Open a pull request** from your fork. Only `models/<name>/` changes, plus the lines you add to `env/requirements.in`.

Step by step, with the model card and examples: [docs/submitting.md](docs/submitting.md).

**One shared environment.** All models run in one Python environment, [env/requirements.txt](env/requirements.txt), so that results stay comparable. In your `requirements.txt`, write your model's version, as listed there. If your model needs a package that is missing, ask for it in the pull request text and add one line `name==version` to [env/requirements.in](env/requirements.in). The maintainer decides on any other version.

## Ask your agent

Your agent can do steps 1 to 4. Tell it:

```
Add my model to the STRABLE leaderboard, test it, and open a pull request.
```

It reads [AGENTS.md](AGENTS.md) (Claude Code reads it through [CLAUDE.md](CLAUDE.md)) and follows [.claude/skills/add-model/SKILL.md](.claude/skills/add-model/SKILL.md).

## What happens next

- Automatic checks run, and Claude reviews the pull request twice.
- If the checks and both reviews pass, the pull request merges by itself. Otherwise the maintainer looks at it.
- After the merge, we run your model on all 108 tables × 3 folds, and it appears on the board.

How this works: [docs/automation.md](docs/automation.md).

## Check your model locally

With Python 3.12 or later:

```
pip install -c env/requirements.txt -e ".[smoke]" -r models/<name>/requirements.txt
python -m strable_leaderboard.smoke models/<name>
```

The first line installs this package, the packages the test needs and your model's packages, at the shared environment's versions. The second runs your model on outer fold 0 of three small tables and checks its card and predictions; it does not compute leaderboard scores.

## Layout

| Path | |
|---|---|
| [models/](models/) | One folder per model: `_template/` to start from, `ridge/` as a complete example (the board's Ridge results are the paper's) |
| [strable_leaderboard/](strable_leaderboard/) | Folds, scores, the model contract and the smoke test |
| [splits/](splits/) | The folds of all 108 tables: for each row, its test fold and its validation groups |
| [env/](env/) | The shared environment: `requirements.txt` (exact versions), `requirements.in` (where a new package is added) |
| [Paper](https://arxiv.org/abs/2605.12292) | The evaluation protocol: data, cross-validation, scores, time, encoders |
| [docs/submitting.md](docs/submitting.md) | How to add a model: the steps, the contract, the model card, examples |
| [docs/automation.md](docs/automation.md) | What happens after a pull request is opened |
| [site/](site/) | The page and its data: `results.json`, `results.parquet`, and every fold in `results_per_fold.csv` and `results_per_fold.parquet` |
| [AGENTS.md](AGENTS.md) | Where coding agents start |

Maintainer files: `scripts/`, `tests/`.

## Contact

Gioia Blayer, gioia.blayer@inria.fr

## Citation

```bibtex
@misc{blayer2026strablebenchmarkingtabularmachine,
      title={STRABLE: Benchmarking Tabular Machine Learning with Strings},
      author={Gioia Blayer and Myung Jun Kim and Félix Lefebvre and Lennart Purucker and Alan Arazi and Eilam Shapira and Roi Reichart and Frank Hutter and Marine Le Morvan and David Holzmüller and Gaël Varoquaux},
      year={2026},
      eprint={2605.12292},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.12292},
}
```

## Issues and ideas

[Open an issue](../../issues/new/choose) for a question, a bug, a wrong score, a table problem or a feature.
