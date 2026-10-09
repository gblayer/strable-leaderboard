<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/banner-dark.svg">
  <img alt="STRABLE leaderboard: the living leaderboard of 108 real-world tables with raw strings" src="docs/banner-light.svg" width="100%">
</picture>

[![Leaderboard](https://img.shields.io/badge/leaderboard-gblayer.github.io%2Fstrable--leaderboard-4E0000?style=flat-square&labelColor=13273F)](https://gblayer.github.io/strable-leaderboard/)
[![Paper](https://img.shields.io/badge/paper-arXiv%202605.12292-223631?style=flat-square&labelColor=13273F)](https://arxiv.org/abs/2605.12292)
[![Data](https://img.shields.io/badge/data-Hugging%20Face-223631?style=flat-square&labelColor=13273F)](https://huggingface.co/datasets/inria-soda/STRABLE-benchmark)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-Evaluation%20and%20Datasets%20track-4E0000?style=flat-square&labelColor=13273F)](#) <!-- TODO: link to the NeurIPS 2026 video presentation -->

The living leaderboard of [STRABLE](https://arxiv.org/abs/2605.12292), the benchmark of 108 real-world tables with raw strings: 76 regression and 32 classification tasks. One score and one run time per pipeline, Pareto-optimal pipelines first.

If you benchmark on STRABLE, [star this repository](https://github.com/gblayer/strable-leaderboard/stargazers) and cite the paper ([BibTeX](#citation)). To claim a rank, name the board version and date shown at the top of the page.

**Leaderboard:** https://gblayer.github.io/strable-leaderboard/

**Status: pre-launch.** Protocol, cross-validation folds, the model contract with its smoke test, the review automation and the site are here. The first results come with the launch set in November 2026.

## Submit a model

Clone this repository, then tell your coding agent:

```
Add my model to the STRABLE leaderboard, test it, and open a pull request.
```

It follows [.claude/skills/add-model/SKILL.md](.claude/skills/add-model/SKILL.md). By hand, and for closed or anonymous models: [docs/submitting.md](docs/submitting.md), which also has the diagram of what happens to a pull request.

You submit code, not results. After the merge we run the model on all 108 tables and the board updates.

## Issues and ideas

[Open an issue](../../issues/new/choose) for a bug, a wrong score, a table problem or a feature.

## Layout

| Path | |
|---|---|
| [docs/plan.md](docs/plan.md) | How the leaderboard works and what comes next |
| [docs/protocol.md](docs/protocol.md) | Data, cross-validation, score, time, encoders, hardware, result records |
| [docs/submitting.md](docs/submitting.md) | Submitting: the contract, the model card, versions, open/closed/anonymous, examples |
| [docs/automation.md](docs/automation.md) | Checks, Claude review, the runner, issues, safety, setup |
| [docs/site.md](docs/site.md) | Where the site lives, hosting, visits, design |
| [docs/decisions.md](docs/decisions.md) | Every decision, with its date |
| [docs/research.md](docs/research.md) | Notes on TabArena, the paper and other leaderboards |
| [splits/](splits/) | Outer and inner cross-validation folds of all 108 tables |
| [strable_leaderboard/](strable_leaderboard/) | Splits, metrics, model contract, smoke test, one run (`run.py`), results check |
| [scripts/](scripts/) | Runner: prepare, make_tasks, run_array, gather_results; import_paper_results; build_site_data |
| [site/results.json](site/results.json) | The aggregates the page shows; per-fold results stay private |
| [models/](models/) | One folder per model: `_template/` to start from, `ridge/` as a working example |
| [site/index.html](site/index.html) | The page served at https://gblayer.github.io/strable-leaderboard/ |
| [.github/workflows/](.github/workflows/) | Checks, Claude review, Claude on issues, Pages deployment |

## Develop

```
pip install -e ".[dev]"
pytest
python -m strable_leaderboard.smoke models/ridge
python -m strable_leaderboard.run models/ridge --encoder tfidf --table drug-ndc --fold 0 --device cpu
```

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
