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

The board currently holds the paper's 75 Figure 4a pipelines, version v0.1.0-pre.

## Submit a model

<!-- The page's "Submit a pipeline" button links to this heading (#submit-a-model): keep its text. -->

You submit code, not results.

1. **Fork** the repository on GitHub, your own copy since you cannot push to this one, and clone your fork.
2. **Add one folder**, `models/<name>/`, with `model.py`, `model.yaml` and `requirements.txt`, starting from [models/_template/](models/_template/). One folder per model version, named with lowercase letters, digits, dots and hyphens; the version lives in `model.yaml`, and the names of the paper's learners are reserved ([list](docs/submitting.md)). No container, no bash script: every model runs in one shared environment, and each pin in your `requirements.txt` must equal its version in [env/requirements.txt](env/requirements.txt). A package that is not there: add one line `name==version` to [env/requirements.in](env/requirements.in) in the same pull request; never change an existing pin. Your coding agent can do it; tell it:
   ```
   Add my model to the STRABLE leaderboard, test it, and open a pull request.
   ```
   It follows [AGENTS.md](AGENTS.md) (Claude Code reads it through [CLAUDE.md](CLAUDE.md)) and [.claude/skills/add-model/SKILL.md](.claude/skills/add-model/SKILL.md).
3. **Run the smoke test**, as in [Check your model locally](#check-your-model-locally).
4. **Open a pull request** from your fork. Only `models/<name>/` may change, plus added lines in `env/requirements.in`.

**What happens next.** Automatic checks run and Claude reviews the pull request twice. If everything passes, it is merged automatically; otherwise the maintainer reviews it. After the merge we run the model on all 108 tables × 3 folds and the board updates. Details: [docs/submitting.md](docs/submitting.md).

## Check your model locally

With Python 3.12 or later:

```
pip install -c env/requirements.txt -e ".[smoke]" -r models/<name>/requirements.txt
python -m strable_leaderboard.smoke models/<name>
```

The first line installs this package, the smoke-test tools and your model's packages, at the shared environment's versions. The second runs your model on outer fold 0 of three small tables and checks its card and predictions; it does not compute leaderboard scores.

## Layout

| Path | |
|---|---|
| [models/](models/) | One folder per model: `_template/` to start from, `ridge/` as a working example |
| [strable_leaderboard/](strable_leaderboard/) | Splits, metrics, the model contract and the smoke test |
| [splits/](splits/) | Outer and inner cross-validation folds of all 108 tables |
| [env/](env/) | The shared environment every model runs in: `requirements.txt` (pins), `requirements.in` (where new packages are added) |
| [docs/protocol.md](docs/protocol.md) | Data, cross-validation, score, time, encoders, hardware |
| [docs/submitting.md](docs/submitting.md) | The contract, the model card, versions, examples |
| [AGENTS.md](AGENTS.md) | Where coding agents start |

Maintainer and infrastructure files: `scripts/`, `site/`, `tests/`, `docs/`.

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

[Open an issue](../../issues/new/choose) for a bug, a wrong score, a table problem or a feature.
