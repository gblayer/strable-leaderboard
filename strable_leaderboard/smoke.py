"""Smoke test for one model folder: outer fold 0 of three small tables, one per task.

    python -m strable_leaderboard.smoke models/<name>
    python -m strable_leaderboard.smoke models/<name> --data-dir /path/to/STRABLE-benchmark --device cuda --n-jobs 8
    python -m strable_leaderboard.smoke models/<name> --static     # card, pins and code only

It fails on any missing or invalid model card field (``strable_leaderboard.card``), on a pin
that the shared environment does not satisfy (``strable_leaderboard.check_requirements``) and
on code that ``FORBIDDEN`` flags as making its own folds or fixing its own seed: a
regular-expression hint, easy to get around; the Claude review is the real check. Then it loads
``Model`` from ``model.py``, encodes strings with Tf-Idf when the model needs numbers, and checks
the shapes and ranges of the predictions. Threads are capped to ``--n-jobs`` (default 1) as in the
runner. It does not measure leaderboard scores: the full run
happens on the cluster.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from pathlib import Path

from .card import card_problems, load_card
from .check_requirements import problems as pin_problems
from .data import load_table
from .fit_predict import limit_threads
from .metrics import checked_score
from .model import load_model_class
from .splits import SEED, outer_splits

SMOKE_TABLES = {
    "regression": "china-overseas-finance-inventory",  # 550 rows
    "binary": "animalandveterinary-event",  # 1,345 rows
    "multiclass": "drug-ndc",  # 583 rows, 4 classes
}
# A hint, not a guarantee: models never make their own folds or seeds (the runner gives the outer
# fold, the seed is self.seed, early stopping uses self.validation_split(y)), and the review checks it.
FORBIDDEN = {
    r"\b(train_test_split|(Repeated)?(Stratified)?(Group)?K[Ff]old|(Stratified)?ShuffleSplit|LeaveOneOut|"
    r"PredefinedSplit|TimeSeriesSplit|cross_val_(score|predict)|cross_validate)\b":
        "makes its own split or cross-validation; use self.validation_split(y)",
    r"\b(GridSearchCV|RandomizedSearchCV|HalvingGridSearchCV|HalvingRandomSearchCV|optuna|hyperopt)\b":
        "searches hyperparameters; only default configurations are accepted",
    r"\b(random_state|seed|random_seed)\s*=\s*-?\d+":
        "fixes its own seed; pass self.seed",
    r"\b(np\.random\.seed|random\.seed|torch\.manual_seed|torch\.cuda\.manual_seed(_all)?|set_seed)\s*\(\s*-?\d+":
        "fixes its own seed; pass self.seed",
}


def code_problems(folder: Path) -> list[str]:
    """Lines of model.py that make their own folds, search hyperparameters or fix a seed."""
    out = []
    for n, line in enumerate((Path(folder) / "model.py").read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        for pattern, why in FORBIDDEN.items():
            if re.search(pattern, code):
                out.append(f"model.py line {n} {why}: {line.strip()}")
    return out


def static_problems(folder: Path) -> list[str]:
    """Card, pins and code problems of a model folder, without running it."""
    import yaml

    folder = Path(folder)
    missing = [f for f in ("model.py", "model.yaml", "requirements.txt") if not (folder / f).exists()]
    if missing:
        return [f"missing {', '.join(missing)}"]
    out = [f"card: {p}" for p in card_problems(yaml.safe_load((folder / "model.yaml").read_text()))]
    out += [f"pins: {p}" for p in pin_problems(folder)]
    out += [f"protocol: {p}" for p in code_problems(folder)]
    return out


def check_one(model_cls, card: dict, table: str, data_dir=None, device: str = "cpu", n_jobs: int = 1) -> dict:
    X, y, task = load_table(table, data_dir=data_dir)
    train_idx, test_idx = outer_splits(y, task)[0]
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    if not model_cls.handles_strings:
        from .encoders import tfidf_encoder

        encoder = tfidf_encoder()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories")
            X_train = encoder.fit_transform(X_train, y_train)
            X_test = encoder.transform(X_test)

    model = model_cls(task=task, seed=SEED, device=device, n_jobs=n_jobs)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if task == "regression":
        out = {"pred": model.predict(X_test)}
    else:
        out = {"proba": model.predict_proba(X_test), "classes": model.classes_}
    predict_s = time.perf_counter() - t0
    score = checked_score(task, y_train, y_test, **out)
    return {"table": table, "task": task, "score": score, "fit_s": fit_s, "predict_s": predict_s,
            "n_train": len(train_idx), "n_test": len(test_idx)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="models/<name>")
    parser.add_argument("--data-dir", type=Path, default=None, help="local copy of the dataset (optional)")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    parser.add_argument("--n-jobs", type=int, default=1, help="CPU threads the model may use (the runner gives 32)")
    parser.add_argument("--static", action="store_true", help="check the card, the pins and the code, do not run")
    args = parser.parse_args(argv)

    found = static_problems(args.folder)
    for p in found:
        print(f"  FAIL  {p}")
    if found:
        return 1
    card = load_card(args.folder)
    model_cls = load_model_class(args.folder)
    if bool(card["handles_strings"]) != bool(model_cls.handles_strings):
        print("  FAIL  handles_strings differs between model.yaml and model.py")
        return 1
    limit_threads(args.n_jobs)
    print(f"{card['name']} {card['version']}  ({'end-to-end' if model_cls.handles_strings else 'numeric-only, Tf-Idf encoder'})")
    print("  ok    card, pins and code")
    if args.static:
        return 0
    if args.device == "auto":
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    failures = 0
    for task, table in SMOKE_TABLES.items():
        if task not in card["tasks"]:
            print(f"  skip  {task:<10} not in model.yaml tasks")
            continue
        try:
            r = check_one(model_cls, card, table, data_dir=args.data_dir, device=args.device, n_jobs=args.n_jobs)
            print(f"  ok    {task:<10} {table:<34} score {r['score']:.3f}  fit {r['fit_s']:.2f}s  predict {r['predict_s']:.2f}s")
        except Exception as exc:  # report every failure, then exit non-zero
            failures += 1
            print(f"  FAIL  {task:<10} {table:<34} {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
