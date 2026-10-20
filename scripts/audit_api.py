"""Audit a model before its run: does it learn from the rows it is given, or does it look the answers up?

    python scripts/audit_api.py models/<name> [--data-dir DIR] [--seed N]
    python scripts/audit_api.py models/<name> --dry honest|cheat     # a local fake vendor: nothing leaves

The STRABLE tables are public, and an API model receives the training rows and the test features, so
it could look the test labels up instead of predicting them, or adapt to the test batch. These are the
black-box probes of TabArena's audit (autogluon/tabarena, commit 860337d,
packages/tabarena/src/tabarena/tools/audit_system.py, Apache-2.0: see NOTICE), with its thresholds,
ported to the board's model interface: ``fit``, then ``predict`` or ``predict_proba``, on the numbers a
numeric-only model gets (the Tf-Idf encoder) or on the raw table. They run on outer fold 0 of the three
smoke tables and on a synthetic table drawn from a fresh seed, which has no public copy:

transduction  A test row's prediction must not depend on the other rows of the batch: the first half of
              the test rows alone, the rows in reverse order, and the first half next to as many decoy
              rows (numbers shifted by ten standard deviations, in the column's own type, and unseen
              strings), each compared with the full batch. The difference (mean absolute difference of the probabilities, or of the
              predictions over the target's standard deviation) PASSes within max(3 x noise, 0.001),
              noise being that of a second, identical run, WARNs within 3 times that, FAILs beyond.
labels        Classification: the training classes relabeled by a cyclic shift; regression: the target
              negated. An honest model follows the new labels. FAIL when its accuracy on the original
              labels beats that on the new ones by 0.1 and the majority rate by 0.05 (regression: its
              correlation with the original target is above 0.3 and above that with the negated one);
              WARN when it follows the new labels 0.1 worse than the real ones (regression: 0.2).
shuffled      The training labels shuffled across rows. An honest model loses its skill above chance (AUC
              0.5, the majority rate, R² 0): FAIL when it keeps more than 80% of it, WARN above 50%, both
              only when its score is more than 0.1 above chance.
jitter        Noise of 1% of a feature's standard deviation on the numeric test features. An exact-row
              lookup breaks, a model degrades smoothly. The relative change of the error (1 - AUC, log
              loss, RMSE) is compared with that of a local HistGradientBoosting on the same noise, on the
              test rows without an exact copy among the training rows when there are 30 or more: FAIL
              above 100% and 4 times the reference's change (at least 5%), WARN above 25% and twice;
              classification also needs the error to grow by more than 0.01.

Each fit gets its training rows as the board gives them, so a client that sends groups
(``StrableModel.inner_folds``) sends the board's inner folds on the smoke tables and the same seeded
8-fold assignment on the synthetic table. About eight calls per table; ``baseline`` (INFO) gives the error next to the reference's. The thresholds
catch blatant cheating, not subtle cheating: a provider who knows these probes can work around them, so
a passing audit is necessary, not sufficient. Exit code 1 when a probe FAILs.

The model runs in this process, as in the smoke test: for the maintainer's API models. Without ``--dry``
it reads the key file (strable_leaderboard.api) and calls the real endpoint. ``--dry honest`` sends the
calls to scripts/fake_api.py instead, ``--dry cheat`` to a fake vendor that knows the test labels of the
public tables, as one who looked them up would: the audit must FAIL it.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fake_api import FakeVendor, honest  # noqa: E402
from strable_leaderboard.api import AUTH, KEY, URL, child_env  # noqa: E402
from strable_leaderboard.card import load_card  # noqa: E402
from strable_leaderboard.data import load_table  # noqa: E402
from strable_leaderboard.fit_predict import limit_threads  # noqa: E402
from strable_leaderboard.model import load_model_class  # noqa: E402
from strable_leaderboard.smoke import SMOKE_TABLES  # noqa: E402
from strable_leaderboard.splits import SEED, outer_splits  # noqa: E402

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
UNSEEN = "__audit_unseen__"
JITTER_MIN_ROWS = 30


@dataclass
class Table:
    name: str
    task: str
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    public: bool = True

    @property
    def classes(self) -> list:
        return sorted(set(self.y_train.tolist()) | set(self.y_test.tolist()))


def smoke_table(table: str, data_dir) -> Table:
    X, y, task = load_table(table, data_dir=data_dir)
    train, test = outer_splits(y, task)[0]
    return Table(table, task, X.iloc[train], y[train], X.iloc[test], y[test])


def synthetic_table(task: str, seed: int, n: int = 2000) -> Table:
    """A fresh table with no public copy: scikit-learn's generators, one string column, a 75/25 split."""
    from sklearn.datasets import make_classification, make_friedman1

    rng = np.random.default_rng(seed)
    if task == "regression":
        X, y = make_friedman1(n_samples=n, n_features=8, noise=0.5, random_state=seed)
    else:
        X, y = make_classification(n_samples=n, n_features=10, n_informative=6, n_redundant=2, flip_y=0.03,
                                   n_classes=2 if task == "binary" else 3, class_sep=0.8, random_state=seed)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]).assign(s=rng.choice(list("abcd"), size=n))
    test = rng.random(n) < 0.25
    return Table(f"synthetic-{task}", task, frame[~test], y[~test], frame[test], y[test], public=False)


def encoded(t: Table) -> Table:
    """A numeric-only model's view of the table: the Tf-Idf encoder fitted on the training rows."""
    from strable_leaderboard.encoders import tfidf_encoder

    encoder = tfidf_encoder()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories")
        return Table(t.name, t.task, encoder.fit_transform(t.X_train, t.y_train), t.y_train,
                     encoder.transform(t.X_test), t.y_test, t.public)


def numeric_columns(X: pd.DataFrame) -> list:
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
            and X[c].nunique() > 2]


def decoys(X: pd.DataFrame, X_train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Rows far from the training rows: numbers shifted by 10 standard deviations, in the column's own type
    (integers whole and in range; TabArena casts floats to float64, which changes the other rows of the batch
    too), strings never seen."""
    rng = np.random.default_rng(seed)
    out = X.sample(frac=1.0, random_state=seed).copy()
    numeric = set(numeric_columns(X_train))
    for col in out.columns:
        if col in numeric:
            std, signs = float(X_train[col].std()) or 1.0, rng.choice([-1.0, 1.0], size=len(out))
            if pd.api.types.is_integer_dtype(out[col].dtype):
                bounds = np.iinfo(out[col].dtype)
                shifted = out[col].astype(float) + max(1.0, round(10 * std)) * (signs if bounds.min < 0 else 1.0)
                out[col] = shifted.clip(bounds.min, bounds.max).astype(out[col].dtype)
            else:
                out[col] = (out[col].astype(float) + 10 * std * signs).astype(out[col].dtype)
        elif not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = UNSEEN
    out.index = [f"decoy_{i}" for i in range(len(out))]
    return out


def jitter(X: pd.DataFrame, X_train: pd.DataFrame, seed: int, scale: float = 0.01) -> pd.DataFrame:
    rng, out = np.random.default_rng(seed), X.copy()
    for col in numeric_columns(X_train):
        out[col] = out[col].astype(float) + rng.normal(0.0, scale * (float(X_train[col].std()) or 1.0), size=len(out))
    return out


def error(t: Table, y, pred: np.ndarray) -> float:
    """TabArena's errors: RMSE, 1 - AUC (binary), log loss (multiclass)."""
    from sklearn.metrics import log_loss, roc_auc_score

    if t.task == "regression":
        return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - pred) ** 2)))
    if t.task == "binary":
        return float(1.0 - roc_auc_score(np.asarray(y) == t.classes[-1], pred[:, -1]))
    p = np.clip(pred, 1e-15, 1.0)
    return float(log_loss(y, p / p.sum(axis=1, keepdims=True), labels=t.classes))


def label_of(t: Table, pred: np.ndarray) -> np.ndarray:
    return np.asarray(t.classes)[pred.argmax(axis=1)]


class Probed:
    """The model under audit, called as the runner calls it: a fresh model per fit."""

    def __init__(self, cls, t: Table, n_jobs: int):
        self.cls, self.t, self.n_jobs, self.version = cls, t, n_jobs, None

    def fit(self, y):
        return self.cls(task=self.t.task, seed=SEED, device="cpu", n_jobs=self.n_jobs).fit(self.t.X_train, y)

    def predict(self, model, X) -> np.ndarray:
        """Values, or probabilities with one column per class of the table."""
        self.version = getattr(model, "api_version", None) or self.version
        if self.t.task == "regression":
            return np.asarray(model.predict(X), dtype=float).reshape(-1)
        proba, out = np.asarray(model.predict_proba(X), dtype=float), np.zeros((len(X), len(self.t.classes)))
        for j, c in enumerate(np.asarray(model.classes_).tolist()):
            out[:, self.t.classes.index(c)] = proba[:, j]
        return out


def reference(t: Table, X: pd.DataFrame) -> np.ndarray:
    out = honest(t.task, t.X_train, t.y_train, X, seed=0)
    if t.task == "regression":
        return np.asarray(out, dtype=float)
    full = np.zeros((len(X), len(t.classes)))
    for j, c in enumerate(out[0]):
        full[:, t.classes.index(c)] = out[1][:, j]
    return full


def against_noise(diff: float, tol: float) -> str:
    return PASS if diff <= tol else WARN if diff <= 3 * tol else FAIL


def degradation(t: Table, base: np.ndarray, jit: np.ndarray, rows: np.ndarray) -> dict | None:
    y = t.y_test[rows]
    if rows.sum() < JITTER_MIN_ROWS or (t.task == "binary" and len(set(y.tolist())) < 2):
        return None
    before, after = error(t, y, base[rows]), error(t, y, jit[rows])
    return {"rows": int(rows.sum()), "error": before, "jitter_error": after, "change": (after - before) / max(before, 1e-6)}


def jitter_verdict(t: Table, base, jit, ref_base, ref_jit) -> tuple[str, str]:
    train = set(map(tuple, t.X_train.astype(str).to_numpy()))
    copied = np.asarray([tuple(r) in train for r in t.X_test[t.X_train.columns].astype(str).to_numpy()])
    rows = ~copied if (~copied).sum() >= JITTER_MIN_ROWS else np.ones(len(copied), dtype=bool)
    own, ref = degradation(t, base, jit, rows), degradation(t, ref_base, ref_jit, rows)
    if own is None or ref is None:
        return INFO, "too few rows to judge"
    ok = t.task == "regression" or own["jitter_error"] - own["error"] > 0.01
    bound = max(ref["change"], 0.05)
    verdict = FAIL if own["change"] > 1.0 and own["change"] > 4 * bound and ok else \
        WARN if own["change"] > 0.25 and own["change"] > 2 * bound and ok else PASS
    change = lambda s: "from zero" if s["error"] < 1e-6 else f"{s['change']:+.1%}"  # noqa: E731
    where = "" if not copied.any() else " without a training copy" if rows.sum() < len(rows) else \
        " (all: too few without a training copy)"
    return verdict, (f"1% feature noise on the {own['rows']} test rows{where}: error {own['error']:.4f} -> "
                     f"{own['jitter_error']:.4f} ({change(own)}); reference {ref['error']:.4f} -> "
                     f"{ref['jitter_error']:.4f} ({change(ref)})")


def probe(cls, t: Table, seed: int, n_jobs: int = 1) -> list[tuple[str, str, str, str]]:
    """(verdict, probe, table, summary) of every probe on one table."""
    m, reg = Probed(cls, t, n_jobs), t.task == "regression"
    scale = (float(np.std(t.y_train)) or 1.0) if reg else 1.0
    diff = lambda a, b: float(np.mean(np.abs(a - b)) / scale)  # noqa: E731
    model = m.fit(t.y_train)
    base = m.predict(model, t.X_test)
    ref_base = reference(t, t.X_test)
    base_err, ref_err = error(t, t.y_test, base), error(t, t.y_test, ref_base)
    out = [(INFO, "baseline", t.name, f"error {base_err:.4f} vs local reference {ref_err:.4f}, {len(t.X_train)} train / "
                                      f"{len(t.X_test)} test rows" + (f", version {m.version}" if m.version else ""))]

    noise = diff(base, m.predict(m.fit(t.y_train), t.X_test))
    tol, half = max(3 * noise, 1e-3), np.arange(math.ceil(len(t.X_test) / 2))
    X_half = t.X_test.iloc[half]
    changes = {"half": diff(base[half], m.predict(model, X_half)),
               "reversed": diff(base, m.predict(model, t.X_test.iloc[::-1])[::-1]),
               "decoys": diff(base[half], m.predict(model, pd.concat([X_half, decoys(X_half, t.X_train, seed)]))[:len(half)])}
    verdicts = [against_noise(v, tol) for v in changes.values()]
    out.append((FAIL if FAIL in verdicts else WARN if WARN in verdicts else PASS, "transduction", t.name,
                f"repeat noise {noise:.2e}; change with the batch: " + ", ".join(f"{k} {v:.2e}" for k, v in changes.items())
                + f" (tolerance {tol:.2e})"))

    if reg:
        flip = m.predict(m.fit(-t.y_train), t.X_test)
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
        c_base, c_follow, c_orig = corr(base, t.y_test), corr(flip, -t.y_test), corr(flip, t.y_test)
        verdict = FAIL if c_orig > 0.3 and c_orig > c_follow else WARN if c_follow < c_base - 0.2 else PASS
        out.append((verdict, "labels", t.name, f"negated target: correlation with -y {c_follow:.3f} (real labels "
                                               f"{c_base:.3f}), with the original y {c_orig:.3f}"))
    else:
        train_classes = sorted(set(t.y_train.tolist()))
        shift = {c: train_classes[(i + 1) % len(train_classes)] for i, c in enumerate(train_classes)}
        perm = label_of(t, m.predict(m.fit(np.asarray([shift[v] for v in t.y_train])), t.X_test))
        acc_base = float(np.mean(label_of(t, base) == t.y_test))
        acc_follow = float(np.mean(perm == np.asarray([shift.get(v, v) for v in t.y_test])))
        acc_orig = float(np.mean(perm == t.y_test))
        majority = float(pd.Series(t.y_test).value_counts(normalize=True).iloc[0])
        verdict = FAIL if acc_orig > acc_follow + 0.1 and acc_orig > majority + 0.05 else \
            WARN if acc_follow < acc_base - 0.1 else PASS
        out.append((verdict, "labels", t.name, f"classes shifted {shift}: accuracy on the new labels {acc_follow:.3f} "
                                               f"(real labels {acc_base:.3f}), on the original ones {acc_orig:.3f} "
                                               f"(majority {majority:.3f})"))

    shuffled = m.predict(m.fit(np.random.default_rng(seed).permutation(t.y_train)), t.X_test)
    if t.task == "binary":
        what, chance, score, score_base = "AUC", 0.5, 1.0 - error(t, t.y_test, shuffled), 1.0 - base_err
    elif t.task == "multiclass":
        what, chance = "accuracy", float(pd.Series(t.y_test).value_counts(normalize=True).iloc[0])
        score, score_base = float(np.mean(label_of(t, shuffled) == t.y_test)), float(np.mean(label_of(t, base) == t.y_test))
    else:
        r2 = lambda p: float(1.0 - np.mean((t.y_test - p) ** 2) / np.var(t.y_test))  # noqa: E731
        what, chance, score, score_base = "R²", 0.0, r2(shuffled), r2(base)
    kept = (score - chance) / max(score_base - chance, 1e-9)
    above = score > chance + 0.1
    out.append((FAIL if above and kept > 0.8 else WARN if above and kept > 0.5 else PASS, "shuffled", t.name,
                f"{what} with shuffled training labels {score:.3f} (chance {chance:.3f}, real labels {score_base:.3f}): "
                f"{kept:.0%} of the skill above chance kept"))

    if numeric_columns(t.X_train):
        X_jit = jitter(t.X_test, t.X_train, seed)
        verdict, summary = jitter_verdict(t, base, m.predict(model, X_jit), ref_base, reference(t, X_jit))
        out.append((verdict, "jitter", t.name, summary))
    return out


def tables(card: dict, data_dir, seed: int) -> list[Table]:
    found = [smoke_table(table, data_dir) for task, table in SMOKE_TABLES.items() if task in card["tasks"]]
    found.append(synthetic_table("binary" if "binary" in card["tasks"] else card["tasks"][0], seed))
    return [t if card["end_to_end"] else encoded(t) for t in found]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="models/<name>")
    parser.add_argument("--data-dir", type=Path, help="local copy of the dataset (default: download the smoke tables)")
    parser.add_argument("--seed", type=int, help="default: a fresh seed, printed so that the audit can be repeated")
    parser.add_argument("--dry", choices=["honest", "cheat"], help="call a local fake vendor, not the endpoint")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    card, cls = load_card(args.folder), load_model_class(args.folder)
    if args.dry and card["hardware"] != "api":
        parser.error("--dry needs an API model (hardware: api)")
    seed = args.seed if args.seed is not None else int(time.time()) % 2**31
    limit_threads(args.n_jobs)
    print(f"audit_api: {card['name']} {card['version']}, seed {seed}" + (f", fake {args.dry} vendor" if args.dry else ""))
    data, results = tables(card, args.data_dir, seed), []
    with contextlib.ExitStack() as stack:
        if args.dry:
            vendor = stack.enter_context(FakeVendor(key="dry-run", auth=card["api"]["auth"], cheat=args.dry == "cheat"))
            os.environ.update({URL: vendor.url, KEY: "dry-run", AUTH: card["api"]["auth"]})
            for t in data:
                if t.public:  # what a vendor who looked the public tables up would know
                    vendor.leak(t.X_test, t.y_test)
        else:
            os.environ.update(child_env(args.folder, card))
        for t in data:
            print(f"--- {t.name} ({t.task}, {len(t.X_train)} train / {len(t.X_test)} test rows)", flush=True)
            try:
                found = probe(cls, t, seed, args.n_jobs)
            except Exception as exc:  # an outage, or a model that refuses the probes: a reviewer reads it
                found = [(FAIL, "probe", t.name, f"the model raised {type(exc).__name__}: {str(exc)[:300]}")]
            for verdict, name, table, summary in found:
                print(f"[{verdict:<4}] {name:<12} {table:<34} {summary}", flush=True)
            results += found
    counts = {v: sum(r[0] == v for r in results) for v in (FAIL, WARN, PASS, INFO)}
    print("audit_api summary: " + ", ".join(f"{n} {v}" for v, n in counts.items() if n))
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
