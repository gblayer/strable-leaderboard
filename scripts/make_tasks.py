"""List the (model, encoder, table, fold) tasks that have no result yet.

    python scripts/make_tasks.py models/tabpfn-2.5 [models/...] > tasks.txt

A numeric-only model gets the four encoders, an end-to-end model one line per table and fold.
Tasks with a row in $STRABLE_RUNS (default runs/) or in $STRABLE_RESULTS/<model>.csv are left
out, so the list is what the job array still has to do. Rows imported from the paper (hardware
"paper") do not count: the runner re-runs them on the reference hardware, and the new rows replace
them. ``tables`` and ``folds`` narrow the list (the dispatcher's test runs).
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strable_leaderboard.data import MANIFEST  # noqa: E402
from strable_leaderboard.encoders import ENCODERS  # noqa: E402
from strable_leaderboard.card import load_card  # noqa: E402

FOLDS = 3


def done(model: str, runs: Path, results: Path) -> set:
    keys = set()
    for p in (runs / model).rglob("*.json") if (runs / model).exists() else []:
        keys.add((p.parents[1].name, p.parent.name, int(p.stem)))
    csv = results / f"{model}.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df = df[df["hardware"].astype(str) != "paper"]
        keys |= set(zip(df["encoder"], df["table"], df["fold"]))
    return keys


def tasks(folders, runs=Path(os.environ.get("STRABLE_RUNS", ROOT / "runs")),
          results=Path(os.environ.get("STRABLE_RESULTS", "results")), tables=None, folds=range(FOLDS)):
    manifest = json.loads(MANIFEST.read_text())["tables"]
    out = []
    for folder in folders:
        card = load_card(Path(folder))
        encoders = ["none"] if card["end_to_end"] else list(ENCODERS)
        have = done(Path(folder).name, runs, results)
        for encoder in encoders:
            for table, meta in sorted(manifest.items()):
                if meta["task"] not in card["tasks"] or (tables is not None and table not in tables):
                    continue
                for fold in folds:
                    if (encoder, table, fold) not in have:
                        out.append(f"{folder} {encoder} {table} {fold}")
    return out


if __name__ == "__main__":
    print("\n".join(tasks(sys.argv[1:])))
