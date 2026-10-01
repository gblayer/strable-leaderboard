"""List the (model, encoder, table, fold) tasks that have no result yet.

    python scripts/make_tasks.py models/tabpfn-2.5 [models/...] > tasks.txt

A numeric-only model gets the three encoders, an end-to-end model one line per table and fold.
Tasks with a row in runs/ or in $STRABLE_RESULTS/<model>.csv are left out, so the list is what
the job array still has to do.
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
from strable_leaderboard.smoke import load_card  # noqa: E402

FOLDS = 3


def done(model: str, runs: Path, results: Path) -> set:
    keys = set()
    for p in (runs / model).rglob("*.json") if (runs / model).exists() else []:
        keys.add((p.parents[1].name, p.parent.name, int(p.stem)))
    csv = results / f"{model}.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        keys |= set(zip(df["encoder"], df["table"], df["fold"]))
    return keys


def tasks(folders, runs=ROOT / "runs", results=Path(os.environ.get("STRABLE_RESULTS", "results"))):
    tables = json.loads(MANIFEST.read_text())["tables"]
    out = []
    for folder in folders:
        card = load_card(Path(folder))
        encoders = ["none"] if card["handles_strings"] else list(ENCODERS)
        have = done(Path(folder).name, runs, results)
        for encoder in encoders:
            for table, meta in sorted(tables.items()):
                if meta["task"] not in card["tasks"]:
                    continue
                for fold in range(FOLDS):
                    if (encoder, table, fold) not in have:
                        out.append(f"{folder} {encoder} {table} {fold}")
    return out


if __name__ == "__main__":
    print("\n".join(tasks(sys.argv[1:])))
