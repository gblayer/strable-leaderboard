"""Gather the rows written by strable_leaderboard.run into <results>/<model>.csv.

    python scripts/gather_results.py runs $STRABLE_RESULTS

One file per model folder, one row per (encoder, variant, table, fold), sorted. Existing rows
of the same (encoder, variant, table, fold) are replaced, rows imported from the paper included;
other rows are kept, so a re-run of a few tables updates only those.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strable_leaderboard.check_results import COLUMNS  # noqa: E402

KEY = ["encoder", "variant", "table", "fold"]


def gather(runs: Path, results: Path) -> list[Path]:
    written = []
    for model_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        rows = [json.loads(p.read_text()) for p in model_dir.rglob("*.json")]
        if not rows:
            continue
        new = pd.DataFrame(rows)
        out = results / f"{model_dir.name}.csv"
        if out.exists():
            old = pd.read_csv(out)
            keep = ~old.set_index(KEY).index.isin(new.set_index(KEY).index)
            new = pd.concat([old[keep], new], ignore_index=True)
        new = new.reindex(columns=COLUMNS + [c for c in new.columns if c not in COLUMNS])
        new.sort_values(KEY).to_csv(out, index=False)
        written.append(out)
        print(out, len(new), "rows")
    return written


if __name__ == "__main__":
    runs, results = Path(sys.argv[1]), Path(sys.argv[2])
    results.mkdir(exist_ok=True)
    gather(runs, results)
