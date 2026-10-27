"""Import the TILE paper's STRABLE pipelines into results/<model>.csv, next to the rows already there.

    python scripts/import_tile_results.py <folder of the TILE CSVs> --out $STRABLE_RESULTS

Exactly the pipelines listed in ``scripts/tile_selection.yaml`` are imported, by source CSV, method
token and the settings in ``where``; a listed pipeline without rows in its CSV is an error. The rows are
made as the paper's (``convert`` of scripts/import_paper_results.py): ``total_s`` only, the source's
``run_time``, with ``hardware`` and ``package`` set to ``paper``; ``hardware`` is instead the GPU of the
source row when its CSV records one (column ``gpu``), else the pipeline's own when the selection gives
it (the GPU its job logs show). For TILE, ``run_time`` counts the
language-model extraction of the whole table, the reduction and the target encoding, then fit and
predict, as the runner counts an encoder's extraction time. In each results file, rows of the same
(encoder, variant, table, fold) are replaced and all others kept.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gather_results import KEY  # noqa: E402
from import_paper_results import convert  # noqa: E402
from strable_leaderboard.check_results import COLUMNS  # noqa: E402
from strable_leaderboard.data import MANIFEST  # noqa: E402

SELECTION = ROOT / "scripts" / "tile_selection.yaml"


def select(frame: pd.DataFrame, methods: dict) -> pd.DataFrame:
    """The rows of the listed methods at their ``where`` settings, with ``run_time`` in seconds."""
    keep = pd.Series(False, index=frame.index)
    for method, p in methods.items():
        rows = frame["method"] == method
        for column, value in p.get("where", {}).items():
            rows &= frame[column].astype(str) == str(value)
        keep |= rows
    sel = frame[keep]
    # TILE's CSV writes times as one-element lists: "[143.8712]"
    return sel.assign(run_time=pd.to_numeric(sel["run_time"].astype(str).str.strip("[]")))


def hardware_of(df: pd.DataFrame, frames: dict, pipelines: dict) -> list[str]:
    """Each converted row's hardware: the GPU its source row records (column ``gpu``), else its
    pipeline's own ``hardware``, else ``paper``."""
    own = {(p["model"], p["encoder"], p["variant"]): p.get("hardware", "paper")
           for methods in pipelines.values() for p in methods.values()}
    gpus = {}
    for csv, methods in pipelines.items():
        if "gpu" in frames[csv]:
            for method, p in methods.items():
                for r in frames[csv][frames[csv]["method"] == method].itertuples():
                    gpus[(p["model"], p["encoder"], p["variant"], r.data_name, int(r.fold_index))] = r.gpu
    keys = zip(df["model"], df["encoder"], df["variant"], df["table"], df["fold"])
    return [gpus.get(key, own[key[:3]]) for key in keys]


def merge(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """A results file's rows: ``old`` without the keys ``new`` has, then ``new``, sorted."""
    if old is not None:
        keep = ~old.set_index(KEY).index.isin(new.set_index(KEY).index)
        new = pd.concat([old[keep], new], ignore_index=True)
    return new.reindex(columns=COLUMNS + [c for c in new.columns if c not in COLUMNS]).sort_values(KEY)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("compiled", type=Path, help="the folder of the TILE authors' compiled CSVs")
    parser.add_argument("--out", type=Path, default=os.environ.get("STRABLE_RESULTS", "results"))
    args = parser.parse_args(argv)
    tables = json.loads(MANIFEST.read_text())["tables"]
    selection = yaml.safe_load(SELECTION.read_text())
    pipelines = selection["pipelines"]
    frames = {csv: select(pd.read_csv(args.compiled / csv), methods) for csv, methods in pipelines.items()}
    df, problems = convert(frames, tables, pipelines)
    if problems:
        print("\n".join(problems))
        return 1
    dates = {(p["model"], p["encoder"], p["variant"]): str(p.get("date", selection["date"]))
             for methods in pipelines.values() for p in methods.values()}
    df["date"] = [dates[key] for key in zip(df["model"], df["encoder"], df["variant"])]
    df["hardware"] = hardware_of(df, frames, pipelines)
    args.out.mkdir(exist_ok=True)
    for model, rows in df.groupby("model"):
        out = args.out / f"{model}.csv"
        old = pd.read_csv(out, float_precision="round_trip") if out.exists() else None
        merge(old, rows).to_csv(out, index=False)
        for (encoder, variant), g in rows.groupby(["encoder", "variant"]):
            print(out, f"{encoder}+{model}", variant, len(g), "rows")
    print(len(df.groupby(["model", "encoder", "variant"])), "pipelines imported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
