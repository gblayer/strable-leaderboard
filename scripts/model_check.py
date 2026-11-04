"""The weekly check for new models (.github/workflows/model-check.yml): new models among the top of a public
leaderboard of tabular models (SPACE, below).

    python scripts/model_check.py find --issues titles.txt     # JSON list of the new top models, at most --max
    python scripts/model_check.py snapshot                     # every model on the leaderboard now, one per line
    python scripts/model_check.py title model.json             # the issue title of a model
    python scripts/model_check.py issue model.json answer.json  # the issue text, from the research pass

A model is a leaderboard row's name without its variant ("TabDPT (tuned)" and "TabDPT (default)" are TabDPT,
"TabFM+" is TabFM). It is new when it is not on the board (the names of models/*/model.yaml and of
site/results.json), not in .github/models_seen.txt and not the title of an issue already opened (--issues: one
title per line). Only a new model whose best row is in the top --top rows by Elo is listed: one below it is
looked at again each week. The leaderboard is the largest table of the Space's data/ folder (every entrant).
Fails when its files cannot be read.
"""

import argparse
import csv
import io
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = "https://huggingface.co/spaces/TabArena/leaderboard"
FOLDERS = "https://huggingface.co/api/spaces/TabArena/leaderboard/tree/main/data"
TABLE = SPACE + "/resolve/main/data/{}/imputation_no/splits_all/tasks_all/datasets_all/website_leaderboard.csv"
KNOWN = ROOT / ".github" / "models_seen.txt"
LINK = re.compile(r"^\[(.+)\]\((\S+)\)$")
FIELDS = {"type": "TypeName", "licence": "License", "commercial": "Commercial", "organisation": "Organization",
          "hardware": "Hardware", "verified": "Verified", "tags": "Tags"}
TITLE = "New model: "
MIN_MODELS = 20  # fewer models read means the leaderboard's files changed shape


def family(name: str) -> str:
    """The model of a leaderboard name: without its variant in brackets and a trailing +."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", name.strip()).rstrip("+").strip()


def key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", family(name).lower())


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as reply:
        return reply.read()


def leaderboard_rows() -> list[dict]:
    """The rows of the full leaderboard: the largest table of the Space's data/ folder (every entrant)."""
    folders = [f["path"].split("/")[-1] for f in json.loads(fetch(FOLDERS)) if f["type"] == "directory"]
    tables = []
    for folder in folders:
        try:
            tables.append(list(csv.DictReader(io.StringIO(fetch(TABLE.format(folder)).decode()))))
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
    return max(tables, key=len, default=[])


def elo(row: dict) -> float:
    column = next((c for c in row if c.startswith("Elo") and "CI" not in c), None)
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return float("-inf")


def models(rows: list[dict]) -> dict[str, dict]:
    """One entry per model, keyed by key(): its name, link, variants, the rank and Elo of its best row (1 is the
    top of the leaderboard) and the leaderboard's columns of that row."""
    out = {}
    for rank, row in enumerate(sorted(rows, key=elo, reverse=True), 1):
        match = LINK.match(row.get("Model", "").strip())
        name, url = match.groups() if match else (row.get("Model", "").strip(), "")
        if not name:
            continue
        entry = out.setdefault(key(name), {"name": family(name), "url": url, "variants": [], "rank": rank,
                                           "elo": elo(row), "of": len(rows),
                                           **{k: row.get(v, "") for k, v in FIELDS.items()}})
        if name not in entry["variants"]:
            entry["variants"].append(name)
    return out


def board_names() -> set[str]:
    """The keys of the models on the board: its cards (not the test copies) and the learners of its page."""
    names = set()
    for card in (ROOT / "models").glob("*/model.yaml"):
        if card.parent.name.startswith(("test-", "_")):
            continue
        match = re.search(r"^name:\s*(.+?)\s*(#.*)?$", card.read_text(), re.M)
        if match:
            names.add(match.group(1).strip("\"'"))
    site = ROOT / "site" / "results.json"
    if site.exists():
        names |= {p["lrn"] for p in json.loads(site.read_text())["pipelines"]}
    return {key(n) for n in names}


def known_names(path: Path = KNOWN) -> set[str]:
    lines = path.read_text().splitlines() if path.exists() else []
    return {key(line) for line in lines if line.strip() and not line.startswith("#")}


def issue_names(titles: list[str]) -> set[str]:
    return {key(t[len(TITLE):]) for t in titles if t.startswith(TITLE)}


def new_models(found: dict[str, dict], skip: set[str], limit: int, top: int) -> list[dict]:
    """The models not in ``skip`` whose best row is in the top ``top`` rows, best first."""
    new = sorted((m for k, m in found.items() if k not in skip and m["rank"] <= top), key=lambda m: m["rank"])
    return [dict(m, i=i) for i, m in enumerate(new[:limit])]


def title(model: dict) -> str:
    return TITLE + re.sub(r"\s+", " ", model["name"])[:80]


def cell(text) -> str:
    text = ", ".join(map(str, text)) if isinstance(text, list) else str(text or "not found")
    return re.sub(r"\s+", " ", text).replace("|", "\\|")[:400]


HOW = {"end_to_end": "end-to-end: it reads the text columns itself (one row on the board)",
       "encoders": "with the board's four string encoders: numbers only (one row per encoder)",
       "unclear": "unclear"}


def issue(model: dict, answer: dict) -> str:
    """The issue text: the leaderboard's line for the model, then the research pass's answer, with its sources."""
    lines = [
        f"**{cell(model['name'])}** is a new model among the top of a public leaderboard of tabular models "
        f"(rank {model['rank']} of {model['of']} rows by Elo, {model['elo']:.0f}), and not on this board. "
        f"Listed as: {cell(model['type'])}, by {cell(model['organisation'])}, "
        f"licence {cell(model['licence'])}, variants {cell(model['variants'])}. Link: {cell(model['url'])}",
        "",
        "| Question | Answer |",
        "|---|---|",
        f"| Can the board run it? | {cell(answer['runnable'])} |",
        f"| How | {HOW.get(answer['mode'], cell(answer['mode']))} |",
        f"| What it is | {cell(answer['summary'])} |",
        f"| Tasks | {cell(answer['tasks'])} |",
        f"| Weights | {cell(answer['weights'])} |",
        f"| Licence | {cell(answer['licence'])} |",
        f"| Package | {cell(answer['install'])} |",
        f"| Text columns | {cell(answer['text_columns'])} |",
        f"| Hardware | {cell(answer['hardware'])} |",
    ]
    if answer.get("blockers"):
        lines += ["", "**What stops it:**"] + [f"- {cell(b)}" for b in answer["blockers"]]
    lines += ["", "**Sources:**"] + ([f"- \"{cell(e['quote'])}\" ({cell(e['url'])})" for e in answer.get("evidence", [])]
                                     or ["- none could be fetched"])
    lines += ["", "Written by the weekly check for new models (`.github/workflows/model-check.yml`) from the model's public "
              "documents: check them before acting. To add the model, open a pull request with `models/<name>/` "
              "(`docs/submitting.md`). To dismiss it, close this issue: it is not opened again."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    find = sub.add_parser("find")
    find.add_argument("--issues", type=Path, help="titles of the issues already opened, one per line")
    find.add_argument("--max", type=int, default=5)
    find.add_argument("--top", type=int, default=10, help="only models with a row in the top TOP rows by Elo")
    sub.add_parser("snapshot")
    sub.add_parser("title").add_argument("model", type=Path)
    text = sub.add_parser("issue")
    text.add_argument("model", type=Path)
    text.add_argument("answer", type=Path)
    args = parser.parse_args()

    if args.command in ("find", "snapshot"):
        found = models(leaderboard_rows())
        if len(found) < MIN_MODELS:
            sys.exit(f"read {len(found)} models from {SPACE}: its files changed shape")
        if args.command == "snapshot":
            print("\n".join(sorted(m["name"] for m in found.values())))
            return
        titles = args.issues.read_text().splitlines() if args.issues else []
        skip = board_names() | known_names() | issue_names(titles)
        print(json.dumps(new_models(found, skip, args.max, args.top)))
    elif args.command == "title":
        print(title(json.loads(args.model.read_text())))
    else:
        answer = json.loads(args.answer.read_text().strip() or "null")
        if not isinstance(answer, dict) or "runnable" not in answer:
            sys.exit(f"no answer from the research pass in {args.answer}")
        sys.stdout.write(issue(json.loads(args.model.read_text()), answer))


if __name__ == "__main__":
    main()
