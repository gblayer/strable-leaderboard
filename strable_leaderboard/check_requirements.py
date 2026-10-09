"""Check that a model's requirements.txt is satisfied by the shared environment.

    python -m strable_leaderboard.check_requirements models/<name> [models/...] [--installed]

Every model runs in ONE shared environment, built from env/requirements.txt (the paper's
versions). A model's requirements.txt may only list ``name==version`` pins, and each pin must
equal the shared environment's version. ``--installed`` also compares with the packages
installed in the running interpreter. A model that needs another version proposes the change to
env/requirements.txt in its pull request; that never merges automatically.
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

import yaml

ENV_REQUIREMENTS = Path(__file__).resolve().parent.parent / "env" / "requirements.txt"
LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[A-Za-z0-9,._-]+\])?==([A-Za-z0-9.+!_-]+)$")


def normalize(name: str) -> str:
    """PEP 503: Scikit_Learn and scikit-learn are the same package."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path) -> tuple[dict[str, str], list[str]]:
    """``({name: version}, problems)`` from a requirements file: only ``name==version`` lines."""
    pins, problems = {}, []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = LINE.match(line)
        if not m:
            problems.append(f"{line!r}: only exact pins name==version are allowed (no ranges, URLs, paths or options)")
            continue
        pins[normalize(m.group(1))] = m.group(3)
    return pins, problems


def problems(folder: Path, env_file: Path = ENV_REQUIREMENTS, installed: bool = False) -> list[str]:
    """Everything that keeps the shared environment from satisfying ``folder``'s requirements."""
    folder = Path(folder)
    req = folder / "requirements.txt"
    if not req.exists():
        return ["requirements.txt is missing"]
    env, _ = read_pins(env_file)
    pins, out = read_pins(req)
    if normalize("strable-leaderboard") in pins:
        out.append("do not list strable-leaderboard itself")
    for name, version in sorted(pins.items()):
        if name not in env:
            out.append(f"{name}=={version}: not in the shared environment (env/requirements.txt); "
                       "propose it there, the maintainer decides")
        elif env[name] != version:
            out.append(f"{name}=={version}: the shared environment has {name}=={env[name]}; "
                       f"pin {env[name]} or propose the change in env/requirements.txt")
        elif installed:
            try:
                have = metadata.version(name)
            except metadata.PackageNotFoundError:
                have = None
            if have != version:
                out.append(f"{name}=={version}: installed here is {have}; rebuild the env with scripts/prepare.py")
    card_path = folder / "model.yaml"
    if card_path.exists():
        card = yaml.safe_load(card_path.read_text()) or {}
        m = LINE.match(str(card.get("package", "")))
        if m and pins.get(normalize(m.group(1))) != m.group(3):
            out.append(f"model.yaml package {card['package']} is not pinned the same way in requirements.txt")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--env", type=Path, default=ENV_REQUIREMENTS, help="the shared environment's pins")
    parser.add_argument("--installed", action="store_true", help="also compare with the installed packages")
    args = parser.parse_args(argv)
    status = 0
    for folder in args.folders:
        found = problems(folder, args.env, args.installed)
        print(f"{folder.name}: " + ("pins ok" if not found else "PROBLEMS"))
        for p in found:
            print("  ", p)
        status |= bool(found)
    return status


if __name__ == "__main__":
    sys.exit(main())
