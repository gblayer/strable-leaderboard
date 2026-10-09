"""Check that a model's requirements.txt is satisfied by the shared environment.

    python -m strable_leaderboard.check_requirements models/<name> [models/...] [--base-env <folder>]

Every model runs in ONE shared environment, env/requirements.txt (the paper's versions), locked
from env/requirements.in by scripts/lock_env.sh. A model's requirements.txt may only list
``name==version`` pins, and each pin must equal the shared environment's version. A package the
shared environment lacks may be added to env/requirements.in in the same pull request; the
maintainer then relocks. ``--base-env`` gives the base branch's env/ folder: every pin there must
be unchanged, since a pull request may only add packages.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ENV = Path(__file__).resolve().parent.parent / "env"
ENV_REQUIREMENTS, ENV_INPUT = ENV / "requirements.txt", ENV / "requirements.in"
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[A-Za-z0-9,._-]+\])?==([A-Za-z0-9.+!_-]+)$")


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
        m = PIN.match(line)
        if not m:
            problems.append(f"{line!r}: only exact pins name==version are allowed (no ranges, URLs, paths or options)")
            continue
        pins[normalize(m.group(1))] = m.group(3)
    return pins, problems


def problems(folder: Path, env_files=(ENV_REQUIREMENTS, ENV_INPUT)) -> list[str]:
    """Everything that keeps the shared environment, as ``env_files`` pin it, from satisfying
    ``folder``'s requirements. The default also accepts a package just added to env/requirements.in;
    the runner passes only the lock."""
    folder = Path(folder)
    req = folder / "requirements.txt"
    if not req.exists():
        return ["requirements.txt is missing"]
    env = {}
    for path in env_files:
        env.update(read_pins(path)[0])
    pins, out = read_pins(req)
    if normalize("strable-leaderboard") in pins:
        out.append("do not list strable-leaderboard itself")
    for name, version in sorted(pins.items()):
        if name not in env:
            out.append(f"{name}=={version}: not in the shared environment; add this pin to env/requirements.in")
        elif env[name] != version:
            out.append(f"{name}=={version}: the shared environment has {name}=={env[name]}; pin that version")
    card_path = folder / "model.yaml"
    if card_path.exists():
        card = yaml.safe_load(card_path.read_text()) or {}
        m = PIN.match(str(card.get("package", "")))
        if m and pins.get(normalize(m.group(1))) != m.group(3):
            out.append(f"model.yaml package {card['package']} is not pinned the same way in requirements.txt")
    return out


def pin_changes(base: Path, head: Path = ENV) -> list[str]:
    """Pins of the base branch's env/ that ``head`` changes or removes: a pull request only adds."""
    files = ("requirements.in", "requirements.txt")
    old = {f: read_pins(base / f)[0] for f in files}
    new_in, found = read_pins(head / "requirements.in")
    new = {"requirements.in": new_in, "requirements.txt": read_pins(head / "requirements.txt")[0]}
    out = [f"env/{f}: {n}=={v} is changed or removed; existing pins never change in a pull request"
           for f in files for n, v in sorted(old[f].items()) if new[f].get(n) != v]
    locked = old["requirements.txt"]
    out += [f"env/requirements.in: {n}=={v}, but the shared environment has {n}=={locked[n]}"
            for n, v in sorted(new_in.items()) if n not in old["requirements.in"] and locked.get(n, v) != v]
    return out + [f"env/requirements.in: {p}" for p in found]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", type=Path)
    parser.add_argument("--base-env", type=Path, help="the base branch's env/ folder")
    args = parser.parse_args(argv)
    status = 0
    if args.base_env:
        found = pin_changes(args.base_env)
        print("env: " + ("only additions" if not found else "PROBLEMS"), *found, sep="\n   ")
        status |= bool(found)
    for folder in args.folders:
        found = problems(folder)
        print(f"{folder.name}: " + ("pins ok" if not found else "PROBLEMS"))
        for p in found:
            print("  ", p)
        status |= bool(found)
    return status


if __name__ == "__main__":
    sys.exit(main())
