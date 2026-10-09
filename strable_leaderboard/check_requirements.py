"""Check that a model's requirements.txt is satisfied by the shared environment.

    python -m strable_leaderboard.check_requirements models/<name> [models/...] [--base-env <folder>]
    python -m strable_leaderboard.check_requirements --report <requirements.in>   # what a new list would change
    python -m strable_leaderboard.check_requirements --lock                       # scripts/lock_env.sh

Every model runs in ONE shared environment, env/requirements.txt (the paper's versions), locked
from env/requirements.in by scripts/lock_env.sh. A model's requirements.txt may only list
``name==version`` pins, and each pin must equal the shared environment's version. A package the
shared environment lacks may be added to env/requirements.in in the same pull request; the
maintainer then relocks. ``--base-env`` gives the base branch's env/ folder: every pin there must
be unchanged, since a pull request may only add packages.

``--lock`` regenerates env/requirements.txt with uv, the paper's full freeze (env/paper-freeze.txt)
as constraints. ``--report`` resolves another requirements.in the same way, never building a
package, and prints for the maintainer the packages it adds and every existing version it changes;
when that fails, it resolves again without the freeze and with the current pins of
env/requirements.in loosened to bare names, keeping the current versions where uv can, to show
what would have to move.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent / "env"
ENV_REQUIREMENTS, ENV_INPUT, FREEZE = ENV / "requirements.txt", ENV / "requirements.in", ENV / "paper-freeze.txt"
UV_COMPILE = ["uv", "pip", "compile", "--python-version", "3.12", "--python-platform", "x86_64-manylinux_2_28",
              "--no-header", "--no-annotate", "--quiet"]
LOCK_HEADER = """# The shared environment, every package pinned to the version of the STRABLE paper environment
# (salts_env, 2026-03-23, Python 3.12.12). Generated from env/requirements.in; do not edit by hand.
# scripts/prepare.py installs it into cache/env together with this package (-e .).
"""
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
        import yaml

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


def compile_lock(requirements_in: Path, out: Path, freeze: bool = True, *extra: str) -> subprocess.CompletedProcess:
    """uv's resolution of ``requirements_in`` into ``out``; uv keeps the versions ``out`` already pins where it can."""
    constraints = ["-c", str(FREEZE)] if freeze else []
    return subprocess.run(UV_COMPILE + [str(requirements_in), *constraints, *extra, "-o", str(out)],
                          capture_output=True, text=True)


def lock() -> None:
    """Regenerate env/requirements.txt from env/requirements.in, the paper's freeze as constraints."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "requirements.txt"
        r = compile_lock(ENV_INPUT, out)
        if r.returncode:
            sys.exit(r.stderr)
        ENV_REQUIREMENTS.write_text(LOCK_HEADER + out.read_text())


def lock_changes(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """Markdown lines: the packages ``new`` adds, then every version of ``old`` it changes or drops."""
    from packaging.version import Version

    added = [f"`{n}=={v}`" for n, v in sorted(new.items()) if n not in old]
    out = ["- Added: " + (", ".join(added) if added else "none")]
    for n, v in sorted(old.items()):
        if new.get(n) == v:
            continue
        down = n in new and Version(new[n]) < Version(v)
        out.append(f"- `{n}` {v} → {new.get(n, 'removed')}" + (" **downgrade**" if down else ""))
    if len(out) == 1:
        out.append("- No existing package changes version.")
    return out


def report(requirements_in: Path, base: Path = ENV) -> str:
    """For the maintainer: what locking ``requirements_in`` as --lock does would add to and change in
    ``base``'s lock. Only ``name==version`` lines are read, and no package is ever built."""
    matches = [m for raw in Path(requirements_in).read_text().splitlines()
               if (m := PIN.match(raw.split("#", 1)[0].strip()))]
    lines = [m.group(0) for m in matches]
    current = read_pins(base / "requirements.in")[0]
    loose = [m.group(1) + (m.group(2) or "") if normalize(m.group(1)) in current else m.group(0) for m in matches]
    old = read_pins(base / "requirements.txt")[0]
    head = ["**Shared environment**: this pull request's env/requirements.in, locked as scripts/lock_env.sh does."]
    with tempfile.TemporaryDirectory() as tmp:
        pins, out = Path(tmp) / "requirements.in", Path(tmp) / "requirements.txt"
        pins.write_text("\n".join(lines) + "\n")
        r = compile_lock(pins, out, True, "--no-build")
        if r.returncode:
            head = [f"**Shared environment: the paper's freeze cannot hold**, scripts/lock_env.sh would fail:",
                    "```", r.stderr.strip()[-1200:], "```",
                    "Resolved without the freeze, keeping the current versions where possible:"]
            pins.write_text("\n".join(loose) + "\n")
            out.write_text((base / "requirements.txt").read_text())  # uv prefers the versions out pins
            r = compile_lock(pins, out, False, "--no-build")
            if r.returncode:
                return "\n".join(head[:-1] + ["It does not resolve without the freeze either:", "```",
                                                r.stderr.strip()[-1200:], "```"])
        return "\n".join(head + lock_changes(old, read_pins(out)[0]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", type=Path)
    parser.add_argument("--base-env", type=Path, help="the base branch's env/ folder")
    parser.add_argument("--report", type=Path, help="a requirements.in to compare with the current lock")
    parser.add_argument("--lock", action="store_true", help="regenerate env/requirements.txt")
    args = parser.parse_args(argv)
    if args.lock:
        return lock()
    if args.report:
        print(report(args.report))
        return 0
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
