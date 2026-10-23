"""Download and build what runs need, on a node with internet (the login node).

    python scripts/prepare.py [models/<name> ...] [--cache-dir cache]

Everything goes into the cache folder, ``cache/`` of this clone by default ($STRABLE_CACHE
overrides it). It is gitignored and never reaches GitHub:

- ``env/``: the ONE shared Python environment of every model, built from env/requirements.txt
  (the paper's versions) plus this package (-e .). Rebuilt only when env/requirements.txt changes,
  next to the live one, then swapped in: a task already running keeps the env it started with.
  The one it replaces stays as ``env.old-<time>/`` for ``OLD_ENV_DAYS`` days.
- ``data/``: the dataset at the pinned revision.
- ``weights/<encoder>/``: the language-model encoders' checkpoints.
- ``weights/<model>/``: each model's checkpoint at its pinned commit, only the card's
  ``checkpoint.files`` when it lists them (``strable_leaderboard.card.fetch_checkpoint``, which the
  smoke test uses too), refused above 20 GB unless the folder is in ``LARGE_CHECKPOINTS`` (this
  file, so only the maintainer can change it).
- ``sandbox/base.sif``: the Debian image scripts/sandbox.sh runs model code in, with gcc and the C
  library headers, pinned by digest; pulled when Singularity is installed, and again when ``IMAGE``
  changes (``sandbox/base.sif.source`` records what it was pulled from).

Each model's requirements.txt must be satisfied by the shared environment
(``strable_leaderboard.check_requirements``); a model that needs another version stops here.
uv is used when available, pip otherwise. PYTHONNOUSERSITE=1 keeps ~/.local out of the env.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strable_leaderboard.check_requirements import ENV_REQUIREMENTS, problems as pin_problems  # noqa: E402
from strable_leaderboard.card import checkpoint_bytes, default_cache, fetch_checkpoint, load_card  # noqa: E402

PYTHON = "3.12.12"  # the paper's interpreter
MAX_CHECKPOINT_BYTES = 20e9
OLD_ENV_DAYS = 3  # longer than the longest task (48 h): no job still reads an env replaced that long ago
LARGE_CHECKPOINTS: set[str] = set()  # model folders allowed a checkpoint above the cap
# Debian bookworm with gcc and the C library headers (buildpack-deps:bookworm, 351 MB compressed), for
# every model: Triton, which some packages run (Kumo Tabular's), compiles a small C launcher when its
# first kernel runs. The Python headers it also needs come from the bound interpreter, cache/python.
IMAGE = "docker://buildpack-deps:bookworm@sha256:b22578185ebfdbd9a4050f697638b9c9696b70616d9b01bddfa79145db2b6dfd"


def env_python(cache: Path) -> Path:
    return cache / "env" / "bin" / "python"


def build_env(cache: Path, force: bool = False) -> Path:
    """Build cache/env from env/requirements.txt, unless it is already built from the same file.

    The new env is built in cache/env.new, then two renames swap it in. A running task keeps the old
    one: the sandbox bound that directory, and a rename does not move it from under the mount. A task
    that starts after the swap gets the new env.
    """
    env, stamp = cache / "env", cache / "env" / ".requirements.sha256"
    digest = hashlib.sha256(ENV_REQUIREMENTS.read_bytes()).hexdigest()
    if not force and stamp.exists() and stamp.read_text().strip() == digest and env_python(cache).exists():
        print("shared env up to date:", env)
        return env
    for old in cache.glob("env.old-*"):
        if time.time() - old.stat().st_mtime > OLD_ENV_DAYS * 86400:
            shutil.rmtree(old)
    new = cache / "env.new"
    if new.exists():  # a build that stopped half-way
        shutil.rmtree(new)
    run_env = dict(os.environ, PYTHONNOUSERSITE="1")
    uv = shutil.which("uv")
    if uv:
        # The interpreter and uv's package cache live in the cache folder too, next to the env,
        # so nothing lands in the home folder and uv can hard-link instead of copying.
        run_env.update(UV_PYTHON_INSTALL_DIR=str(cache / "python"), UV_CACHE_DIR=str(cache / "uv"),
                       UV_PYTHON_PREFERENCE="only-managed", UV_LINK_MODE="hardlink")
        subprocess.run([uv, "python", "install", PYTHON], check=True, env=run_env)
        # Relocatable: its scripts still work once renamed to cache/env
        subprocess.run([uv, "venv", "--relocatable", "--python", PYTHON, str(new)], check=True, env=run_env)
        install = [uv, "pip", "install", "--python", str(new / "bin" / "python")]
    else:  # its console scripts keep the cache/env.new path; the runner only runs python -m
        subprocess.run([sys.executable, "-m", "venv", str(new)], check=True, env=run_env)
        install = [str(new / "bin" / "python"), "-m", "pip", "install", "-q"]
    # The lock is complete: installed as it is, without dependencies it leaves out (check_requirements.EXCLUDES)
    subprocess.run(install + ["--no-deps", "-r", str(ENV_REQUIREMENTS)], check=True, env=run_env)
    subprocess.run(install + ["--no-deps", "-e", str(ROOT)], check=True, env=run_env)
    (new / stamp.name).write_text(digest + "\n")
    if env.exists():
        old = cache / ("env.old-" + time.strftime("%Y%m%d-%H%M%S"))
        os.replace(env, old)
        os.utime(old)  # its age counts from the swap
    os.replace(new, env)
    print("shared env built:", env)
    return env


def pull_image(cache: Path) -> None:
    """The sandbox image: only a root file system; the env and the data are bound in read-only."""
    image, source = cache / "sandbox" / "base.sif", cache / "sandbox" / "base.sif.source"
    if not shutil.which("singularity") or (image.exists() and source.exists() and source.read_text().strip() == IMAGE):
        return
    tmp = cache / "sandbox" / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SINGULARITY_CACHEDIR=str(cache / "sandbox" / ".singularity"), SINGULARITY_TMPDIR=str(tmp))
    new = image.with_suffix(".sif.new")
    new.unlink(missing_ok=True)
    subprocess.run(["singularity", "pull", str(new), IMAGE], check=True, env=env)
    os.replace(new, image)  # a running job keeps reading the image it opened
    source.write_text(IMAGE + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-env", action="store_true", help="skip the shared environment")
    parser.add_argument("--no-download", action="store_true", help="skip the dataset and the checkpoints")
    parser.add_argument("--rebuild-env", action="store_true")
    args = parser.parse_args(argv)
    cache = (args.cache_dir or default_cache()).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    (cache / ".gitignore").write_text("*\n")  # ignored on every branch, whatever .gitignore says

    status = 0
    for folder in args.folders:
        found = pin_problems(folder, (ENV_REQUIREMENTS,))  # the lock only: what cache/env has
        if found:
            print(f"{folder.name}: requirements not satisfied by env/requirements.txt (relock with "
                  "scripts/lock_env.sh after adding a package):", *found, sep="\n  ")
            status = 1
    if status:
        return status
    if not args.no_env:
        build_env(cache, force=args.rebuild_env)
        pull_image(cache)
    if args.no_download:
        return 0

    from huggingface_hub import snapshot_download

    from strable_leaderboard import HF_DATASET, HF_REVISION
    from strable_leaderboard.encoders import ENCODERS

    snapshot_download(HF_DATASET, repo_type="dataset", revision=HF_REVISION, local_dir=cache / "data")
    for spec in ENCODERS.values():
        if spec["checkpoint"]:
            # original/ holds a second copy of the weights in Meta's format, which nothing reads
            snapshot_download(spec["checkpoint"], revision=spec["revision"], local_dir=cache / "weights" / spec["model"],
                              ignore_patterns=["original/*"])
    for folder in args.folders:
        ckpt = load_card(folder)["checkpoint"]
        if ckpt:
            size = checkpoint_bytes(ckpt)
            if size > MAX_CHECKPOINT_BYTES and folder.name not in LARGE_CHECKPOINTS:
                print(f"{folder.name}: the checkpoint is {size / 1e9:.1f} GB, above {MAX_CHECKPOINT_BYTES / 1e9:.0f} GB; "
                      "the maintainer can allow it in scripts/prepare.py, LARGE_CHECKPOINTS")
                status = 1
                continue
            fetch_checkpoint(ckpt, cache / "weights" / folder.name)
        print("prepared", folder.name)
    return status


if __name__ == "__main__":
    sys.exit(main())
