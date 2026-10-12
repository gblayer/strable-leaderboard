"""Download and build what runs need, on a node with internet (the login node).

    python scripts/prepare.py [models/<name> ...] [--cache-dir cache]

Everything goes into the cache folder, ``cache/`` of this clone by default ($STRABLE_CACHE
overrides it). It is gitignored and never reaches GitHub:

- ``env/``: the ONE shared Python environment of every model, built from env/requirements.txt
  (the paper's versions) plus this package (-e .). Rebuilt only when env/requirements.txt changes.
- ``data/``: the dataset at the pinned revision.
- ``weights/<encoder>/``: the language-model encoders' checkpoints.
- ``weights/<model>/``: each model's checkpoint at its pinned commit, refused above 20 GB unless
  the folder is in ``LARGE_CHECKPOINTS`` (this file, so only the maintainer can change it).
- ``sandbox/base.sif``: the small Debian image scripts/sandbox.sh runs model code in, pinned by
  digest, pulled once when Singularity is installed.

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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strable_leaderboard.check_requirements import ENV_REQUIREMENTS, problems as pin_problems  # noqa: E402
from strable_leaderboard.card import load_card  # noqa: E402

PYTHON = "3.12.12"  # the paper's interpreter
MAX_CHECKPOINT_BYTES = 20e9
LARGE_CHECKPOINTS: set[str] = set()  # model folders allowed a checkpoint above the cap
IMAGE = "docker://debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587"


def default_cache() -> Path:
    return Path(os.environ.get("STRABLE_CACHE", ROOT / "cache"))


def env_python(cache: Path) -> Path:
    return cache / "env" / "bin" / "python"


def build_env(cache: Path, force: bool = False) -> Path:
    """Build cache/env from env/requirements.txt, unless it is already built from the same file."""
    env, stamp = cache / "env", cache / "env" / ".requirements.sha256"
    digest = hashlib.sha256(ENV_REQUIREMENTS.read_bytes()).hexdigest()
    if not force and stamp.exists() and stamp.read_text().strip() == digest and env_python(cache).exists():
        print("shared env up to date:", env)
        return env
    if env.exists():
        shutil.rmtree(env)
    run_env = dict(os.environ, PYTHONNOUSERSITE="1")
    uv = shutil.which("uv")
    if uv:
        # The interpreter and uv's package cache live in the cache folder too, next to the env,
        # so nothing lands in the home folder and uv can hard-link instead of copying.
        run_env.update(UV_PYTHON_INSTALL_DIR=str(cache / "python"), UV_CACHE_DIR=str(cache / "uv"),
                       UV_PYTHON_PREFERENCE="only-managed", UV_LINK_MODE="hardlink")
        subprocess.run([uv, "python", "install", PYTHON], check=True, env=run_env)
        subprocess.run([uv, "venv", "--python", PYTHON, str(env)], check=True, env=run_env)
        install = [uv, "pip", "install", "--python", str(env_python(cache))]
    else:
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True, env=run_env)
        install = [str(env_python(cache)), "-m", "pip", "install", "-q"]
    subprocess.run(install + ["-r", str(ENV_REQUIREMENTS)], check=True, env=run_env)
    subprocess.run(install + ["--no-deps", "-e", str(ROOT)], check=True, env=run_env)
    stamp.write_text(digest + "\n")
    print("shared env built:", env)
    return env


def checkpoint_bytes(repo: str, revision: str) -> int:
    """The size of every file of the checkpoint at that revision, as Hugging Face reports it."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    return sum(f.size or 0 for f in info.siblings)


def pull_image(cache: Path) -> None:
    """The sandbox image: only a root file system; the env and the data are bound in read-only."""
    image = cache / "sandbox" / "base.sif"
    if image.exists() or not shutil.which("singularity"):
        return
    image.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SINGULARITY_CACHEDIR=str(cache / "sandbox" / ".singularity"))
    subprocess.run(["singularity", "pull", str(image), IMAGE], check=True, env=env)


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
            size = checkpoint_bytes(ckpt["repo"], ckpt["revision"])
            if size > MAX_CHECKPOINT_BYTES and folder.name not in LARGE_CHECKPOINTS:
                print(f"{folder.name}: the checkpoint is {size / 1e9:.1f} GB, above {MAX_CHECKPOINT_BYTES / 1e9:.0f} GB; "
                      "the maintainer can allow it in scripts/prepare.py, LARGE_CHECKPOINTS")
                status = 1
                continue
            snapshot_download(ckpt["repo"], revision=ckpt["revision"], local_dir=cache / "weights" / folder.name)
        print("prepared", folder.name)
    return status


if __name__ == "__main__":
    sys.exit(main())
