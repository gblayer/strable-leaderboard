"""Download what a run needs into the shared cache, on a node with internet.

    python scripts/prepare.py models/<name> [models/...] --cache-dir $STRABLE_CACHE

- the dataset at the pinned revision into <cache>/data/
- the encoders' checkpoints into <cache>/weights/<encoder>/
- each model's checkpoint into <cache>/weights/<model>/
- each model's environment into <cache>/envs/<model>/ (a venv with the pinned requirements)
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from strable_leaderboard import HF_DATASET, HF_REVISION  # noqa: E402
from strable_leaderboard.encoders import ENCODERS  # noqa: E402
from strable_leaderboard.smoke import load_card  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--no-envs", action="store_true")
    args = parser.parse_args(argv)
    from huggingface_hub import snapshot_download

    cache = args.cache_dir
    snapshot_download(HF_DATASET, repo_type="dataset", revision=HF_REVISION, local_dir=cache / "data")
    for encoder, spec in ENCODERS.items():
        if spec["checkpoint"]:
            snapshot_download(spec["checkpoint"], local_dir=cache / "weights" / encoder)
    for folder in args.folders:
        card = load_card(folder)
        if card["checkpoint"]:
            snapshot_download(card["checkpoint"]["repo"], revision=card["checkpoint"]["revision"],
                              local_dir=cache / "weights" / folder.name)
        if not args.no_envs:
            env = cache / "envs" / folder.name
            subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
            subprocess.run([str(env / "bin" / "pip"), "install", "-q", "-e", f"{ROOT}[smoke]", "-r",
                            str(folder / "requirements.txt"), "sentence-transformers"], check=True)
        print("prepared", folder.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
