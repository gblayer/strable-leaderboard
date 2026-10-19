"""The model card, ``models/<name>/model.yaml``: every field, its allowed values, and the check.

``card_problems`` lists everything wrong with a card; ``load_card`` raises on the first list.
An API model (``hardware: api``) also has an ``api`` block, nothing secret (``strable_leaderboard.api``).
The smoke test fails on any problem, so a card is complete before a pull request is opened.
``fetch_checkpoint`` downloads the card's weights, for the smoke test and scripts/prepare.py alike.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from .check_requirements import match_pin

REQUIRED = ["name", "version", "end_to_end", "tasks", "variant", "hardware", "package", "checkpoint", "licence",
            "pretraining_data"]
OPTIONAL = ["code", "paper", "api"]
VARIANTS = {"default"}  # the package's default configuration
HARDWARE = {"gpu", "cpu", "api"}  # api: runs on its provider's servers
API_AUTH = {"Bearer", "Api-Key"}  # the api block's auth: a public API, or a provider's raw endpoint
TASKS = {"regression", "binary", "multiclass"}
# What the model was pre-trained on, as its documentation says; "none" for a model without pre-training.
PRETRAINING = {"synthetic", "real", "mixed", "none", "not disclosed"}
# Values copied from models/_template/model.yaml and never replaced.
PLACEHOLDERS = {"My Model", "my-model==1.0.0", "my-org/my-model", "0123456789abcdef",
                "https://github.com/my-org/my-model", "https://arxiv.org/abs/0000.00000"}
COMMIT = re.compile(r"^[0-9a-f]{40}$")
ROOT = Path(__file__).resolve().parent.parent


def card_problems(card) -> list[str]:
    """Everything missing or invalid in a parsed model.yaml, as messages."""
    if not isinstance(card, dict):
        return ["model.yaml must be a mapping of fields"]
    out = [f"missing field {k}" for k in REQUIRED if k not in card]
    out += [f"unknown field {k} (allowed: {', '.join(REQUIRED + OPTIONAL)})" for k in card if k not in REQUIRED + OPTIONAL]
    for key, value in card.items():
        if isinstance(value, str) and value in PLACEHOLDERS:
            out.append(f"{key} still holds the template's placeholder {value!r}")
    for key in ["name", "version", "licence"]:
        if key in card and (not isinstance(card[key], str) or not card[key].strip()):
            out.append(f"{key} must be a non-empty string (quote versions: \"2.5\")")
    if "end_to_end" in card and not isinstance(card["end_to_end"], bool):
        out.append("end_to_end must be true or false")
    if "tasks" in card and (not isinstance(card["tasks"], list) or not card["tasks"] or not set(card["tasks"]) <= TASKS):
        out.append(f"tasks must be a non-empty list from {sorted(TASKS)}")
    for key, allowed in [("variant", VARIANTS), ("hardware", HARDWARE), ("pretraining_data", PRETRAINING)]:
        if key in card and (not isinstance(card[key], str) or card[key] not in allowed):
            out.append(f"{key} must be one of {sorted(allowed)}")
    if "package" in card and not (isinstance(card["package"], str) and match_pin(card["package"])):
        out.append("package must be one exact pin, as in requirements.txt: name==version, or "
                   "name @ git+https://...@<full 40-character commit hash>")
    if "checkpoint" in card and card["checkpoint"] is not None:
        ckpt = card["checkpoint"]
        if not isinstance(ckpt, dict) or set(ckpt) - {"files"} != {"repo", "revision"}:
            out.append("checkpoint must be null or {repo, revision}, with an optional files list")
        else:
            files = ckpt.get("files", ["*"])
            if not isinstance(files, list) or not files or not all(isinstance(f, str) and f for f in files):
                out.append("checkpoint.files must be a non-empty list of file names or patterns")
            if not isinstance(ckpt["repo"], str) or ckpt["repo"].count("/") != 1:
                out.append("checkpoint.repo must be a Hugging Face repository, owner/name")
            if not COMMIT.match(str(ckpt["revision"])):
                out.append("checkpoint.revision must be the full 40-character commit hash, never a branch or tag")
    for key in ["code", "paper"]:
        if key in card and not (isinstance(card[key], str) and card[key].startswith(("https://", "http://"))):
            out.append(f"{key} must be a URL")
    return out + api_problems(card)


def api_problems(card: dict) -> list[str]:
    """The api block: required with hardware: api, absent otherwise; auth and concurrency only, so
    that the endpoint and key never land in the card."""
    api = card.get("api")
    if card.get("hardware") != "api":
        return ["api is only for hardware: api"] if "api" in card else []
    if not isinstance(api, dict) or set(api) != {"auth", "concurrency"}:
        return ["hardware: api needs api: {auth: Bearer or Api-Key, concurrency: <tasks at once>}; the endpoint "
                "and key go in private/api_keys/<model folder>.env, never in the card"]
    out = [] if isinstance(api["auth"], str) and api["auth"] in API_AUTH else ["api.auth must be Bearer or Api-Key"]
    c = api["concurrency"]
    if not (isinstance(c, int) and not isinstance(c, bool) and c > 0):
        out.append("api.concurrency must be a positive whole number")
    return out


def load_card(folder: Path) -> dict:
    """Read model.yaml and raise ValueError listing every problem."""
    card = yaml.safe_load((Path(folder) / "model.yaml").read_text())
    found = card_problems(card)
    if found:
        raise ValueError("model.yaml: " + "; ".join(found))
    return card


def default_cache() -> Path:
    """``cache/`` of this clone, or $STRABLE_CACHE: gitignored, never on GitHub."""
    return Path(os.environ.get("STRABLE_CACHE", ROOT / "cache"))


def checkpoint_bytes(ckpt: dict) -> int:
    """The size of the files ``fetch_checkpoint`` downloads, as Hugging Face reports it."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    info = HfApi().model_info(ckpt["repo"], revision=ckpt["revision"], files_metadata=True)
    files = filter_repo_objects(info.siblings, allow_patterns=ckpt.get("files"), key=lambda f: f.rfilename)
    return sum(f.size or 0 for f in files)


def fetch_checkpoint(ckpt: dict, dest: Path) -> Path:
    """Download the checkpoint into ``dest``: ``dest/<file>`` is the Hugging Face repository at the
    pinned revision, only the files ``checkpoint.files`` names when the card lists them. Files
    already there are not downloaded again."""
    from huggingface_hub import snapshot_download

    snapshot_download(ckpt["repo"], revision=ckpt["revision"], allow_patterns=ckpt.get("files"), local_dir=dest)
    return Path(dest)
