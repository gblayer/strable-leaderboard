"""The model card, ``models/<name>/model.yaml``: every field, its allowed values, and the check.

``card_problems`` lists everything wrong with a card; ``load_card`` raises on the first list.
The smoke test fails on any problem, so a card is complete before a pull request is opened.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from .check_requirements import PIN
from .data import MANIFEST

REQUIRED = ["name", "version", "handles_strings", "tasks", "variant", "access", "hardware", "package", "checkpoint",
            "licence", "trained_on_strable"]
OPTIONAL = ["code", "paper", "anonymous"]
VARIANTS = {"default"}  # tuned and tuned-ensembled come later
ACCESS = {"open", "closed"}
HARDWARE = {"gpu", "cpu"}
TASKS = {"regression", "binary", "multiclass"}
# Values copied from models/_template/model.yaml and never replaced.
PLACEHOLDERS = {"My Model", "my-model==1.0.0", "my-org/my-model", "0123456789abcdef",
                "https://github.com/my-org/my-model", "https://arxiv.org/abs/0000.00000"}
COMMIT = re.compile(r"^[0-9a-f]{40}$")


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
    if "handles_strings" in card and not isinstance(card["handles_strings"], bool):
        out.append("handles_strings must be true or false")
    if "tasks" in card and (not isinstance(card["tasks"], list) or not card["tasks"] or not set(card["tasks"]) <= TASKS):
        out.append(f"tasks must be a non-empty list from {sorted(TASKS)}")
    for key, allowed in [("variant", VARIANTS), ("access", ACCESS), ("hardware", HARDWARE)]:
        if key in card and card[key] not in allowed:
            out.append(f"{key} must be one of {sorted(allowed)}")
    if "package" in card and not (isinstance(card["package"], str) and PIN.match(card["package"])):
        out.append("package must be one pin, name==version")
    if "checkpoint" in card and card["checkpoint"] is not None:
        ckpt = card["checkpoint"]
        if not isinstance(ckpt, dict) or set(ckpt) != {"repo", "revision"}:
            out.append("checkpoint must be null or {repo, revision}")
        else:
            if not isinstance(ckpt["repo"], str) or ckpt["repo"].count("/") != 1:
                out.append("checkpoint.repo must be a Hugging Face repository, owner/name")
            if not COMMIT.match(str(ckpt["revision"])):
                out.append("checkpoint.revision must be the full 40-character commit hash, never a branch or tag")
    if "trained_on_strable" in card:
        seen = card["trained_on_strable"]
        if not isinstance(seen, list):
            out.append("trained_on_strable must be a list of table names, or []")
        else:
            unknown = sorted(set(map(str, seen)) - set(json.loads(MANIFEST.read_text())["tables"]))
            if unknown:
                out.append(f"trained_on_strable names unknown tables {unknown[:5]}")
    for key in ["code", "paper"]:
        if key in card and not (isinstance(card[key], str) and card[key].startswith(("https://", "http://"))):
            out.append(f"{key} must be a URL")
    if "anonymous" in card and not isinstance(card["anonymous"], bool):
        out.append("anonymous must be true or false")
    return out


def load_card(folder: Path) -> dict:
    """Read model.yaml and raise ValueError listing every problem."""
    card = yaml.safe_load((Path(folder) / "model.yaml").read_text())
    found = card_problems(card)
    if found:
        raise ValueError("model.yaml: " + "; ".join(found))
    return card
