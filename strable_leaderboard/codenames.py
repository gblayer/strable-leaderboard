"""Codenames for anonymous models, such as ``jumping-carrot``.

Names are drawn with the ``secrets`` module, so they carry no information about the model
or the submitter. The mapping from codename to model lives only in the private repository.
"""

from __future__ import annotations

import secrets

ADJECTIVES = [
    "jumping", "sleepy", "brave", "fuzzy", "silent", "curious", "wobbly", "sunny", "clever", "dizzy",
    "gentle", "hungry", "lucky", "mellow", "nimble", "plucky", "quiet", "rusty", "shiny", "swift",
    "tiny", "velvet", "whistling", "zesty", "bouncy", "cosmic", "dapper", "fearless", "giddy", "humble",
]
NOUNS = [
    "carrot", "otter", "teapot", "pebble", "walrus", "biscuit", "comet", "lantern", "noodle", "pickle",
    "falcon", "muffin", "badger", "cactus", "kiwi", "marble", "pepper", "quokka", "radish", "sprout",
    "tulip", "umbrella", "waffle", "yak", "zephyr", "acorn", "banjo", "clover", "dumpling", "gecko",
]


def codename(taken=()) -> str:
    """A fresh ``adjective-noun`` name that is not in ``taken``."""
    taken = set(taken)
    if len(taken) >= len(ADJECTIVES) * len(NOUNS):
        raise RuntimeError("no codename left")
    while True:
        name = f"{secrets.choice(ADJECTIVES)}-{secrets.choice(NOUNS)}"
        if name not in taken:
            return name
