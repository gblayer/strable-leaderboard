"""Codenames for anonymous models: an action verb and a fruit or vegetable, ``jumping-carrot``.

Names are drawn with the ``secrets`` module, so they say nothing about the model or the
submitter. The mapping from codename to model lives only in the private repository.
"""

from __future__ import annotations

import secrets

VERBS = [
    "jumping", "dancing", "rolling", "spinning", "hopping", "racing", "diving", "gliding", "bouncing", "drifting",
    "sprinting", "soaring", "swimming", "climbing", "sliding", "skipping", "floating", "twirling", "marching", "leaping",
    "sailing", "surfing", "galloping", "waddling", "wandering", "juggling", "whistling", "tumbling", "zooming", "dashing",
]
PRODUCE = [
    "carrot", "kiwi", "turnip", "radish", "mango", "papaya", "pumpkin", "leek", "fennel", "quince",
    "lychee", "plum", "parsnip", "celery", "beetroot", "okra", "guava", "pepper", "fig", "melon",
    "cabbage", "rhubarb", "apricot", "zucchini", "artichoke", "cherry", "lemon", "pear", "peach", "olive",
    "chard", "yam", "endive", "tomato", "cucumber", "broccoli", "sprout", "pomelo", "nectarine", "shallot",
]


def codename(taken=()) -> str:
    """A fresh ``verb-produce`` name that is not in ``taken``."""
    taken = set(taken)
    if len(taken) >= len(VERBS) * len(PRODUCE):
        raise RuntimeError("no codename left")
    while True:
        name = f"{secrets.choice(VERBS)}-{secrets.choice(PRODUCE)}"
        if name not in taken:
            return name
