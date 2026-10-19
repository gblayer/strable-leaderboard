"""API models: models that run on their provider's servers (``hardware: api`` in the card).

The card's ``api`` block says how to call the model, nothing secret: ``auth``, ``Bearer`` (a public API)
or ``Api-Key`` (a provider's raw endpoint), and ``concurrency``, the most tasks the board runs at once.
The endpoint and the key live only in ``private/api_keys/<model folder>.env`` ($STRABLE_API_KEYS
overrides the folder), git-ignored, readable by its owner only (chmod 600), two lines:

    STRABLE_API_URL=https://...
    STRABLE_API_KEY=...

The trusted parent, ``strable_leaderboard.run``, reads it (``child_env``) and passes these two
variables, with STRABLE_API_AUTH from the card, to that model's sandboxed child only. The model sends
its request with ``post``. The board sends what every model gets: the training rows with their
labels and the test rows without theirs; the test labels never leave the board, which scores the
predictions. Nothing here prints the key or the endpoint.

``post`` is ported from TabArena's Chakra-Tab client (autogluon/tabarena, commit 860337d,
packages/tabarena/src/tabarena/systems/chakra_tab/system.py, Apache-2.0: see NOTICE), changed to read
the endpoint, key and scheme from the runner's variables and to hide them in its errors.
"""

from __future__ import annotations

import base64
import io
import os
import stat
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
URL, KEY, AUTH = "STRABLE_API_URL", "STRABLE_API_KEY", "STRABLE_API_AUTH"
SCHEMES = ("Bearer", "Api-Key")
LOCAL = ("127.0.0.1", "localhost", "::1")  # plain http only to this machine: a test vendor
ATTEMPTS, WAIT = 4, 30.0  # tries per request, and the pause before try n is WAIT * (n - 1) seconds


def key_file(folder: Path) -> Path:
    return Path(os.environ.get("STRABLE_API_KEYS", ROOT / "private" / "api_keys")) / f"{Path(folder).name}.env"


def read_keys(path: Path) -> dict[str, str]:
    """The endpoint and key of a key file, checked: present, owner-only, exactly the two variables."""
    if not path.is_file():
        raise ValueError(f"no key file {path}: write {URL}=... and {KEY}=... into it, then chmod 600")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError(f"{path} is readable by others: chmod 600 {path}")
    found = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, value = line.partition("=")
            if name.strip() not in (URL, KEY):
                raise ValueError(f"{path}: only {URL} and {KEY} are allowed, found {name.strip()[:40]!r}")
            found[name.strip()] = value.strip()
    if not (found.get(URL) and found.get(KEY)):
        raise ValueError(f"{path} needs both {URL} and {KEY}")
    url = urlsplit(found[URL])
    if not (url.scheme == "https" or (url.scheme == "http" and url.hostname in LOCAL)):
        raise ValueError(f"{path}: {URL} must be an https:// address")
    return found


def child_env(folder: Path, card: dict) -> dict[str, str]:
    """The variables the child of an API model gets; nothing for other models."""
    if card.get("hardware") != "api":
        return {}
    return {**read_keys(key_file(folder)), AUTH: card["api"]["auth"]}


def parquet_b64(df) -> dict:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return {"format": "parquet_b64", "bytes": base64.b64encode(buf.getvalue()).decode()}


def post(body: dict, timeout: float = 7200) -> dict:
    """POST ``body`` as JSON to the endpoint and return the JSON reply. Transport errors, HTTP 429 and
    5xx, and a 200 reply that is not JSON are tried again with a growing pause; any other error, or
    a reply with an ``error`` field, fails at once."""
    import requests

    url, key, auth = os.environ.get(URL), os.environ.get(KEY), os.environ.get(AUTH)
    if not (url and key and auth in SCHEMES):
        raise RuntimeError(f"no API endpoint or key: the runner sets {URL}, {KEY} and {AUTH} from the key file")

    def hide(text: str) -> str:
        for secret in (key, url, urlsplit(url).hostname or url):
            text = text.replace(secret, "***")
        return text

    failure = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            r = requests.post(url, json=body, headers={"Authorization": f"{auth} {key}"}, timeout=timeout)
        except requests.RequestException as exc:
            failure, again = f"{type(exc).__name__}: {str(exc)[:300]}", True
        else:
            try:
                reply = r.json()
            except ValueError:
                reply = None  # a gateway's HTML error page
            if r.status_code == 200 and isinstance(reply, dict) and "error" not in reply:
                return reply
            failure = f"HTTP {r.status_code}{'' if reply is not None else ' (not JSON)'}: {r.text[:300]}"
            again = r.status_code == 429 or r.status_code >= 500 or (r.status_code == 200 and reply is None)
        if not again or attempt == ATTEMPTS:
            break
        time.sleep(WAIT * attempt)
    raise RuntimeError(f"API failed after {attempt} attempt(s): {hide(failure)}")
