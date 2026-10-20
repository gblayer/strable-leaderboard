"""API models: models that run on their provider's servers (``hardware: api`` in the card).

The card's ``api`` block says how to call the model, nothing secret: ``auth`` (``Bearer``, or ``Api-Key``
for an endpoint that wants it) and ``concurrency``, the most tasks the board runs at once. The base URL
must be https://, unless the block also says ``allow_http: true``: the rows and the key then travel
unencrypted (the STRABLE tables are public, the key is not). The base URL and the key live only in
``private/api_keys/<model folder>.env`` ($STRABLE_API_KEYS overrides the folder), git-ignored, readable
by its owner only (chmod 600), two lines:

    STRABLE_API_URL=https://api.yhatlabs.com
    STRABLE_API_KEY=...

The trusted parent, ``strable_leaderboard.run``, reads it (``child_env``) and passes these two
variables, with STRABLE_API_AUTH from the card, to that model's sandboxed child only. The board sends
what every model gets: the training rows with their labels and the test rows without theirs; the test
labels never leave the board, which scores the predictions. Nothing here prints the key or the URL.

``request`` calls ``<base URL><path>``. It tries again after a transport error, HTTP 408, 409, 429
(except ``quota_exceeded``) and 5xx, and a 200 that is not JSON, waiting ``Retry-After`` when the reply
gives it, with the same ``Idempotency-Key`` on every try, as YHat Labs' error documentation asks
(https://yhatlabs.com/models/docs/api/errors.html); any other error, or a 200 that carries ``error``,
fails at once. Its retry loop is ported from TabArena's Chakra-Tab client (autogluon/tabarena, commit
860337d, packages/tabarena/src/tabarena/systems/chakra_tab/system.py, Apache-2.0: see NOTICE).

``table_predict`` is ``POST /v1/table/predict`` of YHat Labs' API: a table goes inline as base64
Parquet, or, above ``INLINE_BYTES`` (8 MB, as their documentation says their client does), through
``POST /v1/uploads`` and by reference; a body stays under ``MAX_BODY`` (64 MB). Each call uploads its
own copy of a table and deletes it afterwards, since a key may hold only 200 tables.
"""

from __future__ import annotations

import base64
import email.utils
import hashlib
import io
import json
import os
import stat
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
URL, KEY, AUTH = "STRABLE_API_URL", "STRABLE_API_KEY", "STRABLE_API_AUTH"
SCHEMES = ("Bearer", "Api-Key")
ATTEMPTS, WAIT, MAX_WAIT = 4, 30.0, 600.0  # tries per request; the pause before try n: Retry-After, else WAIT * (n - 1)
AGAIN = {408, 409, 429}  # and 5xx
INLINE_BYTES, MAX_BODY = 8 * 2**20, 64 * 2**20
SLEEP = time.sleep  # the tests replace it


def key_file(folder: Path) -> Path:
    return Path(os.environ.get("STRABLE_API_KEYS", ROOT / "private" / "api_keys")) / f"{Path(folder).name}.env"


def read_keys(path: Path, allow_http: bool = False) -> dict[str, str]:
    """The base URL and key of a key file, checked: present, owner-only, exactly the two variables, an
    https:// URL (or http:// with ``allow_http``)."""
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
    scheme = urlsplit(found[URL]).scheme
    if not (scheme == "https" or (scheme == "http" and allow_http)):
        raise ValueError(f"{path}: {URL} must be an https:// address (http:// only with api.allow_http: true in the card)")
    return found


def child_env(folder: Path, card: dict) -> dict[str, str]:
    """The variables the child of an API model gets; nothing for other models."""
    if card.get("hardware") != "api":
        return {}
    return {**read_keys(key_file(folder), card["api"].get("allow_http", False)), AUTH: card["api"]["auth"]}


def _endpoint() -> tuple[str, str, str]:
    url, key, auth = os.environ.get(URL), os.environ.get(KEY), os.environ.get(AUTH)
    if not (url and key and auth in SCHEMES):
        raise RuntimeError(f"no API endpoint or key: the runner sets {URL}, {KEY} and {AUTH} from the key file")
    return url.rstrip("/"), key, auth


def _hide(text: str) -> str:
    url, key, _ = _endpoint()
    for secret in (key, url, urlsplit(url).hostname or url):
        text = text.replace(secret, "***")
    return text


def _retry_after(r) -> float | None:
    value = r.headers.get("Retry-After") if r is not None else None
    if not value:
        return None
    try:
        return float(value)
    except ValueError:  # an HTTP date
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - time.time()) if when else None


def request(method: str, path: str, body: dict | None = None, data: bytes | None = None, auth: bool = True,
            timeout: float = 7200) -> dict:
    """``method`` on ``<base URL><path>`` (or on ``path`` itself when it is a full URL), JSON ``body`` or raw
    ``data``; the JSON reply. Tries again as the module docstring says."""
    import requests

    base, key, scheme = _endpoint()
    outside = bool(urlsplit(path).scheme)  # a signed storage URL: no header of the API's own
    url = path if outside else base + path
    headers = {"Authorization": f"{scheme} {key}"} if auth else {}
    if not outside:
        headers["Idempotency-Key"] = uuid.uuid4().hex  # the same on every try of this request
    if data is not None:
        headers["Content-Type"] = "application/octet-stream"
    failure, r = "", None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            r = requests.request(method, url, json=body, data=data, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            failure, again, r = f"{type(exc).__name__}: {str(exc)[:300]}", True, None
        else:
            try:
                reply = r.json()
            except ValueError:
                reply = None  # a gateway's HTML error page
            error = reply.get("error") if isinstance(reply, dict) else None
            if 200 <= r.status_code < 300 and isinstance(reply, dict) and not error:
                return reply
            if isinstance(error, dict):
                failure = f"HTTP {r.status_code} {error.get('type')}: {str(error.get('message'))[:300]}" + \
                    (f" (request {error['request_id']})" if error.get("request_id") else "")
            else:
                failure = f"HTTP {r.status_code}{'' if reply is not None else ' (not JSON)'}: {r.text[:300]}"
            quota = isinstance(error, dict) and error.get("type") == "quota_exceeded"
            again = (r.status_code in AGAIN and not quota) or r.status_code >= 500 or (r.status_code == 200 and reply is None)
        if not again or attempt == ATTEMPTS:
            break
        wait = _retry_after(r)
        SLEEP(min(MAX_WAIT, wait if wait is not None else WAIT * attempt))
    raise RuntimeError(f"API failed after {attempt} attempt(s): {_hide(failure)}")


def post(path: str, body: dict) -> dict:
    return request("POST", path, body)


def parquet_bytes(df, unique: bool = False) -> bytes:
    """The table as a Parquet file; ``unique`` adds a random id to its metadata, so that its hash, and the
    id of its upload, belong to this call alone."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    if unique:
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"strable_call": uuid.uuid4().hex.encode()})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def parquet_b64(df) -> dict:
    return {"format": "parquet_b64", "bytes": base64.b64encode(parquet_bytes(df)).decode()}


def send_upload(upload: dict, data: bytes) -> None:
    """Send the file where ``POST /v1/uploads`` said. The documentation names upload.method, upload.url and
    upload.auth, and gives PUT /v1/uploads/{id} an application/octet-stream body; it does not say what a
    signed storage URL wants beyond no key. Here: the raw Parquet bytes, with that content type, to
    upload.url (a path on the API, or a full URL), with the key only when upload.auth is true."""
    request(upload.get("method", "PUT"), upload["url"], data=data, auth=bool(upload.get("auth")))


def upload(df) -> str:
    """``POST /v1/uploads`` with the file's SHA-256 and size, then the file unless it is already there; its id."""
    data = parquet_bytes(df, unique=True)
    slot = post("/v1/uploads", {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data), "format": "parquet"})
    if not slot.get("exists"):
        send_upload(slot["upload"], data)
    return slot["id"]


def table_predict(train, test, **fields) -> dict:
    """``POST /v1/table/predict``: fit on ``train`` (with its target column) and predict ``test`` in one
    call; ``fields`` are the other body fields (target, model, preset, problem_type, eval_metric, seed,
    groups). A table above INLINE_BYTES is uploaded first and deleted after the call."""
    uploaded, tables = [], {}
    try:
        for name, df in (("train", train), ("test", test)):
            data = parquet_bytes(df)
            if len(data) > INLINE_BYTES:
                uploaded.append(upload(df))
                tables[name] = {"format": "table", "id": uploaded[-1]}
            else:
                tables[name] = {"format": "parquet_b64", "bytes": base64.b64encode(data).decode()}
        body = {**tables, **fields}
        size = len(json.dumps(body))
        if size > MAX_BODY:
            raise ValueError(f"the request body would be {size / 2**20:.0f} MB, above {MAX_BODY / 2**20:.0f} MB")
        return post("/v1/table/predict", body)
    finally:
        for table_id in uploaded:
            try:
                request("DELETE", f"/v1/uploads/{table_id}")
            except Exception as exc:  # a table left behind is deleted by the provider after a while
                print(f"could not delete the uploaded table {table_id}: {type(exc).__name__}", file=sys.stderr)

