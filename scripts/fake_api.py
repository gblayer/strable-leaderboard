"""A fake API vendor, for the tests and the audit's dry mode: a local HTTP server that speaks YHat Labs'
public tabular API (https://yhatlabs.com/models/docs/api/overview.html) and never leaves this machine.

    python scripts/fake_api.py [--port 8765] [--key test-key] [--auth Bearer] [--model models/ridge] [--cheat]

``POST /v1/table/predict`` fits on train and predicts test in one call. Body: train, target, test, and
model, preset, problem_type, eval_metric, time_limit, seed, groups, time_col; any other field is refused.
A table is a list of row objects, {format: parquet_b64, bytes} or an uploaded one, {format: table, id,
rows, drop}. The groups column of train is no feature. The reply: predictions, classes and probabilities,
fit {rows, features, seconds, problem_type, metric, preset, seed, version} and usage {rows, credits}.
``POST /v1/uploads`` {sha256, size_bytes, format} answers {id, exists, upload: {method, url, auth}};
``PUT /v1/uploads/{id}`` takes the Parquet bytes, checked against the hash; ``DELETE`` removes them.

Errors are {error: {type, message, request_id}}: a wrong Authorization header gets 401, a predict body
above ``max_body`` (64 MB) 413 request_too_large, an eval_metric outside ``metrics`` 400. ``fail`` lists
replies to give first: a status alone answers an HTML page (a gateway's error), a tuple (status, type,
seconds) a JSON error with Retry-After: seconds (None for none). The model is scikit-learn's
HistGradientBoosting, its strings ordinal-encoded; ``--model`` wraps a board model instead (its code then
runs here, unsandboxed).

``--cheat`` (``FakeVendor(cheat=True)``) is a dishonest vendor, for the audit to catch: it answers the
true label of every test row it finds among its leaked labels (``leak``), as a vendor who looked the
public tables up would, and the honest model's prediction on any other row.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VERSION = "fake-1.0"
FIELDS = {"train", "target", "test", "quantiles", "set_coverage", "model", "preset", "problem_type", "eval_metric",
          "time_limit", "seed", "groups", "time_col"}
METRICS = {"regression": {"rmse", "mae", "r2"}, "binary": {"log_loss", "roc_auc", "accuracy", "f1"}}
METRICS["multiclass"] = METRICS["binary"]
TABLE = re.compile(r"^/v1/uploads/(tbl_[a-f0-9]{40})$")


class Refused(Exception):
    def __init__(self, status: int, kind: str, message: str):
        super().__init__(message)
        self.status, self.kind = status, kind


def row_keys(X: pd.DataFrame) -> list[tuple]:
    return [tuple(row) for row in X.astype(str).to_numpy()]


def honest(task: str, X_train: pd.DataFrame, y, X_test: pd.DataFrame, seed: int, model=None):
    """Predictions (regression) or (classes, probabilities) of the honest model."""
    if model:
        from strable_leaderboard.model import load_model_class

        m = load_model_class(Path(model))(task=task, seed=seed).fit(X_train, y)
        return m.predict(X_test) if task == "regression" else (list(m.classes_), m.predict_proba(X_test))
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.preprocessing import OrdinalEncoder
    from threadpoolctl import threadpool_limits

    strings = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    if strings:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X_train[strings].astype(str))
        X_train, X_test = X_train.copy(), X_test.copy()
        for X in (X_train, X_test):
            X[strings] = encoder.transform(X[strings].astype(str))
    with threadpool_limits(4):
        if task == "regression":
            return HistGradientBoostingRegressor(random_state=seed).fit(X_train, y).predict(X_test)
        m = HistGradientBoostingClassifier(random_state=seed).fit(X_train, y)
        return list(m.classes_), m.predict_proba(X_test)


class FakeVendor:
    """The server, on 127.0.0.1 and a free port: ``with FakeVendor() as v: ... v.url`` (the base URL).
    ``seen`` records what each predict call carried (never the key); ``uploads`` holds the stored
    tables, ``deleted`` the ids deleted; ``calls`` counts every request."""

    def __init__(self, key: str = "test-key", auth: str = "Bearer", model=None, cheat: bool = False, port: int = 0,
                 metrics=None, max_body: int = 64 * 2**20):
        self.key, self.auth, self.model, self.cheat, self.metrics, self.max_body = key, auth, model, cheat, metrics, max_body
        self.leaked, self.seen, self.fail, self.calls = {}, [], [], 0
        self.slots, self.uploads, self.deleted = {}, {}, []
        vendor = self

        class Handler(BaseHTTPRequestHandler):
            def serve(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                status, reply, headers = vendor.answer(self.command, self.path, self.headers.get("Authorization"), body)
                data = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/html" if isinstance(reply, bytes) else "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            do_POST = do_PUT = do_DELETE = serve

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def leak(self, X: pd.DataFrame, y) -> None:
        """Labels the cheating vendor knows, by the values of their feature row."""
        self.leaked.update(zip(row_keys(X), np.asarray(y).tolist()))

    def answer(self, method: str, path: str, authorization: str | None, body: bytes) -> tuple[int, dict | bytes, dict]:
        self.calls += 1
        if self.fail:
            failure = self.fail.pop(0)
            if isinstance(failure, int):
                return failure, b"<html>502 Bad Gateway</html>", {}
            status, kind, seconds = failure
            return status, error(kind, "the fake vendor refuses this one"), {} if seconds is None else {"Retry-After": str(seconds)}
        try:
            if authorization != f"{self.auth} {self.key}":
                raise Refused(401, "authentication_error", "invalid API key")
            if method == "POST" and path == "/v1/table/predict":
                if len(body) > self.max_body:  # an upload's file may reach 1 GB
                    raise Refused(413, "request_too_large", f"the body is above {self.max_body} bytes")
                return 200, self.predict(json.loads(body)), {}
            if method == "POST" and path == "/v1/uploads":
                return 200, self.slot(json.loads(body)), {}
            m = TABLE.match(path)
            if m and method == "PUT" and m.group(1) in self.slots:
                if hashlib.sha256(body).hexdigest() != self.slots[m.group(1)]:
                    raise Refused(400, "invalid_request_error", "the file does not match its sha256")
                self.uploads[m.group(1)] = body
                return 200, {}, {}
            if m and method == "DELETE" and m.group(1) in self.uploads:
                del self.uploads[m.group(1)]
                self.deleted.append(m.group(1))
                return 200, {}, {}
            raise Refused(404, "not_found", f"no {method} {path}")
        except Refused as exc:
            return exc.status, error(exc.kind, str(exc)), {}

    def slot(self, req: dict) -> dict:
        if not re.fullmatch(r"[a-f0-9]{64}", str(req.get("sha256"))) or not isinstance(req.get("size_bytes"), int):
            raise Refused(400, "invalid_request_error", "sha256 and size_bytes are required")
        table_id = "tbl_" + req["sha256"][:40]
        self.slots[table_id] = req["sha256"]
        return {"id": table_id, "exists": table_id in self.uploads,
                "upload": {"method": "PUT", "url": f"/v1/uploads/{table_id}", "auth": True}}

    def table(self, part) -> tuple[pd.DataFrame, str]:
        if isinstance(part, list):
            return pd.DataFrame(part), "rows"
        if part.get("format") == "parquet_b64":
            return pd.read_parquet(io.BytesIO(base64.b64decode(part["bytes"]))), "parquet_b64"
        if part.get("format") == "table" and part.get("id") in self.uploads:
            df = pd.read_parquet(io.BytesIO(self.uploads[part["id"]]))
            df = df.iloc[part["rows"]] if "rows" in part else df
            return df.drop(columns=part.get("drop", [])).reset_index(drop=True), "table"
        raise Refused(404, "not_found", "unknown table")

    def predict(self, req: dict) -> dict:
        unknown = sorted(set(req) - FIELDS)
        if unknown or not {"train", "target", "test"} <= set(req):
            raise Refused(400, "invalid_request_error", f"unknown fields {unknown}, or train, target or test missing")
        t0 = time.perf_counter()
        (train, train_format), (test, test_format) = self.table(req["train"]), self.table(req["test"])
        y = train.pop(req["target"]).to_numpy()
        task = req.get("problem_type") or ("regression" if np.issubdtype(y.dtype, np.floating) else "multiclass")
        groups = req.get("groups")
        if groups and groups not in train.columns:
            raise Refused(400, "invalid_request_error", f"groups {groups!r} is not a train column")
        folds = train.pop(groups).tolist() if groups else None  # the groups column is no feature
        metric = req.get("eval_metric") or ("rmse" if task == "regression" else "log_loss")
        refused = metric not in (self.metrics if self.metrics is not None else METRICS[task])
        self.seen.append({"fields": sorted(req), "target_in_test": req["target"] in test.columns,
                          "groups_in_test": bool(groups) and groups in test.columns, "groups": groups, "folds": folds,
                          "train_format": train_format, "test_format": test_format, "train": len(train), "test": len(test),
                          "model": req.get("model"), "preset": req.get("preset"), "seed": req.get("seed"),
                          "eval_metric": metric, "refused": refused})
        if refused:
            raise Refused(400, "invalid_request_error", f"eval_metric {metric!r} is not supported for {task}")
        out = honest(task, train, y, test.drop(columns=[groups] if groups in test.columns else []), req.get("seed", 0),
                     self.model)
        known = [self.leaked.get(k) for k in row_keys(test)] if self.cheat else []
        if task == "regression":
            pred = np.asarray(out, dtype=float)
            for i, v in enumerate(known):
                if v is not None:
                    pred[i] = v
            reply = {"predictions": pred.tolist()}
        else:
            classes, proba = out[0], np.asarray(out[1], dtype=float)
            for i, v in enumerate(known):
                if v is not None and v in classes:
                    proba[i] = 0.02 / (len(classes) - 1)
                    proba[i, classes.index(v)] = 0.98
            classes = [c.item() if hasattr(c, "item") else c for c in classes]
            reply = {"predictions": [classes[i] for i in proba.argmax(axis=1)], "classes": classes,
                     "probabilities": proba.tolist()}
        reply["fit"] = {"rows": len(train), "features": train.shape[1], "seconds": round(time.perf_counter() - t0, 4),
                        "problem_type": task, "metric": metric, "preset": req.get("preset", "fast"),
                        "seed": req.get("seed", 0), "version": VERSION}
        reply["usage"] = {"rows": len(test), "credits": 0}
        return reply


def error(kind: str, message: str) -> dict:
    return {"error": {"type": kind, "message": message, "request_id": "req_fake"}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--key", default="test-key")
    parser.add_argument("--auth", default="Bearer", choices=["Bearer", "Api-Key"])
    parser.add_argument("--model", help="a board model folder to answer with, instead of HistGradientBoosting")
    parser.add_argument("--cheat", action="store_true")
    args = parser.parse_args(argv)
    with FakeVendor(args.key, args.auth, args.model, args.cheat, args.port) as vendor:
        print(f"fake vendor at {vendor.url}, Authorization: {args.auth} <--key>; Ctrl-C stops it", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
