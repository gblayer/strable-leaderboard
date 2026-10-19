"""A fake API vendor, for the tests and the audit's dry mode: a local HTTP server that speaks the
Chakra-Tab protocol of TabArena's client and never leaves this machine.

    python scripts/fake_api.py [--port 8765] [--key test-key] [--auth Bearer] [--model models/ridge] [--cheat]

One POST per task: JSON {op, train, target, test, preset, problem_type, eval_metric, time_limit, seed},
the tables as base64 Parquet, the target column in train only. The reply: {predictions} for regression,
{classes, probabilities} for classification, and fit: {version, fit_s, predict_s, total_s}. A wrong
Authorization header gets HTTP 401; ``metrics`` (FakeVendor) restricts the eval_metric it accepts and
answers HTTP 400 naming any other. The model is scikit-learn's HistGradientBoosting, its strings
ordinal-encoded; ``--model`` wraps a board model instead (its code then runs here, unsandboxed).

``--cheat`` (``FakeVendor(cheat=True)``) is a dishonest vendor, for the audit to catch: it answers the
true label of every test row it finds among its leaked labels (``leak``), as a vendor who looked the
public tables up would, and the honest model's prediction on any other row.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
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


def table(part: dict) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(base64.b64decode(part["bytes"])))


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
    """The server, on 127.0.0.1 and a free port: ``with FakeVendor() as v: ... v.url``. ``seen`` records
    what each request carried (never the key); ``fail`` lists HTTP statuses to answer first, with an
    HTML page (a gateway's error), for the client's retries."""

    def __init__(self, key: str = "test-key", auth: str = "Bearer", model=None, cheat: bool = False, port: int = 0,
                 metrics=None):
        self.key, self.auth, self.model, self.cheat, self.metrics = key, auth, model, cheat, metrics
        self.leaked, self.seen, self.fail, self.calls = {}, [], [], 0
        vendor = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                status, reply = vendor.answer(self.headers.get("Authorization"),
                                              self.rfile.read(int(self.headers.get("Content-Length", 0))))
                data = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/html" if isinstance(reply, bytes) else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/tabular/predict"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def leak(self, X: pd.DataFrame, y) -> None:
        """Labels the cheating vendor knows, by the values of their feature row."""
        self.leaked.update(zip(row_keys(X), np.asarray(y).tolist()))

    def answer(self, authorization: str | None, body: bytes) -> tuple[int, dict | bytes]:
        self.calls += 1
        if self.fail:
            return self.fail.pop(0), b"<html>502 Bad Gateway</html>"
        if authorization != f"{self.auth} {self.key}":
            return 401, {"error": "unauthorized"}
        t0 = time.perf_counter()
        req = json.loads(body)
        train, test = table(req["train"]), table(req["test"])
        y = train.pop(req["target"]).to_numpy()
        refused = self.metrics is not None and req["eval_metric"] not in self.metrics
        self.seen.append({"fields": sorted(req), "target_in_test": req["target"] in test.columns, "train": len(train),
                          "test": len(test), "preset": req["preset"], "seed": req["seed"],
                          "eval_metric": req["eval_metric"], "refused": refused})
        if refused:
            return 400, {"error": f"unsupported eval_metric {req['eval_metric']!r}; use one of {sorted(self.metrics)}"}
        task = req["problem_type"]
        out = honest(task, train, y, test, req["seed"], self.model)
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
            reply = {"classes": [c.item() if hasattr(c, "item") else c for c in classes], "probabilities": proba.tolist()}
        seconds = round(time.perf_counter() - t0, 4)
        reply["fit"] = {"version": VERSION, "fit_s": seconds, "predict_s": 0.0, "total_s": seconds}
        return 200, reply


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
