"""Checks API models: the card's api block, the key file, the client's retries, and the smoke test of an example
API model against the local fake vendor, through the sandbox, with its key and the network for its child only."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from strable_leaderboard import api
from strable_leaderboard import run as run_module
from strable_leaderboard.card import card_problems
from strable_leaderboard.check_results import check
from strable_leaderboard.smoke import SMOKE_TABLES, static_problems

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fake_api import FakeVendor  # noqa: E402

EXAMPLE = ROOT / "tests" / "api_model"
CARD = yaml.safe_load((EXAMPLE / "model.yaml").read_text())
TABLES = json.loads((ROOT / "splits" / "manifest.json").read_text())["tables"]
# Records the flags and the variable names it gets, then runs the command with only the variables that
# --cleanenv would pass (SINGULARITYENV_*), from the outputs folder.
SINGULARITY = """#!/bin/bash
args=("$@"); i=0
while [ "${args[$i]}" != --pwd ]; do i=$((i + 1)); done
echo "${args[*]:0:$((i + 3))} | $(env | grep -o '^SINGULARITYENV_[A-Z_]*' | sort | tr '\\n' ' ')" >> "$SANDBOX_LOG"
vars=(); while IFS= read -r line; do vars+=("${line#SINGULARITYENV_}"); done < <(env | grep '^SINGULARITYENV_')
cd "${args[$((i + 1))]}" && exec env -i PATH="$PATH" "${vars[@]}" "${args[@]:$((i + 3))}"
"""


def test_an_api_card_needs_its_api_block_and_nothing_secret():
    assert card_problems(CARD) == [] and static_problems(EXAMPLE) == []
    bare = {k: v for k, v in CARD.items() if k != "api"}
    assert "hardware: api needs api" in card_problems(bare)[0]
    for block in [{"auth": "Bearer", "concurrency": 2, "url": "https://x"}, {"auth": "Bearer", "concurrency": 2, "key": "k"}]:
        assert "never in the card" in card_problems(dict(CARD, api=block))[0]
    assert card_problems(dict(CARD, api={"auth": "Basic", "concurrency": 2})) == ["api.auth must be Bearer or Api-Key"]
    for concurrency in [0, True, "8", 1.5]:
        assert card_problems(dict(CARD, api={"auth": "Api-Key", "concurrency": concurrency})) == [
            "api.concurrency must be a positive whole number"]
    assert card_problems(dict(CARD, hardware="cpu")) == ["api is only for hardware: api"]
    assert card_problems(dict(CARD, api=dict(CARD["api"], allow_http="yes"))) == ["api.allow_http must be true or false"]


def write_keys(folder: Path, text: str, mode: int = 0o600) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "api_model.env"
    path.write_text(text)
    path.chmod(mode)
    return path


def test_the_key_file_is_checked_and_never_printed(tmp_path, monkeypatch):
    monkeypatch.setenv("STRABLE_API_KEYS", str(tmp_path))
    with pytest.raises(ValueError, match="no key file"):
        api.child_env(EXAMPLE, CARD)
    good = "# the vendor's raw endpoint\nSTRABLE_API_URL=https://api.example.org/v1\nSTRABLE_API_KEY=sk-secret-123\n"
    path = write_keys(tmp_path, good, 0o640)
    https_only = dict(CARD, api={"auth": "Bearer", "concurrency": 2})  # no allow_http: https:// only
    with pytest.raises(ValueError, match="chmod 600") as exc:
        api.child_env(EXAMPLE, https_only)
    path.chmod(0o600)
    assert api.child_env(EXAMPLE, https_only) == {"STRABLE_API_URL": "https://api.example.org/v1",
                                                  "STRABLE_API_KEY": "sk-secret-123", "STRABLE_API_AUTH": "Bearer"}
    assert api.child_env(EXAMPLE, dict(CARD, hardware="cpu")) == {}
    messages = [str(exc.value)]
    plain = good.replace("https://api.example.org", "http://api.example.org")
    for text, msg in [(good + "OTHER=sk-secret-123\n", "only STRABLE_API_URL and STRABLE_API_KEY"),
                      ("STRABLE_API_KEY=sk-secret-123\n", "needs both"), (plain, "http:// only with api.allow_http")]:
        write_keys(tmp_path, text)
        with pytest.raises(ValueError, match=msg) as exc:
            api.child_env(EXAMPLE, https_only)
        messages.append(str(exc.value))
    assert not any("sk-secret" in m for m in messages)
    assert CARD["api"]["allow_http"] is True  # the example's card accepts plain http, to any host
    assert api.child_env(EXAMPLE, CARD)["STRABLE_API_URL"] == "http://api.example.org/v1"


def body(rows=8):
    X = pd.DataFrame({"x": np.arange(rows, dtype=float)})
    return {"op": "fit_predict", "train": api.parquet_b64(X.assign(t=X["x"] * 2)), "target": "t",
            "test": api.parquet_b64(X), "preset": "medium", "problem_type": "regression", "eval_metric": "r2",
            "time_limit": None, "seed": 1234}


def test_the_client_retries_server_errors_and_hides_its_secrets(monkeypatch):
    monkeypatch.setattr(api, "WAIT", 0)
    with FakeVendor(key="sk-secret-123") as vendor:
        monkeypatch.setenv("STRABLE_API_URL", vendor.url)
        monkeypatch.setenv("STRABLE_API_AUTH", "Bearer")
        monkeypatch.setenv("STRABLE_API_KEY", "sk-secret-123")
        vendor.fail = [503, 200]  # a gateway's page, then a 200 that is not JSON: both tried again
        reply = api.post(body())
        assert len(reply["predictions"]) == 8 and reply["fit"]["version"] == "fake-1.0" and len(vendor.seen) == 1
        vendor.fail = [503] * api.ATTEMPTS
        with pytest.raises(RuntimeError, match=f"after {api.ATTEMPTS} attempt"):
            api.post(body())
        monkeypatch.setenv("STRABLE_API_KEY", "sk-wrong-456")
        with pytest.raises(RuntimeError, match="after 1 attempt.*HTTP 401"):  # a client error fails at once
            api.post(body())
        monkeypatch.setenv("STRABLE_API_URL", vendor.url.replace(str(vendor.server.server_address[1]), "9"))
    with pytest.raises(RuntimeError, match="ConnectionError") as exc:  # nothing listens: a transport error
        api.post(body())
    assert "127.0.0.1" not in str(exc.value) and "sk-wrong" not in str(exc.value)
    monkeypatch.delenv("STRABLE_API_KEY")
    with pytest.raises(RuntimeError, match="no API endpoint or key"):
        api.post(body())


def test_a_server_refusing_r2_fails_the_task_with_a_message_naming_it(tmp_path, monkeypatch, capfd):
    from strable_leaderboard.model import load_model_class
    from strable_leaderboard.smoke import check_one

    fake_tables(tmp_path / "data")
    with FakeVendor(key="k", metrics={"rmse", "roc_auc", "log_loss"}) as vendor:  # what TabArena's client sends
        for name, value in [("STRABLE_API_URL", vendor.url), ("STRABLE_API_KEY", "k"), ("STRABLE_API_AUTH", "Bearer")]:
            monkeypatch.setenv(name, value)
        with pytest.raises(RuntimeError, match=r"HTTP 400: .*'r2'.*\(sent eval_metric 'r2' for regression\)"):
            check_one(load_model_class(EXAMPLE), CARD, SMOKE_TABLES["regression"], data_dir=tmp_path / "data")
        assert check_one(load_model_class(EXAMPLE), CARD, SMOKE_TABLES["binary"], data_dir=tmp_path / "data")["score"] > 0.5
        assert [(s["eval_metric"], s["refused"]) for s in vendor.seen] == [("r2", True), ("roc_auc", False)]  # no retry
        # the same through the runner: the task fails, its log names the metric
        monkeypatch.setenv("STRABLE_API_KEYS", str(tmp_path / "keys"))
        write_keys(tmp_path / "keys", f"STRABLE_API_URL={vendor.url}\nSTRABLE_API_KEY=k\n")
        monkeypatch.chdir(ROOT)
        with pytest.raises(Exception):
            run_module.run(EXAMPLE, "tfidf", SMOKE_TABLES["regression"], 0, tmp_path / "data", None, "cpu", 1, sandbox=False)
        assert "sent eval_metric 'r2' for regression" in capfd.readouterr().err and len(vendor.seen) == 3


def fake_tables(data_dir: Path) -> None:
    """The three smoke tables, made up, with the manifest's row counts so that the board's folds apply."""
    rng = np.random.default_rng(0)
    for task, table in SMOKE_TABLES.items():
        n = TABLES[table]["n_rows"]
        X = pd.DataFrame({"x": rng.normal(size=n), "s": rng.choice(["red apple", "green pear", "blue plum"], size=n)})
        signal = X["x"].to_numpy() + (X["s"] == "red apple").to_numpy() + rng.normal(size=n) * 0.5
        y = {"regression": signal, "binary": (signal > 0.5).astype(int),
             "multiclass": np.digitize(signal, [0, 0.8, 1.5])}[task]
        (data_dir / table).mkdir(parents=True)
        X.assign(target=y).to_parquet(data_dir / table / "data.parquet")
        kind = {"regression": "regression", "binary": "b-classification", "multiclass": "m-classification"}[task]
        (data_dir / table / "config.json").write_text(json.dumps({"target_name": "target", "task": kind}))


@pytest.fixture
def sandboxed(tmp_path, monkeypatch):
    """A fake singularity on PATH, the made-up smoke tables, and a key file for the example model."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "singularity").write_text(SINGULARITY)
    (bin_dir / "singularity").chmod(0o755)
    (tmp_path / "base.sif").write_text("")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    for name, value in [("STRABLE_IMAGE", tmp_path / "base.sif"), ("STRABLE_CACHE", tmp_path / "cache"),
                        ("SANDBOX_LOG", tmp_path / "sandbox.log"), ("TMPDIR", tmp_path),
                        ("STRABLE_API_KEYS", tmp_path / "keys")]:
        monkeypatch.setenv(name, str(value))
    monkeypatch.setenv("STRABLE_API_KEY", "a-key-in-the-shell")  # never reaches a child
    monkeypatch.chdir(ROOT)
    fake_tables(tmp_path / "data")
    return tmp_path


def test_the_example_api_model_passes_the_smoke_test_through_the_sandbox(sandboxed, monkeypatch, capsys):
    rows, run = [], run_module.run
    monkeypatch.setattr(run_module, "run", lambda *a, **k: rows.append(run(*a, **k)) or rows[-1])
    with FakeVendor(key="sk-test-123") as vendor:
        write_keys(sandboxed / "keys", f"STRABLE_API_URL={vendor.url}\nSTRABLE_API_KEY=sk-test-123\n")
        assert run_module.smoke(EXAMPLE, sandboxed / "data", None, "cpu", 1, sandbox=True) == 0, capsys.readouterr().out
    assert capsys.readouterr().out.count("  ok    ") == 3
    assert [r["hardware"] for r in rows] == ["api"] * 3 and {r["api_version"] for r in rows} == {"fake-1.0"}
    assert all(r["score"] > 0.5 for r in rows)
    # one request per table: the warm-ups send nothing; train carries the target, test does not
    assert len(vendor.seen) == 3 and not any(s["target_in_test"] for s in vendor.seen)
    assert vendor.seen[0]["fields"] == sorted(["op", "train", "target", "test", "preset", "problem_type", "eval_metric",
                                               "time_limit", "seed"])
    calls = (sandboxed / "sandbox.log").read_text().splitlines()
    assert len(calls) == 9 and not any("--network none" in c for c in calls)  # 2 warm-ups and the timed child, x3
    assert all("SINGULARITYENV_STRABLE_API_KEY" in c and "SINGULARITYENV_STRABLE_API_AUTH" in c for c in calls)


def test_other_models_get_no_network_and_no_key(sandboxed):
    table = SMOKE_TABLES["regression"]
    write_keys(sandboxed / "keys", "STRABLE_API_URL=https://api.example.org\nSTRABLE_API_KEY=k\n")
    row = run_module.run(ROOT / "models" / "ridge", "tfidf", table, 0, sandboxed / "data", None, "cpu", 1)
    assert row["hardware"] == "cpu" and "api_version" not in row
    calls = (sandboxed / "sandbox.log").read_text().splitlines()
    assert len(calls) == 3 and all("--net --network none" in c and "STRABLE_API" not in c for c in calls)


def test_api_rows_carry_one_api_version(tmp_path):
    manifest = {"tables": {"r1": {"task": "regression", "n_rows": 30, "test_rows_per_outer_fold": [10, 10, 10]}}}
    rows = pd.DataFrame([dict(model="m", encoder="tfidf", variant="default", table="r1", task="regression", fold=f,
                              score=0.5, fit_s=None, predict_s=None, total_s=2.0, n_train=20, n_test=10, hardware="api",
                              package="requests==2.32.5", checkpoint_revision="", protocol_version="0.1",
                              date="2026-10-19", api_version="v1") for f in range(3)])
    path = tmp_path / "m.csv"
    for versions, problem in [(["v1"] * 3, None), (["v1", None, "v1"], "need the api_version"),
                              (["v1", "v2", "v1"], "api_version changed during the run: ['v1', 'v2']")]:
        rows.assign(api_version=versions).to_csv(path, index=False)
        found = check(path, manifest)[0]
        assert (found == []) if problem is None else any(problem in p for p in found), found
    rows.drop(columns="api_version").to_csv(path, index=False)
    assert any("need the api_version" in p for p in check(path, manifest)[0])
