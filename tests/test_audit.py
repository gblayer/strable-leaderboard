"""Checks that the audit of scripts/audit_api.py passes the honest fake vendor and fails the one that looks the
public tables' test labels up, on made-up smoke tables and a synthetic table."""

import re
import sys
from functools import partial
from pathlib import Path

import pytest

from test_api import EXAMPLE, fake_tables

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import audit_api  # noqa: E402


@pytest.fixture
def audit(tmp_path, monkeypatch, capsys):
    fake_tables(tmp_path / "data")
    for name in ("STRABLE_API_URL", "STRABLE_API_KEY", "STRABLE_API_AUTH"):
        monkeypatch.setenv(name, "")  # restored after the test: the audit sets them
    monkeypatch.setattr(audit_api, "limit_threads", lambda n: None)  # keeps this process's thread pools
    monkeypatch.setattr(audit_api, "synthetic_table", partial(audit_api.synthetic_table, n=800))

    def run(dry):
        code = audit_api.main([str(EXAMPLE), "--dry", dry, "--seed", "7", "--data-dir", str(tmp_path / "data")])
        lines = capsys.readouterr().out.splitlines()
        return code, {tuple(m.groups()[1:]): m.group(1) for line in lines
                      if (m := re.match(r"\[(\w+)\s*\] (\w+)\s+(\S+)", line))}
    return run


def test_the_honest_vendor_passes_and_the_cheat_fails(audit):
    code, verdicts = audit("honest")
    assert code == 0 and len(verdicts) == 20 and set(verdicts.values()) == {"PASS", "INFO"}
    code, verdicts = audit("cheat")
    assert code == 1
    for table in ("china-overseas-finance-inventory", "animalandveterinary-event", "drug-ndc"):
        assert [verdicts[(p, table)] for p in ("labels", "shuffled", "jitter")] == ["FAIL"] * 3, table
        assert verdicts[("transduction", table)] == "PASS"  # a lookup row by row does not depend on the batch
    assert {v for (p, t), v in verdicts.items() if t == "synthetic-binary"} == {"PASS", "INFO"}  # no public copy


def test_dry_runs_need_an_api_model(audit):
    with pytest.raises(SystemExit):
        audit_api.main([str(ROOT / "models" / "ridge"), "--dry", "honest"])


def test_the_audit_reads_the_key_file_of_a_test_folder(tmp_path, monkeypatch, capsys):
    """Pointed at models/test-<name>/, the audit calls the endpoint of private/api_keys/test-<name>.env."""
    import shutil

    from fake_api import FakeVendor
    from test_api import write_keys

    folder = tmp_path / "models" / "test-api_model"
    shutil.copytree(EXAMPLE, folder)
    for name in ("STRABLE_API_URL", "STRABLE_API_KEY", "STRABLE_API_AUTH"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("STRABLE_API_KEYS", str(tmp_path / "keys"))
    monkeypatch.setattr(audit_api, "limit_threads", lambda n: None)
    monkeypatch.setattr(audit_api, "tables", lambda card, data_dir, seed: [
        audit_api.encoded(audit_api.synthetic_table("binary", seed, n=400))])
    with FakeVendor(key="sk-test") as vendor:
        keys = write_keys(tmp_path / "keys", f"STRABLE_API_URL={vendor.url}\nSTRABLE_API_KEY=sk-test\n")
        keys.rename(tmp_path / "keys" / "test-api_model.env")
        assert audit_api.main([str(folder), "--seed", "3"]) == 0
    assert len(vendor.seen) == 8 and "FAIL" not in capsys.readouterr().out  # eight calls, all served
    # every call carries the model's groups: the board's inner folds of the training rows it was given
    assert all(s["groups"] == "__fold__" and set(s["folds"]) == set(range(8)) and not s["groups_in_test"] for s in vendor.seen)
