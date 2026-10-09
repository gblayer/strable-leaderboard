"""The dispatcher with a fake GitHub, fake sbatch and fake squeue: no cluster, no network."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import dispatch  # noqa: E402

CARD = {"name": "ExtraTrees", "version": "sklearn-1.7", "handles_strings": False, "tasks": ["regression"],
        "variant": "default", "access": "open", "hardware": "cpu", "package": "scikit-learn==1.7.2",
        "checkpoint": None, "licence": "BSD-3-Clause", "trained_on_strable": []}


class FakeGitHub:
    def __init__(self, merged_by="gblayer", labels=(), pulls=True):
        self.posts, self.merged_by, self.labels, self.pulls = [], merged_by, list(labels), pulls
        self.results_pr = {"number": 9, "state": "open", "merged_at": None}

    def get(self, path):
        if path.startswith("/commits/"):
            return [{"number": 3, "merged_at": "2026-10-09T10:00:00Z"}] if self.pulls else []
        if path == "/pulls/3":
            return {"number": 3, "merged_by": {"login": self.merged_by}, "labels": [{"name": n} for n in self.labels]}
        if path == "/pulls/9":
            return self.results_pr
        raise AssertionError(path)

    def post(self, path, data):
        self.posts.append((path, data))
        return {"number": 9}


@pytest.fixture
def clone(tmp_path, monkeypatch):
    """A git clone on main with one model, fake sbatch/squeue on PATH, and a Dispatcher."""
    root = tmp_path / "clone"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "run_array.sh", root / "scripts")
    (root / "private").mkdir()
    (root / "private" / "runner.env").write_text('STRABLE_SBATCH_GPU="--partition=gpuq --gres=gpu:1"\n'
                                                 'STRABLE_SBATCH_CPU="--partition=cpuq"\n')
    (root / "models" / "extratrees").mkdir(parents=True)
    (root / "models" / "extratrees" / "model.yaml").write_text(yaml.safe_dump(CARD))
    git = lambda *a: subprocess.run(["git", *a], cwd=root, check=True, capture_output=True)  # noqa: E731
    git("init", "-q", "-b", "main")
    git("-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "add")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text(f'#!/bin/bash\necho "$@" >> {tmp_path}/sbatch.log\necho 4242\n')
    (bin_dir / "squeue").write_text(f'#!/bin/bash\ncat {tmp_path}/queue 2>/dev/null\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    d = dispatch.Dispatcher(root=root, cache=tmp_path / "cache", gh=FakeGitHub(), python=sys.executable)
    d.todo = ["models/extratrees tfidf t1 0", "models/extratrees tfidf t1 1"]
    d.embeddings = []
    monkeypatch.setattr(d, "missing_tasks", lambda folder: list(d.todo))
    monkeypatch.setattr(d, "missing_embeddings", lambda: list(d.embeddings))
    monkeypatch.setattr(d, "prepare", lambda folder: (True, "ok"))
    monkeypatch.setattr(d, "check_results", lambda csv: (True, "coverage ok"))
    d.published = []
    monkeypatch.setattr(d, "open_results_pr", lambda *a: d.published.append(a) or 9)
    monkeypatch.setattr(d, "tag", lambda *a: None)
    monkeypatch.setattr("gather_results.gather", lambda runs, results: [])
    d.tmp = tmp_path
    return d


def state(d):
    return json.loads((d.state_dir / "extratrees.json").read_text())


def finish_smoke(d, rc="0"):
    (d.state_dir / "smoke").mkdir(parents=True, exist_ok=True)
    (d.state_dir / "smoke" / "extratrees.rc").write_text(rc)
    (d.state_dir / "smoke" / "extratrees.log").write_text("FAIL boom\n")


def test_full_path_without_messages_on_success(clone):
    d = clone
    d.run(pull=False)
    assert state(d)["step"] == "smoke" and state(d)["pr"] == 3
    sbatch = (d.tmp / "sbatch.log").read_text()
    assert "--partition=cpuq" in sbatch and "--time=00:30:00" in sbatch and "gpu" not in sbatch
    assert (d.cache / "logs").is_dir()  # created before submission

    (d.tmp / "queue").write_text("4242\n")  # still running: nothing moves
    d.run(pull=False)
    assert state(d)["step"] == "smoke"

    (d.tmp / "queue").unlink()
    finish_smoke(d)
    d.run(pull=False)
    assert state(d)["step"] == "run" and state(d)["n_tasks"] == 2
    assert (d.state_dir / "extratrees.run.txt").read_text().splitlines() == d.todo

    d.todo = []
    d.run(pull=False)
    assert state(d)["step"] == "published" and state(d)["version"] == "v0.1.0"
    assert yaml.safe_load((d.root / "results" / "versions.yaml").read_text())[0]["note"] == "Added ExtraTrees sklearn-1.7"

    d.gh.results_pr = {"number": 9, "state": "closed", "merged_at": "x", "merge_commit_sha": "abc"}
    d.run(pull=False)
    assert state(d)["step"] == "done"
    assert d.gh.posts == []  # no notification on success


def test_gpu_model_smokes_on_the_reference_gpu(clone):
    d = clone
    card = dict(CARD, hardware="gpu")
    (d.root / "models" / "extratrees" / "model.yaml").write_text(yaml.safe_dump(card))
    d.run(pull=False)
    sbatch = (d.tmp / "sbatch.log").read_text()
    assert "--partition=gpuq --gres=gpu:1" in sbatch


def test_smoke_failure_notifies_once(clone):
    d = clone
    d.run(pull=False)
    finish_smoke(d, rc="1")
    d.run(pull=False)
    d.run(pull=False)
    assert state(d)["step"] == "failed"
    (path, data), = d.gh.posts
    assert path == "/dispatches" and data["event_type"] == "smoke-failed"
    assert data["client_payload"]["pr"] == 3 and "FAIL boom" in data["client_payload"]["reason"]


def test_run_is_resubmitted_once_then_fails(clone):
    d = clone
    d.run(pull=False)
    finish_smoke(d)
    d.run(pull=False)  # first submission
    d.run(pull=False)  # tasks left: one resubmission
    assert state(d)["step"] == "run" and state(d)["retries"] == 1 and d.gh.posts == []
    d.run(pull=False)
    assert state(d)["step"] == "failed" and d.gh.posts[0][1]["event_type"] == "run-failed"


def test_embeddings_come_first_and_are_shared(clone):
    d = clone
    d.embeddings = ["embed llama-3.1-8b t1"]
    d.run(pull=False)
    finish_smoke(d)
    d.run(pull=False)
    assert state(d)["step"] == "embed"
    assert (d.state_dir / "_embed.embed.txt").read_text().strip() == "embed llama-3.1-8b t1"
    assert "--gres=gpu:1" in (d.tmp / "sbatch.log").read_text().splitlines()[-1]
    d.embeddings = []
    d.run(pull=False)
    assert state(d)["step"] == "run"


def test_unapproved_merge_never_runs(clone):
    d = clone
    d.gh.merged_by = "someone-else"
    d.run(pull=False)
    assert state(d)["step"] == "unapproved" and not (d.tmp / "sbatch.log").exists()


def test_auto_merge_needs_the_label(clone):
    d = clone
    d.gh.merged_by, d.gh.labels = "github-actions[bot]", ["claude-approved"]
    assert d.approval(Path("models/extratrees"))[0]
    d.gh.labels = []
    assert not d.approval(Path("models/extratrees"))[0]
    d.gh.pulls = False  # pushed to main directly: only the maintainer can
    assert d.approval(Path("models/extratrees"))[0]


def test_private_failures_do_not_leak_the_name(clone):
    d = clone
    d.notify("run-failed", {"model": "secret-model", "folder": "private/secret-model"}, "secret details")
    payload = d.gh.posts[0][1]["client_payload"]
    assert "secret" not in json.dumps(payload)


def test_only_one_results_pull_request_at_a_time(clone):
    d = clone
    d.state_dir.mkdir(parents=True)
    (d.state_dir / "other.json").write_text(json.dumps({"step": "published"}))
    d.run(pull=False)
    finish_smoke(d)
    d.run(pull=False)
    d.todo = []
    d.run(pull=False)
    d.run(pull=False)
    assert state(d)["step"] == "publish" and d.published == []


def test_lock_blocks_a_second_pass(clone):
    import fcntl

    d = clone
    d.state_dir.mkdir(parents=True)
    with open(d.state_dir / ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert d.run(pull=False) == 0
    assert not (d.state_dir / "extratrees.json").exists()


def test_next_version(clone):
    assert clone.next_version([]) == "v0.1.0"
    assert clone.next_version([{"version": "v0.1.9"}]) == "v0.1.10"


def test_token_file_must_exist_and_be_private(tmp_path):
    path = tmp_path / "github_token"
    with pytest.raises(SystemExit, match="no GitHub token"):
        dispatch.token_file(path)
    path.write_text("github_pat_x\n")
    path.chmod(0o644)
    with pytest.raises(SystemExit, match="chmod 600"):
        dispatch.token_file(path)
    path.chmod(0o600)
    assert dispatch.GitHub(path=path).token() == "github_pat_x"


def test_git_reads_the_token_file_not_the_store(clone, tmp_path):
    path = tmp_path / "github_token"
    path.write_text("github_pat_x\n")
    path.chmod(0o600)
    env = dict(clone.env, STRABLE_GH_TOKEN_FILE=str(path))
    out = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", env=env,
                         capture_output=True, text=True, check=True).stdout
    assert "password=github_pat_x" in out
