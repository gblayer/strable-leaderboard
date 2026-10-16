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

CARD = {"name": "ExtraTrees", "version": "sklearn-1.7", "end_to_end": False, "tasks": ["regression"],
        "variant": "default", "hardware": "cpu", "package": "scikit-learn==1.7.2",
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
    (bin_dir / "sbatch").write_text(f'#!/bin/bash\necho "$@" >> {tmp_path}/sbatch.log\n'
                                    f'echo "$STRABLE_RUNS" >> {tmp_path}/runs.log\necho 4242\n')
    (bin_dir / "squeue").write_text(f'#!/bin/bash\ncat {tmp_path}/queue 2>/dev/null\n')
    (bin_dir / "scontrol").write_text('#!/bin/bash\necho "JobId=4242 ArrayJobId=4242 ArrayTaskId=0-1%16 ArrayTaskThrottle=16"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    d = dispatch.Dispatcher(root=root, cache=tmp_path / "cache", gh=FakeGitHub(), python=sys.executable, ref="main")
    d.todo = ["models/extratrees tfidf t1 0", "models/extratrees tfidf t1 1"]
    d.embeddings = []
    monkeypatch.setattr(d, "missing_tasks", lambda folder, code: list(d.todo))
    monkeypatch.setattr(d, "missing_embeddings", lambda tables=None: list(d.embeddings))
    monkeypatch.setattr(d, "prepare", lambda folder, code: (True, "ok"))
    monkeypatch.setattr(d, "check_results", lambda csv, code: (True, "coverage ok"))
    d.published = []
    monkeypatch.setattr(d, "open_results_pr", lambda *a: d.published.append(a) or 9)
    monkeypatch.setattr(d, "tag", lambda *a: None)
    monkeypatch.setattr("gather_results.gather", lambda runs, results, only=None: [])
    d.tmp = tmp_path
    return d


def commit(root):
    for args in (["add", "."], ["commit", "-q", "-m", "change"]):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True,
                       capture_output=True)


def state(d, name="extratrees"):
    return json.loads((d.state_dir / f"{name}.json").read_text())


def finish_smoke(d, rc="0", name="extratrees"):
    (d.state_dir / "smoke").mkdir(parents=True, exist_ok=True)
    (d.state_dir / "smoke" / f"{name}.rc").write_text(rc)
    (d.state_dir / "smoke" / f"{name}.log").write_text("FAIL boom\n")


def test_full_path_with_one_comment_on_success(clone):
    d = clone
    d.run(fetch=False)
    assert state(d)["step"] == "smoke" and state(d)["pr"] == 3
    sbatch = (d.tmp / "sbatch.log").read_text()
    assert "--partition=cpuq" in sbatch and "--time=00:30:00" in sbatch and "gpu" not in sbatch
    assert (d.cache / "logs").is_dir()  # created before submission

    (d.tmp / "queue").write_text("4242\n")  # still running: nothing moves
    d.run(fetch=False)
    assert state(d)["step"] == "smoke"

    (d.tmp / "queue").unlink()
    finish_smoke(d)
    d.run(fetch=False)
    assert state(d)["step"] == "run" and state(d)["n_tasks"] == 2
    assert (d.state_dir / "extratrees.run.txt").read_text().splitlines() == d.todo
    (path, data), = d.gh.posts  # the run started: the only success comment
    assert path == "/issues/3/comments" and "2 tasks" in data["body"] and "no estimate" in data["body"]

    d.todo = []
    d.run(fetch=False)
    assert state(d)["step"] == "published" and state(d)["version"] == "v0.1.0" and state(d)["leakage"] is None
    assert yaml.safe_load((d.root / "results" / "versions.yaml").read_text())[0]["note"] == "Added ExtraTrees sklearn-1.7"

    d.gh.results_pr = {"number": 9, "state": "closed", "merged_at": "x", "merge_commit_sha": "abc"}
    d.run(fetch=False)
    assert state(d)["step"] == "done" and len(d.gh.posts) == 1  # nothing more on success


def test_gpu_model_smokes_on_the_reference_gpu(clone):
    d = clone
    (d.root / "models" / "extratrees" / "model.yaml").write_text(yaml.safe_dump(dict(CARD, hardware="gpu")))
    d.run(fetch=False)
    assert not (d.tmp / "sbatch.log").read_text().count("gpuq")  # the working copy is not what runs
    (d.state_dir / "extratrees.json").unlink()
    commit(d.root)
    d.run(fetch=False)
    sbatch = (d.tmp / "sbatch.log").read_text()
    assert "--partition=gpuq --gres=gpu:1" in sbatch


def test_smoke_failure_notifies_once(clone):
    d = clone
    d.run(fetch=False)
    finish_smoke(d, rc="1")
    d.run(fetch=False)
    d.run(fetch=False)
    assert state(d)["step"] == "failed"
    (path, data), = d.gh.posts
    assert path == "/dispatches" and data["event_type"] == "smoke-failed"
    assert data["client_payload"]["pr"] == 3 and "FAIL boom" in data["client_payload"]["reason"]


def test_run_is_resubmitted_once_then_fails(clone):
    d = clone
    d.run(fetch=False)
    finish_smoke(d)
    d.run(fetch=False)  # first submission, announced
    d.run(fetch=False)  # tasks left: the second submission, not announced
    assert state(d)["step"] == "run" and state(d)["submissions"] == 2 and len(d.gh.posts) == 1
    d.run(fetch=False)
    assert state(d)["step"] == "failed" and d.gh.posts[1][1]["event_type"] == "run-failed"


def test_embeddings_come_first_and_are_shared(clone):
    d = clone
    d.embeddings = ["embed qwen-3-8b-first30 t1"]
    d.run(fetch=False)
    finish_smoke(d)
    d.run(fetch=False)
    assert state(d)["step"] == "embed"
    assert (d.state_dir / "_embed.embed.txt").read_text().strip() == "embed qwen-3-8b-first30 t1"
    assert "--gres=gpu:1" in (d.tmp / "sbatch.log").read_text().splitlines()[-1]
    d.embeddings = []
    d.run(fetch=False)
    assert state(d)["step"] == "run"


def test_estimate_from_the_queue_and_the_observed_task_times(clone):
    d = clone
    logs = d.cache / "logs"
    logs.mkdir(parents=True)
    for i, (seconds, device) in enumerate([(600, "cpu"), (1200, "cpu"), (1800, "cpu"), (5, "cuda")]):
        (logs / f"1_{i}.out").write_text(f"task {i} on n1: models/x tfidf t 0 (device {device})\nscore\n"
                                         f"elapsed {seconds}s on {device}\n")
    (logs / "2_0.out").write_text("task 0: smoke\nFAIL\n")  # no elapsed line: not counted
    assert sorted(d.task_seconds("cpu")) == [600, 1200, 1800] and d.task_seconds("cuda") == [5]
    (d.tmp / "queue").write_text("77_1\n77_2\n4242_0\n4242_1\n")  # two tasks of another run ahead
    text = d.estimate("4242", 30, "cpu")  # ceil((2 + 30) / 16) = 2 waves of the 1200 s median
    assert "rough automatic estimate: 2 tasks queued ahead, 16 at a time, 20 minutes per task" in text
    assert "in about 40 minutes" in text
    assert dispatch.duration(3 * 3600) == "3 hours" and dispatch.duration(72 * 3600) == "3 days"


def test_the_overlap_label_flags_the_model_when_publishing(clone):
    d = clone
    d.gh.labels = ["claude-approved", "pretraining-overlap"]
    d.run(fetch=False)
    finish_smoke(d)
    d.run(fetch=False)
    d.todo = []
    d.run(fetch=False)
    assert state(d)["step"] == "published" and state(d)["leakage"] == "possible"


def test_flags_keep_other_models(tmp_path):
    path = tmp_path / "flags.json"
    dispatch.write_flags(path, "a", {"leakage": "possible", "pr": 4})
    dispatch.write_flags(path, "b", {"leakage": None, "pr": 5})
    assert json.loads(path.read_text()) == {"a": {"leakage": "possible", "pr": 4}}
    dispatch.write_flags(path, "a", {"leakage": None})
    assert json.loads(path.read_text()) == {}


def test_unapproved_merge_never_runs(clone):
    d = clone
    d.gh.merged_by = "someone-else"
    d.run(fetch=False)
    assert state(d)["step"] == "unapproved" and not (d.tmp / "sbatch.log").exists()


def test_auto_merge_needs_the_label(clone):
    d = clone
    d.gh.merged_by, d.gh.labels = "github-actions[bot]", ["claude-approved"]
    assert d.approval(Path("models/extratrees"))[0]
    d.gh.labels = []
    assert not d.approval(Path("models/extratrees"))[0]
    d.gh.pulls = False  # pushed to main directly: only the maintainer can
    assert d.approval(Path("models/extratrees"))[0]


def test_only_one_results_pull_request_at_a_time(clone):
    d = clone
    d.state_dir.mkdir(parents=True)
    (d.state_dir / "other.json").write_text(json.dumps({"step": "published"}))
    d.run(fetch=False)
    finish_smoke(d)
    d.run(fetch=False)
    d.todo = []
    d.run(fetch=False)
    d.run(fetch=False)
    assert state(d)["step"] == "publish" and d.published == []


def test_lock_blocks_a_second_pass(clone):
    import fcntl

    d = clone
    d.state_dir.mkdir(parents=True)
    with open(d.state_dir / ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert d.run(fetch=False) == 0
    assert not (d.state_dir / "extratrees.json").exists()


def test_next_version(clone):
    assert clone.next_version([]) == "v0.1.0"
    assert clone.next_version([{"version": "v0.1.9"}]) == "v0.1.10"
    assert clone.next_version([{"version": "v0.1.0-pre"}]) == "v0.1.0"
    with pytest.raises(ValueError):
        clone.next_version([{"version": "first"}])


def test_jobs_run_from_the_commit_the_model_started_on(clone):
    d = clone
    d.run(fetch=False)
    start = state(d)["commit"]
    assert "STRABLE_ROOT" not in d.env  # set per submission, never the clone
    (d.root / "scripts" / "later.txt").write_text("a later push\n")
    commit(d.root)
    finish_smoke(d)
    d.run(fetch=False)
    assert state(d)["commit"] == start and state(d)["step"] == "run"
    assert (d.state_dir / "checkouts" / start / "scripts" / "run_array.sh").exists()
    assert not (d.state_dir / "checkouts" / start / "scripts" / "later.txt").exists()
    d.todo = []
    d.run(fetch=False)
    d.gh.results_pr = {"number": 9, "state": "closed", "merged_at": "x", "merge_commit_sha": "abc"}
    d.run(fetch=False)  # done: only the head's checkout is left
    head = d.git("rev-parse", "main").stdout.strip()
    assert state(d)["step"] == "done" and [p.name for p in (d.state_dir / "checkouts").iterdir()] == [head] != [start]


def test_public_text_has_no_absolute_paths():
    text = 'File "/data/lab/x/models/m/model.py", line 3\nsee /home/me/.cache/a.log and cache/logs/1_2.out'
    out = dispatch.public(text, 1000)
    assert "/data/" not in out and "/home/" not in out and ".../model.py" in out and "cache/logs/1_2.out" in out
    assert dispatch.public("x" * 5000, 1500) == "x" * 1500
    assert "https://github.com/a/b" in dispatch.public("https://github.com/a/b", 100)


def test_retry_waits_on_network_errors_only(monkeypatch):
    import urllib.error

    monkeypatch.setattr(dispatch.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError("timed out")
        return "ok"

    assert dispatch.retry(flaky) == "ok" and len(calls) == 3
    calls.clear()

    def refused():
        calls.append(1)
        raise urllib.error.HTTPError("u", 422, "exists", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        dispatch.retry(refused)
    assert len(calls) == 1


def test_board_is_built_from_published_and_paper_files_only(tmp_path):
    import pandas as pd

    for model, hardware in [("ridge", "paper"), ("half-run", "cpu"), ("published", "cpu"), ("new", "cpu")]:
        pd.DataFrame({"hardware": [hardware]}).to_csv(tmp_path / f"{model}.csv", index=False)
    site = tmp_path / "results.json"
    site.write_text(json.dumps({"pipelines": [{"id": "tfidf+published"}, {"id": "ridge"}, {"id": "tfidf+tuned:tuned"}]}))
    assert dispatch.board_models(tmp_path, site, "new") == ["new", "published", "ridge", "tuned"]


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
    env = dict(clone.git_env, STRABLE_GH_TOKEN_FILE=str(path))
    out = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", env=env,
                         capture_output=True, text=True, check=True).stdout
    used_the_file = "password=github_pat_x\n" in out
    del out  # whatever git returned must never reach a failure message
    assert used_the_file, "git did not read the token file"


def test_example_folders_never_run(tmp_path):
    for name in ("ridge", "_template", "my-model-1.0"):
        (tmp_path / "models" / name).mkdir(parents=True)
        (tmp_path / "models" / name / "model.yaml").write_text("name: x\n")
    assert dispatch.Dispatcher.models(None, tmp_path) == [Path("models/my-model-1.0")]


def as_test(d, monkeypatch):
    """The clone's model renamed models/test-extratrees; gather writes one row where it is told."""
    (d.root / "models" / "extratrees").rename(d.root / "models" / "test-extratrees")
    commit(d.root)
    d.gathered, d.checked, d.embed_tables = [], [], []
    d.todo = ["models/test-extratrees tfidf china-overseas-finance-inventory 0"]

    def gather(runs, results, only=None):
        d.gathered.append((runs, results, only))
        (results / f"{only}.csv").write_text("model,score\ntest-extratrees,0.5\n")

    monkeypatch.setattr("gather_results.gather", gather)
    monkeypatch.setattr(d, "check_results", lambda csv, code: d.checked.append(csv) or (True, "coverage ok"))
    monkeypatch.setattr(d, "missing_embeddings", lambda tables=None: d.embed_tables.append(tables) or [])


def test_a_test_folder_runs_every_step_and_publishes_nothing(clone, monkeypatch):
    d, name = clone, "test-extratrees"
    as_test(d, monkeypatch)
    d.run(fetch=False)
    assert state(d, name)["step"] == "smoke" and state(d, name)["pr"] == 3
    finish_smoke(d, name=name)
    d.run(fetch=False)
    assert state(d, name)["step"] == "run" and d.embed_tables == [dispatch.smoke_tables()]
    (path, _), = d.gh.posts  # the run-started comment is kept
    assert path == "/issues/3/comments"
    assert set((d.tmp / "runs.log").read_text().split()) == {str(d.cache / "test" / "runs")}

    d.todo = []
    d.run(fetch=False)
    d.run(fetch=False)
    results = d.cache / "test" / "results"
    assert state(d, name)["step"] == "tested" and "version" not in state(d, name)
    assert d.gathered == [(d.cache / "test" / "runs", results, name)] and d.checked == [results / f"{name}.csv"]
    assert d.published == [] and len(d.gh.posts) == 1  # no results pull request, no other message
    assert not (d.root / "results").exists() and not (d.root / "runs").exists()


def test_a_failed_test_notifies(clone, monkeypatch):
    d, name = clone, "test-extratrees"
    as_test(d, monkeypatch)
    monkeypatch.setattr(d, "check_results", lambda csv, code: (False, "PROBLEM times must be positive"))
    d.run(fetch=False)
    finish_smoke(d, name=name)
    d.run(fetch=False)
    d.todo = []
    d.run(fetch=False)
    assert state(d, name)["step"] == "failed" and d.gh.posts[-1][1]["event_type"] == "results-failed"


def test_test_tasks_are_fold_0_of_the_smoke_tables(tmp_path):
    d = dispatch.Dispatcher(root=tmp_path, cache=tmp_path / "cache", gh=FakeGitHub(), python=sys.executable)
    card = dict(CARD, tasks=["regression", "binary", "multiclass"])
    for name, strings in [("test-numeric", False), ("test-strings", True), ("full", False)]:
        (tmp_path / "models" / name).mkdir(parents=True)
        (tmp_path / "models" / name / "model.yaml").write_text(yaml.safe_dump(dict(card, end_to_end=strings)))
    numeric = d.missing_tasks(Path("models/test-numeric"), tmp_path)
    assert len(numeric) == 12 and {line.split()[2] for line in numeric} == set(dispatch.smoke_tables())
    assert {line.split()[3] for line in numeric} == {"0"} and {line.split()[1] for line in numeric} == \
        {"tfidf", "qwen-3-8b-first30", "llama-3.1-8b-first30", "minilm-l6-v2-pca30"}
    assert len(d.missing_tasks(Path("models/test-strings"), tmp_path)) == 3
    assert len(d.missing_tasks(Path("models/full"), tmp_path)) == 4 * 108 * 3
    row = d.cache / "test" / "runs" / "test-strings" / "none" / dispatch.smoke_tables()[0] / "0.json"
    row.parent.mkdir(parents=True)
    row.write_text("{}")  # a row in cache/test/runs counts for the test; one in runs/ would not
    assert len(d.missing_tasks(Path("models/test-strings"), tmp_path)) == 2
