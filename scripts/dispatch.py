"""The dispatcher: run every merged model without complete results, then open its results PR.

Meant for cron on the login node, every 15 minutes, from the clone on the cluster:

    */15 * * * * cd <clone> && PYTHONNOUSERSITE=1 cache/env/bin/python scripts/dispatch.py >> cache/dispatch/cron.log 2>&1

One pass: take the lock, fetch main, then move every model one step forward. The clone's working
copy is never pulled or changed. Models are read from a checkout of origin/main, and every job of a
model runs from a checkout of the commit recorded in its state when it started
(cache/dispatch/checkouts/<sha>), so neither a later push nor an edit in the clone changes the
code of a model in flight. A model is a folder of models/ on main, except the examples (EXAMPLES). Its steps:

  approve   the pull request that added the folder was merged, by gblayer or by the auto-merge
            gate with the label claude-approved (a folder pushed to main without a pull request was
            pushed by the maintainer).
  prepare   scripts/prepare.py downloads its checkpoint; its pins must match the shared env.
  smoke     run.py --smoke as a short job on the reference hardware (the reference GPU for GPU models).
  embed     numeric-only models: one array fills the missing language-model embeddings.
  run       make_tasks + the job array, submitted at most twice.
  publish   gather_results for this model only, check_results, a line in results/versions.yaml,
            build_site_data from the models already on the board, the paper's files and this one
            into the RESULTS_FILES on a branch results/<model>, with site/flags.json marking the
            model "leakage: possible" when its pull request has the label pretraining-overlap, push, open the results pull request (which goes through
            Checks, the Claude review and the gate like any other).
  published waits for that pull request; merged -> done, and the version is tagged.

A folder named test-<name> is a test of these steps, for the maintainer: its tasks are outer fold 0
of the three smoke tables, its rows and results go to cache/test/runs and cache/test/results, and
instead of publish it gathers and checks its results there and stops at step "tested". It never
touches results/, the board or a results pull request.

State lives in cache/dispatch/<model>.json. GitHub and git network calls are tried three times,
with a growing wait. A failure stops the model (step "failed") and sends ONE repository_dispatch
event (smoke-failed, run-failed, results-failed or runner-failed) that the Notify workflow turns
into a comment mentioning @gblayer; absolute paths in its text are cut to the file name. The one
message on success is a comment on the model's pull request when its run starts: the number of
tasks and a rough automatic estimate of when results will be ready. Delete the state file to retry. GitHub access, API and git, uses the
fine-grained token in $STRABLE_GH_TOKEN_FILE (default ~/.config/strable/github_token, mode 0600;
docs/automation.md, section 9), never the git credential store; it is never printed or logged.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from strable_leaderboard import APPROVED, EXAMPLES, FLAGS, MAINTAINER, OVERLAP, RESULTS_FILES, SLUG, TEST_PREFIX  # noqa: E402
from strable_leaderboard import REPO as DEFAULT_REPO  # noqa: E402

REPO = os.environ.get("STRABLE_REPO", DEFAULT_REPO)
REMOTE = f"https://github.com/{REPO}.git"  # explicit, so git always goes through the token's helper
FETCH_MAIN = (REMOTE, "+refs/heads/main:refs/remotes/origin/main")
GATE_BOT = "github-actions[bot]"
STOPPED = ("done", "failed", "unapproved", "closed", "tested")
VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(-[0-9A-Za-z.-]+)?$")  # v0.1.3, or a pre-release v0.1.0-pre
ELAPSED = re.compile(rb"^elapsed (\d+)s on (\w+)$", re.M)  # the last line of a run task's log (run_array.sh)
ABSOLUTE_PATH = re.compile(r"(?<![\w.:/~-])/(?:[\w.@+~-]+/)+([\w.@+~-]*)")
TOKEN_FILE = Path(os.environ.get("STRABLE_GH_TOKEN_FILE", "~/.config/strable/github_token")).expanduser()
# git reads the same file through this helper; the empty first value drops every other helper (the store).
GIT_HELPER = '!f() { test "$1" = get && printf "username=x-access-token\\npassword=%s\\n" "$(cat "$STRABLE_GH_TOKEN_FILE")"; }; f'


def log(*parts) -> None:
    print(datetime.now().isoformat(timespec="seconds"), *parts, flush=True)


def public(text: str, limit: int) -> str:
    """Text for a public comment: absolute paths cut to their last part, only the end kept."""
    return ABSOLUTE_PATH.sub(r".../\1", text)[-limit:]


def retry(call, tries: int = 3, wait: float = 10.0):
    """``call()``, tried again after network and server errors, waiting 10 s, then 20 s."""
    for attempt in range(tries):
        try:
            return call()
        except (OSError, subprocess.CalledProcessError) as exc:  # URLError and HTTPError are OSErrors
            refused = isinstance(exc, urllib.error.HTTPError) and exc.code < 500 and exc.code != 429
            if refused or attempt == tries - 1:
                raise
            log("retrying after", type(exc).__name__)
            time.sleep(wait * 2 ** attempt)


def token_file(path: Path = TOKEN_FILE) -> Path:
    """The token file, checked: present, and readable by its owner only."""
    if not path.is_file():
        sys.exit(f"no GitHub token at {path}: create the fine-grained token as in docs/automation.md, section 9")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        sys.exit(f"{path} is readable by others: chmod 600 {path}")
    return path


def duration(seconds: float) -> str:
    hours = seconds / 3600
    return f"{seconds / 60:.0f} minutes" if hours < 2 else f"{hours:.0f} hours" if hours < 48 else f"{hours / 24:.0f} days"


def smoke_tables() -> list[str]:
    """A test runs outer fold 0 of the three smoke tables."""
    from strable_leaderboard.smoke import SMOKE_TABLES

    return sorted(SMOKE_TABLES.values())


def write_flags(path: Path, name: str, st: dict) -> None:
    """site/flags.json with this model's entry set from its state, the other models' kept."""
    flags = json.loads(path.read_text()) if path.exists() else {}
    flags.pop(name, None)
    if st.get("leakage"):
        flags[name] = {"leakage": st["leakage"], "pr": st.get("pr")}
    path.write_text(json.dumps(flags, indent=1, sort_keys=True) + "\n")


def board_models(results: Path, site_json: Path, name: str) -> list[str]:
    """What the new board is built from: the models of the published site/results.json, the
    results files imported from the paper only, and the model being published. A pipeline id is
    ``[<encoder>+]<model>[:<variant>]``."""
    published = {p["id"].split("+")[-1].split(":")[0] for p in json.loads(site_json.read_text())["pipelines"]} \
        if site_json.exists() else set()
    paper = {p.stem for p in results.glob("*.csv")
             if (pd.read_csv(p, usecols=["hardware"])["hardware"].astype(str) == "paper").all()}
    return sorted(published | paper | {name})


class GitHub:
    """The few REST calls the dispatcher needs, with the fine-grained token of the token file."""

    def __init__(self, repo: str = REPO, path: Path = TOKEN_FILE):
        self.repo, self.path = repo, path

    def token(self) -> str:
        return token_file(self.path).read_text().strip()

    def call(self, method: str, path: str, data=None):
        req = urllib.request.Request(f"https://api.github.com/repos/{self.repo}{path}", method=method,
                                     data=json.dumps(data).encode() if data is not None else None)
        req.add_header("Authorization", f"Bearer {self.token()}")
        req.add_header("Accept", "application/vnd.github+json")

        def once():
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()

        body = retry(once)
        return json.loads(body) if body else None

    def get(self, path):
        return self.call("GET", path)

    def post(self, path, data):
        return self.call("POST", path, data)


class Dispatcher:
    def __init__(self, root: Path = ROOT, cache: Path | None = None, gh=None, python: str | None = None,
                 ref: str = "origin/main"):
        self.root, self.ref = Path(root), ref
        self.cache = Path(cache or os.environ.get("STRABLE_CACHE", self.root / "cache"))
        self.state_dir = self.cache / "dispatch"
        self.gh = gh or GitHub()
        self.python = python or str(self.cache / "env" / "bin" / "python")
        # Jobs read the code from their checkout (STRABLE_ROOT, set per submission) and keep their
        # outputs, the runner's flags and the cache in the clone.
        self.env = dict(os.environ, PYTHONNOUSERSITE="1", STRABLE_CACHE=str(self.cache), STRABLE_PYTHON=self.python,
                        STRABLE_LOGS=str(self.cache / "logs"), STRABLE_RUNS=str(self.root / "runs"),
                        STRABLE_RUNNER_ENV=str(self.root / "private" / "runner.env"))
        self.git_env = dict(self.env, STRABLE_GH_TOKEN_FILE=str(TOKEN_FILE), GIT_TERMINAL_PROMPT="0",
                            GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="credential.helper", GIT_CONFIG_VALUE_0="",
                            GIT_CONFIG_KEY_1="credential.helper", GIT_CONFIG_VALUE_1=GIT_HELPER)

    # ---- helpers -------------------------------------------------------------------------------
    def sh(self, *cmd, cwd=None, check=True, env=None) -> subprocess.CompletedProcess:
        return subprocess.run([str(c) for c in cmd], cwd=cwd or self.root, env=env or self.env, capture_output=True,
                              text=True, check=check)

    def git(self, *args, cwd=None, check=True) -> subprocess.CompletedProcess:
        return self.sh("git", *args, cwd=cwd, check=check, env=self.git_env)

    def state(self, name: str) -> dict:
        path = self.state_dir / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save(self, name: str, st: dict) -> None:
        st["updated"] = datetime.now().isoformat(timespec="seconds")
        path = self.state_dir / f"{name}.json"
        path.write_text(json.dumps(st, indent=1))

    def checkout(self, sha: str) -> Path:
        """A detached worktree of ``sha``, outside the clone's working copy."""
        path = self.state_dir / "checkouts" / sha
        if not path.exists():
            self.git("worktree", "add", "--detach", "--force", path, sha)
        return path

    def code(self, folder: Path, st: dict) -> Path:
        """The checkout of the model's commit."""
        return self.checkout(st["commit"])

    def models(self, head: Path) -> list[Path]:
        return sorted(Path("models", p.parent.name) for p in (head / "models").glob("*/model.yaml")
                      if SLUG.match(p.parent.name) and p.parent.name not in {"_template", *EXAMPLES})

    def outputs(self, name: str) -> tuple[Path, Path]:
        """(runs, results) of a model: the clone's, or cache/test/ for a test."""
        if name.startswith(TEST_PREFIX):
            return self.cache / "test" / "runs", self.cache / "test" / "results"
        return self.root / "runs", self.root / "results"

    def card(self, folder: Path, code: Path) -> dict:
        return yaml.safe_load((code / folder / "model.yaml").read_text())

    def submit(self, name: str, kind: str, lines: list[str], code: Path) -> str:
        tasks = self.state_dir / f"{name}.{kind}.txt"
        tasks.write_text("\n".join(lines) + "\n")
        (self.cache / "logs").mkdir(parents=True, exist_ok=True)
        env = dict(self.env, STRABLE_ROOT=str(code), STRABLE_RUNS=str(self.outputs(name)[0]))
        job = self.sh("bash", code / "scripts" / "run_array.sh", tasks, env=env).stdout.strip().split(";")[0]
        log(name, kind, f"{len(lines)} tasks, job {job}, code {code.name[:7]}")
        return job

    def running(self, job: str | None) -> bool:
        if not job:
            return False
        r = self.sh("squeue", "-h", "-j", job, "-o", "%i", check=False)
        return r.returncode == 0 and bool(r.stdout.strip())

    def missing_tasks(self, folder: Path, code: Path) -> list[str]:
        from make_tasks import tasks

        runs, results = self.outputs(folder.name)
        narrow = {"tables": smoke_tables(), "folds": [0]} if folder.name.startswith(TEST_PREFIX) else {}
        lines = tasks([str(code / folder)], runs=runs, results=results, **narrow)
        return [line.replace(f"{code}/", "", 1) for line in lines]  # folders relative to the checkout

    def prepare(self, folder: Path, code: Path) -> tuple[bool, str]:
        """Checkpoint download on the login node; fails when the pins do not match the shared env."""
        r = self.sh(self.python, "scripts/prepare.py", "--no-env", folder, cwd=code, check=False)
        return r.returncode == 0, r.stdout + r.stderr

    def check_results(self, csv: Path, code: Path) -> tuple[bool, str]:
        r = self.sh(self.python, "-m", "strable_leaderboard.check_results", csv, cwd=code, check=False)
        return r.returncode == 0, r.stdout + r.stderr

    def task_seconds(self, device: str, last: int = 2000) -> list[int]:
        """Wall-clock seconds of the latest run tasks on ``device`` (cpu or cuda), from their logs."""
        times = []
        for path in sorted((self.cache / "logs").glob("*.out"), key=lambda p: p.stat().st_mtime)[-last:]:
            with open(path, "rb") as f:
                f.seek(max(0, path.stat().st_size - 200))
                times += [int(s) for s, dev in ELAPSED.findall(f.read()) if dev.decode() == device]
        return times

    def estimate(self, job: str, n: int, device: str) -> str:
        """When the results should be ready: the run tasks queued ahead plus this run's, in waves of
        the array's throttle, each wave as long as the median task observed on this device."""
        times = self.task_seconds(device)
        if not times:
            return "No task time has been observed on this hardware yet, so there is no estimate."
        queue = self.sh("squeue", "-h", "-r", "-u", getpass.getuser(), "-n", "strable-run", "-o", "%i", check=False)
        ahead = sum(1 for task in queue.stdout.split() if task.split("_")[0] != job)
        shown = self.sh("scontrol", "show", "job", job, check=False).stdout
        throttle = re.search(r"ArrayTaskThrottle=(\d+)", shown) or re.search(r"ArrayTaskId=\S*%(\d+)", shown)
        slots = int(throttle.group(1)) if throttle else 1
        median = sorted(times)[len(times) // 2]
        seconds = math.ceil((ahead + n) / slots) * median
        ready = datetime.fromtimestamp(time.time() + seconds).astimezone().strftime("%a %d %b %Y, %H:00 %Z")
        return (f"Results expected around {ready}, in about {duration(seconds)} (rough automatic estimate: "
                f"{ahead} tasks queued ahead, {slots} at a time, {duration(median)} per task, the median "
                f"observed on this hardware).")

    def announce(self, st: dict, job: str, n: int, device: str) -> None:
        """The one success comment: on the model's pull request, when its run starts."""
        if not st.get("pr"):
            return
        try:
            body = f"The benchmark of `{st['model']}` started: {n} tasks on the reference hardware. " + \
                self.estimate(job, n, device)
            self.gh.post(f"/issues/{st['pr']}/comments", {"body": body})
        except Exception as exc:  # a missed comment never stops the run
            log("announce failed:", type(exc).__name__)

    def missing_embeddings(self, tables=None) -> list[str]:
        from strable_leaderboard.embed import missing

        return missing(self.cache, tables=tables)

    def notify(self, kind: str, st: dict, reason: str) -> None:
        """One repository_dispatch event; the Notify workflow comments as github-actions[bot]."""
        payload = {"pr": st.get("pr"), "model": st.get("model"), "reason": public(reason, 1500)}
        try:
            self.gh.post("/dispatches", {"event_type": kind, "client_payload": payload})
        except Exception as exc:  # never let a notification failure stop the pass
            log("notify failed:", type(exc).__name__)
        log("notified", kind, payload["model"])

    def fail(self, name: str, st: dict, kind: str, reason: str) -> None:
        st.update(step="failed", failure=kind, reason=reason)
        self.save(name, st)
        self.notify(kind, st, reason)

    # ---- approval --------------------------------------------------------------------------------
    def approval(self, folder: Path) -> tuple[bool, int | None, str]:
        """(approved, pull request number, why)."""
        shas = self.git("log", "--format=%H", "--diff-filter=A", self.ref, "--", folder / "model.yaml").stdout.split()
        if not shas:
            return False, None, "model.yaml not found in main's history"
        pulls = [p for p in self.gh.get(f"/commits/{shas[-1]}/pulls") or [] if p.get("merged_at")]
        if not pulls:
            return True, None, "pushed to main by the maintainer, no pull request"
        pr = self.gh.get(f"/pulls/{pulls[0]['number']}")
        merged_by = (pr.get("merged_by") or {}).get("login")
        labels = {label["name"] for label in pr.get("labels", [])}
        if merged_by == MAINTAINER:
            return True, pr["number"], "merged by the maintainer"
        if merged_by == GATE_BOT and APPROVED in labels:
            return True, pr["number"], "auto-merged with claude-approved"
        return False, pr["number"], f"merged by {merged_by} without {APPROVED}"

    # ---- one model, one step ---------------------------------------------------------------------
    def advance(self, folder: Path, head: str) -> None:
        name = folder.name
        st = self.state(name)
        step = st.get("step")
        if step in STOPPED:
            return
        if step is None:
            st = {"model": name, "folder": str(folder), "commit": head}
            if not self.missing_tasks(folder, self.code(folder, st)):
                return
            ok, pr, why = self.approval(folder)
            st.update(pr=pr, approval=why)
            if not ok:
                st["step"] = "unapproved"
                self.save(name, st)
                log(name, "not approved:", why)
                return
            step = "prepare"
        code = self.code(folder, st)
        if step == "prepare":
            ok, output = self.prepare(folder, code)
            if not ok:
                return self.fail(name, st, "smoke-failed", "prepare: " + output[-1200:])
            (self.state_dir / "smoke" / f"{name}.rc").unlink(missing_ok=True)  # no stale verdict
            st.update(step="smoke", job=self.submit(name, "smoke", [f"smoke {folder}"], code))
            return self.save(name, st)
        if self.running(st.get("job")):
            return
        if step == "smoke":
            rc_file = self.state_dir / "smoke" / f"{name}.rc"
            rc = rc_file.read_text().strip() if rc_file.exists() else "missing"
            if rc != "0":
                log_file = self.state_dir / "smoke" / f"{name}.log"
                tail = "\n".join(log_file.read_text().splitlines()[-12:]) if log_file.exists() else "no log"
                return self.fail(name, st, "smoke-failed", f"smoke test exit {rc} on the reference hardware:\n{tail}")
            step = "embed"
        if step == "embed":
            if not self.card(folder, code).get("handles_strings"):
                todo = self.missing_embeddings(smoke_tables() if name.startswith(TEST_PREFIX) else None)
                if todo:
                    shared = self.state("_embed")
                    if self.running(shared.get("job")):
                        st.update(step="embed", job=shared["job"])
                        return self.save(name, st)
                    if st.get("embed_tries", 0) >= 2:
                        return self.fail(name, st, "run-failed", f"embedding cache still misses {len(todo)} tables")
                    job = self.submit("_embed", "embed", todo, code)
                    self.save("_embed", {"job": job})
                    st.update(step="embed", job=job, embed_tries=st.get("embed_tries", 0) + 1)
                    return self.save(name, st)
            step = "run"
        if step == "run":
            todo = self.missing_tasks(folder, code)
            if not todo:
                step = "publish"
            elif st.get("submissions", 0) >= 2:
                return self.fail(name, st, "run-failed", f"{len(todo)} tasks have no result after two submissions; "
                                 f"first: {todo[0]}; logs in cache/logs/{st.get('job')}_*.out")
            else:
                first = not st.get("submissions")
                st.update(step="run", job=self.submit(name, "run", todo, code), submissions=st.get("submissions", 0) + 1,
                          n_tasks=st.get("n_tasks") or len(todo))
                self.save(name, st)
                if first:
                    gpu = self.card(folder, code)["hardware"] == "gpu"
                    self.announce(st, st["job"], len(todo), "cuda" if gpu else "cpu")
                return
        if step == "publish":
            st["step"] = "publish"
            self.save(name, st)
            return self.tested(folder, st, code) if name.startswith(TEST_PREFIX) else self.publish(folder, st, code)
        if step == "published":
            pr = self.gh.get(f"/pulls/{st['results_pr']}")
            if pr.get("merged_at"):
                self.tag(st.get("version"), pr.get("merge_commit_sha"))
                st["step"] = "done"
            elif pr.get("state") == "closed":
                st["step"] = "closed"
            return self.save(name, st)

    # ---- publish ---------------------------------------------------------------------------------
    def results_pr_open(self) -> bool:
        for path in self.state_dir.glob("*.json"):
            if json.loads(path.read_text()).get("step") == "published":
                return True
        return False

    def next_version(self, versions: list) -> str:
        """The next patch; after a pre-release such as v0.1.0-pre, its release, v0.1.0."""
        if not versions:
            return "v0.1.0"
        m = VERSION.match(str(versions[0]["version"]))
        if not m:
            raise ValueError(f"results/versions.yaml: unreadable version {versions[0]['version']!r}")
        major, minor, patch = (int(x) for x in m.groups()[:3])
        return f"v{major}.{minor}.{patch if m.group(4) else patch + 1}"

    def tested(self, folder: Path, st: dict, code: Path) -> None:
        """A test's last step: gather and check its results in cache/test/results; nothing is published."""
        from gather_results import gather

        name = folder.name
        runs, results = self.outputs(name)
        results.mkdir(parents=True, exist_ok=True)
        gather(runs, results, only=name)
        ok, check_output = self.check_results(results / f"{name}.csv", code)
        if not ok:
            return self.fail(name, st, "results-failed", "check_results:\n" + check_output[-1200:])
        st["step"] = "tested"
        self.save(name, st)
        log(name, f"tested: {len(pd.read_csv(results / f'{name}.csv'))} rows in {results / name}.csv, check_results ok")

    def publish(self, folder: Path, st: dict, code: Path) -> None:
        name = folder.name
        if self.results_pr_open():
            return log(name, "waits: another results pull request is open")
        from gather_results import gather

        results = self.root / "results"
        results.mkdir(exist_ok=True)
        csv, versions_file = results / f"{name}.csv", results / "versions.yaml"
        before = {p: p.read_bytes() if p.exists() else None for p in (csv, versions_file)}

        def undo():  # results/ keeps only what a results pull request published
            for path, data in before.items():
                if data is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(data)

        gather(self.root / "runs", results, only=name)
        ok, check_output = self.check_results(csv, code)
        if not ok:
            undo()
            return self.fail(name, st, "results-failed", "check_results:\n" + check_output[-1200:])
        card = self.card(folder, code)
        labels = {label["name"] for label in self.gh.get(f"/pulls/{st['pr']}").get("labels", [])} if st.get("pr") else set()
        st["leakage"] = "possible" if OVERLAP in labels else None
        versions = yaml.safe_load(before[versions_file] or b"") or []
        version = self.next_version(versions)
        note = f"Added {card['name']} {card['version']}"
        versions_file.write_text(yaml.safe_dump([{"version": version, "date": date.today().isoformat(), "note": note}]
                                                + versions, sort_keys=False))
        try:
            number = self.open_results_pr(name, version, note, check_output, st)
        except Exception as exc:
            undo()
            detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            return self.fail(name, st, "results-failed", f"publishing failed: {type(exc).__name__}: {str(detail)[-600:]}")
        st.update(step="published", results_pr=number, version=version)
        self.save(name, st)
        log(name, "results pull request", number)

    def open_results_pr(self, name: str, version: str, note: str, check_output: str, st: dict) -> int:
        """Branch results/<name> from origin/main with the rebuilt RESULTS_FILES, pushed; the PR opened."""
        branch, wt = f"results/{name}", self.state_dir / "worktrees" / name
        retry(lambda: self.git("fetch", "-q", *FETCH_MAIN))
        self.git("worktree", "remove", "--force", wt, check=False)
        self.git("worktree", "add", "--force", "-B", branch, wt, "origin/main")
        try:
            write_flags(wt / FLAGS, name, st)
            models = board_models(self.root / "results", wt / "site" / "results.json", name)
            # Built with the scripts of origin/main, so the file matches what Checks will validate.
            self.sh(self.python, "scripts/build_site_data.py", "--results", self.root / "results", "--models", *models,
                    cwd=wt)
            self.sh(self.python, "scripts/build_site_data.py", "--check", "site/results.json", cwd=wt)
            self.git("add", *RESULTS_FILES, cwd=wt)
            self.git("commit", "-m", f"Results: {note} ({version})", cwd=wt)
            retry(lambda: self.git("push", "--force", REMOTE, f"HEAD:refs/heads/{branch}", cwd=wt))
        finally:
            self.git("worktree", "remove", "--force", wt, check=False)
        model_pr = f"Model pull request: #{st['pr']}.\n" if st.get("pr") else ""
        if st.get("leakage"):
            model_pr += f"Pre-training overlap: {st['leakage']} (label {OVERLAP} on #{st['pr']}); the board tags its rows.\n"
        body = (f"{note}: {st.get('n_tasks', '?')} tasks run on the reference hardware by the dispatcher.\n"
                f"{model_pr}\n```\n{public(check_output.strip(), 3000)}\n```\n")
        pr = self.gh.post("/pulls", {"title": f"Results: {note} ({version})", "head": branch, "base": "main",
                                     "body": body})
        return pr["number"]

    def tag(self, version: str | None, sha: str | None) -> None:
        if version and sha:
            self.git("fetch", "-q", *FETCH_MAIN, check=False)
            if self.git("tag", version, sha, check=False).returncode == 0:
                retry(lambda: self.git("push", REMOTE, version))

    def prune(self, head: str) -> None:
        """Remove the checkouts that neither main's head nor a model in flight uses."""
        states = [json.loads(p.read_text()) for p in self.state_dir.glob("*.json")]
        keep = {head} | {st.get("commit") for st in states if st.get("step") not in STOPPED}
        for path in (self.state_dir / "checkouts").glob("*"):
            if path.name not in keep:
                self.git("worktree", "remove", "--force", path, check=False)

    # ---- the pass --------------------------------------------------------------------------------
    def run(self, fetch: bool = True) -> int:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_dir / ".lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log("another pass holds the lock")
                return 0
            if fetch:
                try:
                    retry(lambda: self.git("fetch", "-q", *FETCH_MAIN))
                except subprocess.CalledProcessError as exc:
                    self.notify("runner-failed", {}, "git fetch of main failed:\n" + exc.stderr[-800:])
                    return 1
            head = self.git("rev-parse", self.ref).stdout.strip()
            for folder in self.models(self.checkout(head)):
                try:
                    self.advance(folder, head)
                except Exception as exc:
                    st = self.state(folder.name) or {"model": folder.name, "folder": str(folder)}
                    self.fail(folder.name, st, "runner-failed", f"{type(exc).__name__}: {str(exc)[-800:]}")
            self.prune(head)
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-fetch", action="store_true", help="use origin/main as it is (tests, manual passes)")
    args = parser.parse_args(argv)
    token_file()  # fail at once, with the fix, when the token is missing or exposed
    return Dispatcher().run(fetch=not args.no_fetch)


if __name__ == "__main__":
    sys.exit(main())
