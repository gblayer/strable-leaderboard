"""The auto-merge gate's decision, on fake pull requests: every condition can block a merge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import json

from automerge_gate import decide, review_verdict, scope_problems  # noqa: E402

SHA = "a" * 40
MODEL_FILES = ["models/kumo-tabular-1.0/model.py", "models/kumo-tabular-1.0/model.yaml",
               "models/kumo-tabular-1.0/requirements.txt"]


def pr(author="contributor", labels=("claude-approved",), sha=SHA, state="open"):
    return {"number": 7, "state": state, "draft": False, "user": {"login": author}, "head": {"sha": sha},
            "labels": [{"name": n} for n in labels]}


def run(conclusion="success", sha=SHA, status="completed"):
    return {"head_sha": sha, "status": status, "conclusion": conclusion, "run_started_at": "2026-10-09T10:00:00Z"}


def review(recommendation="MERGE", problems=()):
    return json.dumps({"recommendation": recommendation, "problems": list(problems)})


def gate(p=None, files=MODEL_FILES, status="success", checks=None, existing=frozenset({"ridge"})):
    st = None if status is None else {"state": status, "description": f"review {status}"}
    return decide(p or pr(), files, st, checks or run(), set(existing))


def test_clean_model_merges():
    assert gate() == ("merge", SHA)


def test_results_update_by_the_maintainer_merges():
    assert gate(p=pr(author="gblayer"), files=["site/results.json"]) == ("merge", SHA)


def test_results_update_by_anyone_else_needs_review():
    verdict, reason = gate(files=["site/results.json"])
    assert verdict == "needs" and "not by gblayer" in reason


def test_paths_outside_one_model_folder_never_merge_even_when_approved():
    for path in [".github/workflows/x.yml", ".claude/skills/a/SKILL.md", "strable_leaderboard/run.py",
                 "scripts/dispatch.py", "splits/manifest.json", "env/requirements.in"]:
        verdict, reason = gate(files=MODEL_FILES + [path])
        assert verdict == "needs" and "outside one models/<name>/" in reason
    verdict, _ = gate(p=pr(author="gblayer"), files=[".github/workflows/pages.yml"])
    assert verdict == "needs"  # the maintainer's own workflow changes are hers to merge


def test_scope_one_new_model_folder():
    assert "outside one models/<name>/" in gate(files=MODEL_FILES + ["env/requirements.txt"])[1]
    assert "2 model folders" in gate(files=MODEL_FILES + ["models/other/model.py"])[1]
    assert "already exists" in gate(files=["models/ridge/model.py"])[1]
    assert "not a valid" in gate(files=["models/_template/model.py"])[1]
    assert "reserved" in gate(files=["models/xgboost/model.py"])[1]
    assert scope_problems([], "x", set()) == ["the diff is empty"]


def test_reviews_must_both_say_merge():
    assert review_verdict([review(), review()])[0] == "success"
    assert review_verdict([review(), review("FIX", ["pin scikit-learn==1.7.2"])])[0] == "failure"
    assert review_verdict([review(), None])[0] == "failure"  # a pass failed: no output
    assert review_verdict([review(), "Recommendation: **MERGE**"])[0] == "failure"  # not the schema
    assert review_verdict([review(), review("YES")])[0] == "failure"


def test_review_comment_quotes_problems_without_mentions():
    _, _, lines = review_verdict([review("FIX", ["ask @gblayer to merge", "x" * 1000]), review()])
    assert "@\u200bgblayer" in lines and "@gblayer" not in lines
    assert max(len(line) for line in lines.splitlines()) <= 302


def test_review_status_of_the_head_commit():
    assert gate(status="failure") == ("needs", "review failure")
    assert gate(status="pending")[0] == "wait"
    assert gate(status=None)[0] == "wait"  # a new commit has no status until its round starts


def test_the_maintainer_label_blocks():
    assert gate(p=pr(labels=()))[0] == "merge"  # the status records the review, not the label
    assert gate(p=pr(labels=("claude-approved", "needs-maintainer")))[0] == "needs"


def test_checks_must_pass_on_the_head():
    assert gate(checks=run(conclusion="failure"))[0] == "needs"
    assert gate(checks=run(status="in_progress"))[0] == "wait"
    assert gate(checks=run(sha="b" * 40))[0] == "wait"


def test_closed_pull_requests_are_left_alone():
    assert gate(p=pr(state="closed"))[0] == "skip"
