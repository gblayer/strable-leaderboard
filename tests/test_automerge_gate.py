"""The auto-merge gate's decision, on fake pull requests: every condition can block a merge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from automerge_gate import decide, scope_problems  # noqa: E402

SHA = "a" * 40
MODEL_FILES = ["models/extratrees/model.py", "models/extratrees/model.yaml", "models/extratrees/requirements.txt"]


def pr(author="contributor", labels=("claude-approved",), sha=SHA, state="open"):
    return {"number": 7, "state": state, "draft": False, "user": {"login": author}, "head": {"sha": sha},
            "labels": [{"name": n} for n in labels]}


def run(conclusion="success", sha=SHA, status="completed"):
    return {"head_sha": sha, "status": status, "conclusion": conclusion, "run_started_at": "2026-10-09T10:00:00Z"}


def comments(one="MERGE", two="agrees with 1/2: **MERGE**", author="github-actions[bot]", at="2026-10-09T10:05:00Z"):
    return [
        {"user": {"login": author}, "created_at": at,
         "body": f"**Claude review 1/2 · Add ExtraTrees**\nRecommendation: **{one}**\nScope ✓"},
        {"user": {"login": author}, "created_at": at, "body": f"**Claude review 2/2** · {two}"},
    ]


def gate(p=None, files=MODEL_FILES, cs=None, review=None, checks=None, existing=frozenset({"ridge"})):
    return decide(p or pr(), files, comments() if cs is None else cs, review or run(), checks or run(), set(existing))


def test_clean_model_merges():
    assert gate() == ("merge", SHA)


def test_results_update_by_the_maintainer_merges():
    assert gate(p=pr(author="gblayer"), files=["site/results.json"]) == ("merge", SHA)


def test_results_update_by_anyone_else_needs_review():
    verdict, reason = gate(files=["site/results.json"])
    assert verdict == "needs" and "not by gblayer" in reason


def test_protected_paths_never_merge_even_when_approved():
    for path in [".github/workflows/x.yml", ".claude/skills/a/SKILL.md", "strable_leaderboard/run.py",
                 "scripts/dispatch.py", "splits/manifest.json"]:
        verdict, reason = gate(files=MODEL_FILES + [path])
        assert verdict == "needs" and "protected" in reason
    verdict, _ = gate(p=pr(author="gblayer"), files=[".github/workflows/pages.yml"])
    assert verdict == "needs"  # the maintainer's own workflow changes are hers to merge


def test_scope_one_new_model_folder():
    assert "outside one models/<name>/" in gate(files=MODEL_FILES + ["env/requirements.txt"])[1]
    assert "2 model folders" in gate(files=MODEL_FILES + ["models/other/model.py"])[1]
    assert "already exists" in gate(files=["models/ridge/model.py"])[1]
    assert "not a valid" in gate(files=["models/_template/model.py"])[1]
    assert scope_problems([], "x", set()) == ["the diff is empty"]


def test_reviews_must_both_say_merge():
    assert gate(cs=comments(one="FIX"))[0] == "needs"
    assert gate(cs=comments(two="disagrees with 1/2: missed X. Recommendation: **FIX**"))[0] == "needs"
    assert gate(cs=[])[0] == "needs"


def test_review_comments_from_humans_or_older_rounds_do_not_count():
    assert gate(cs=comments(author="contributor"))[0] == "needs"
    assert gate(cs=comments(at="2026-10-09T09:00:00Z"))[0] == "needs"  # before this commit's review run


def test_labels():
    assert gate(p=pr(labels=()))[0] == "needs"
    assert gate(p=pr(labels=("claude-approved", "needs-maintainer")))[0] == "needs"


def test_checks_must_pass_on_the_head():
    assert gate(checks=run(conclusion="failure"))[0] == "needs"
    assert gate(checks=run(status="in_progress"))[0] == "wait"
    assert gate(checks=run(sha="b" * 40))[0] == "wait"


def test_head_moved_since_the_review_waits_for_the_new_review():
    assert gate(review=run(sha="b" * 40))[0] == "wait"
    assert gate(review=run(status="in_progress"))[0] == "wait"
    assert gate(review=run(conclusion="failure"))[0] == "needs"


def test_closed_pull_requests_are_left_alone():
    assert gate(p=pr(state="closed"))[0] == "skip"
