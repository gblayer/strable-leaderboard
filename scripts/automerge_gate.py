"""The auto-merge gate: merge a pull request only when every condition holds, in code, not by Claude.

    python scripts/automerge_gate.py gate --sha <head sha>     # after "Claude review" or "Checks" completes
    python scripts/automerge_gate.py triage --pr <number>      # on every pull request event: scope only
    python scripts/automerge_gate.py reset --pr <number>       # a new commit: drop the bot's old verdict labels

Run by .github/workflows/automerge.yml and claude-review.yml with the built-in GITHUB_TOKEN, on
the base branch: the pull request's code is never checked out, only read through the API.

A pull request is merged (`gh pr merge --merge --match-head-commit <sha>`) only if
1. both Claude review comments of the review run on the current head say MERGE, and review 2/2
   added the label claude-approved (and nobody added needs-maintainer);
2. the Checks workflow succeeded on the current head;
3. the diff touches exactly one new models/<name>/ folder, or only site/results.json in a pull
   request opened by gblayer;
4. nothing under .github/, .claude/, strable_leaderboard/, scripts/, splits/ changes;
5. the head did not move since the reviews (the merge itself also pins the head).
A merge that touches site/ then starts the Pages workflow, since a merge made with GITHUB_TOKEN
starts no push workflow. Otherwise the gate labels needs-maintainer and comments once per commit:
"@gblayer your review is needed: <reason>". While the reviews or Checks still run, it waits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

MAINTAINER = "gblayer"
BOTS = {"github-actions[bot]", "claude[bot]"}
PROTECTED = (".github/", ".claude/", "strable_leaderboard/", "scripts/", "splits/")
SLUG = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
APPROVED, NEEDS = "claude-approved", "needs-maintainer"
REVIEW_WORKFLOW, CHECKS_WORKFLOW = "claude-review.yml", "checks.yml"


def scope_problems(paths: list[str], author: str, existing_models: set[str]) -> list[str]:
    """Conditions 3 and 4 on every path the pull request adds, changes, removes or renames."""
    out = []
    protected = sorted(p for p in paths if p.startswith(PROTECTED))
    if protected:
        out.append("it changes protected paths: " + ", ".join(f"`{p}`" for p in protected[:5]))
    if set(paths) == {"site/results.json"}:
        if author != MAINTAINER:
            out.append(f"a results update opened by {author}, not by {MAINTAINER}")
        return out
    outside = sorted(p for p in paths if not (p.startswith("models/") and p.count("/") >= 2))
    if outside and not protected:
        out.append("it changes files outside one models/<name>/ folder: " + ", ".join(f"`{p}`" for p in outside[:5]))
    folders = sorted({p.split("/")[1] for p in paths if p.startswith("models/") and p.count("/") >= 2})
    if len(folders) > 1:
        out.append(f"it touches {len(folders)} model folders")
    for name in folders:
        if not SLUG.match(name) or name == "_template":
            out.append(f"`models/{name}/` is not a valid model folder name")
        elif name in existing_models:
            out.append(f"`models/{name}/` already exists on main: a change to a merged model")
    if not paths:
        out.append("the diff is empty")
    return out


def review_verdicts(comments: list[dict], since: str) -> tuple[str | None, str | None]:
    """The latest bot-written 1/2 and 2/2 comments posted after ``since`` (the review run's start)."""
    one = two = None
    for c in sorted(comments, key=lambda c: c["created_at"]):
        if c["user"]["login"] not in BOTS or c["created_at"] < since:
            continue
        body = c["body"]
        if "Claude review 1/2" in body:
            one = body
        elif "Claude review 2/2" in body:
            two = body
    return one, two


def says_merge(one: str | None, two: str | None) -> bool:
    return bool(one and two and "Recommendation: **MERGE**" in one and "agrees with 1/2" in two
                and "disagrees" not in two and "**MERGE**" in two)


def decide(pr: dict, paths: list[str], comments: list[dict], review_run: dict | None, checks_run: dict | None,
           existing_models: set[str]) -> tuple[str, str]:
    """("merge" | "wait" | "needs" | "skip", reason)."""
    if pr["state"] != "open":
        return "skip", "closed"
    if pr.get("draft"):
        return "wait", "draft"
    scope = scope_problems(paths, pr["user"]["login"], existing_models)
    if scope:
        return "needs", "; ".join(scope)
    sha = pr["head"]["sha"]
    if not review_run or review_run["head_sha"] != sha or review_run["status"] != "completed":
        return "wait", "the Claude review of this commit has not finished"
    if review_run["conclusion"] != "success":
        return "needs", f"the Claude review workflow ended with {review_run['conclusion']}"
    if not checks_run or checks_run["head_sha"] != sha or checks_run["status"] != "completed":
        return "wait", "Checks on this commit have not finished"
    if checks_run["conclusion"] != "success":
        return "needs", f"Checks ended with {checks_run['conclusion']} on {sha[:7]}"
    one, two = review_verdicts(comments, review_run["run_started_at"])
    if not says_merge(one, two):
        return "needs", "the two Claude reviews of this commit do not both say MERGE"
    labels = {label["name"] for label in pr["labels"]}
    if APPROVED not in labels:
        return "needs", f"review 2/2 did not add {APPROVED}"
    if NEEDS in labels:
        return "needs", f"the label {NEEDS} is set"
    return "merge", sha


# ---- GitHub through the gh CLI ---------------------------------------------------------------------
REPO = os.environ.get("GITHUB_REPOSITORY", "gblayer/strable-leaderboard")


def gh(*args, check=True) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check).stdout


def api(path: str, paginate: bool = False):
    out = gh("api", *(["--paginate", "--slurp"] if paginate else []), f"repos/{REPO}/{path}")
    data = json.loads(out)
    return [x for page in data for x in page] if paginate else data


def latest_run(workflow: str, sha: str) -> dict | None:
    runs = api(f"actions/workflows/{workflow}/runs?head_sha={sha}&per_page=20")["workflow_runs"]
    return max(runs, key=lambda r: r["run_started_at"], default=None)


def existing_models() -> set[str]:
    return {item["name"] for item in api("contents/models?ref=main") if item["type"] == "dir"}


def changed_paths(number: int) -> list[str]:
    files = api(f"pulls/{number}/files?per_page=100", paginate=True)
    return [f["filename"] for f in files] + [f["previous_filename"] for f in files if f.get("previous_filename")]


def needs_maintainer(pr: dict, reason: str) -> None:
    number, marker = pr["number"], f"<!-- automerge-gate {pr['head']['sha']} -->"
    gh("pr", "edit", str(number), "--repo", REPO, "--add-label", NEEDS)
    if any(marker in c["body"] for c in api(f"issues/{number}/comments?per_page=100", paginate=True)):
        return  # already said for this commit
    gh("pr", "comment", str(number), "--repo", REPO, "--body", f"@{MAINTAINER} your review is needed: {reason}\n\n{marker}")


def gate(pr: dict) -> str:
    verdict, reason = decide(pr, changed_paths(pr["number"]),
                             api(f"issues/{pr['number']}/comments?per_page=100", paginate=True),
                             latest_run(REVIEW_WORKFLOW, pr["head"]["sha"]), latest_run(CHECKS_WORKFLOW, pr["head"]["sha"]),
                             existing_models())
    print(f"#{pr['number']} {pr['head']['sha'][:7]}: {verdict} ({reason})")
    if verdict == "needs":
        needs_maintainer(pr, reason)
    elif verdict == "merge":
        merged = subprocess.run(["gh", "pr", "merge", str(pr["number"]), "--repo", REPO, "--merge",
                                 "--match-head-commit", reason], capture_output=True, text=True)
        if merged.returncode:
            needs_maintainer(pr, f"the merge failed: {merged.stderr.strip()[-300:]}")
            return "needs"
        if any(p.startswith("site/") for p in changed_paths(pr["number"])):
            gh("workflow", "run", "pages.yml", "--repo", REPO, "--ref", "main")
    return verdict


def reset(number: int) -> None:
    """A new commit: remove claude-approved and needs-maintainer when a bot set them, so that the new
    review round decides afresh. A label the maintainer set by hand stays."""
    events = api(f"issues/{number}/events?per_page=100", paginate=True)
    last = {}
    for e in events:
        if e["event"] == "labeled":
            last[e["label"]["name"]] = e["actor"]["login"]
    labels = {label["name"] for label in api(f"issues/{number}")["labels"]}
    for name in (APPROVED, NEEDS):
        if name in labels and last.get(name) in BOTS:
            gh("pr", "edit", str(number), "--repo", REPO, "--remove-label", name)
            print(f"#{number}: removed {name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["gate", "triage", "reset"])
    parser.add_argument("--pr", type=int)
    parser.add_argument("--sha")
    args = parser.parse_args(argv)
    if args.mode == "reset":
        reset(args.pr)
        return 0
    if args.mode == "triage":
        pr = api(f"pulls/{args.pr}")
        scope = scope_problems(changed_paths(pr["number"]), pr["user"]["login"], existing_models())
        if pr["state"] == "open" and scope:
            needs_maintainer(pr, "; ".join(scope))
        print(f"#{pr['number']}: scope " + ("; ".join(scope) if scope else "ok"))
        return 0
    prs = [p for p in api("pulls?state=open&per_page=100", paginate=True) if p["head"]["sha"] == args.sha]
    if not prs:
        print(f"no open pull request has head {args.sha}")
    for pr in prs:
        gate(api(f"pulls/{pr['number']}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
