"""The auto-merge gate: merge a pull request only when every condition holds, in code, not by Claude.

    python scripts/automerge_gate.py reset --pr N --sha S     # a review round starts: bot labels off, status pending
    python scripts/automerge_gate.py verdict --pr N --sha S   # it ends: comment, label, status, then the gate
    python scripts/automerge_gate.py gate --sha S             # Checks completed: the gate
    python scripts/automerge_gate.py triage --pr N            # every pull request event: scope only
    python scripts/automerge_gate.py inputs --pr N --sha S   # what the review round needs, as job outputs

The conditions are written once, in docs/automation.md, section 3. Run by claude-review.yml and
automerge.yml with the built-in GITHUB_TOKEN, on the base branch: the pull request's code is never
checked out, only read through the API. ``verdict`` reads the two review passes' structured
outputs from $REVIEW_ONE and $REVIEW_TWO and records them as the commit status "Claude review" of
the reviewed commit, so a review of an old commit never counts for a newer head. A pull request
that changes env/requirements.in waits for that round, whose env job reports what the change does
to the shared environment ($ENV_REPORT); the "your review is needed" comment carries the report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strable_leaderboard import APPROVED, MAINTAINER, NEEDS, OVERLAP, RESERVED, RESULTS_FILES, SLUG  # noqa: E402
from strable_leaderboard import REPO as DEFAULT_REPO  # noqa: E402

BOTS = {"github-actions[bot]", "claude[bot]"}
CHECKS_WORKFLOW = "checks.yml"
REVIEW_CONTEXT = "Claude review"
ENV_INPUT = "env/requirements.in"  # a contributor may add packages here; the maintainer reviews every change
# The sites the leakage pass may fetch, plus the hosts of the card's code and paper links.
WEB = ("arxiv.org", "huggingface.co", "github.com", "raw.githubusercontent.com")
HOST = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")
CARD_LINK = re.compile(r"^(?:code|paper):\s*[\"']?(https?://[^\s\"'#]+)", re.M)
REPO = os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO)


def scope_problems(paths: list[str], author: str, existing_models: set[str]) -> list[str]:
    """Conditions 3 and 4 on every path the pull request adds, changes, removes or renames."""
    if not paths:
        return ["the diff is empty"]
    if "site/results.json" in paths and set(paths) <= set(RESULTS_FILES):
        return [] if author == MAINTAINER else [f"a results update opened by {author}, not by {MAINTAINER}"]
    out = []
    outside = sorted(p for p in paths if not (p.startswith("models/") and p.count("/") >= 2))
    if outside:
        out.append("it changes files outside one models/<name>/ folder: " + ", ".join(f"`{p}`" for p in outside[:5]))
    folders = sorted({p.split("/")[1] for p in paths if p.startswith("models/") and p.count("/") >= 2})
    if len(folders) > 1:
        out.append(f"it touches {len(folders)} model folders")
    for name in folders:
        if not SLUG.match(name) or name == "_template":
            out.append(f"`models/{name}/` is not a valid model folder name")
        elif name in existing_models:
            out.append(f"`models/{name}/` already exists on main: a change to a merged model")
        elif name in RESERVED:
            out.append(f"`models/{name}/` is reserved for a learner of the paper")
    return out


def web_tools(card_text: str) -> str:
    """The leakage pass's WebFetch permissions: the fixed sites and the hosts of the card's code and paper."""
    hosts = set(WEB) | {urlsplit(url).hostname or "" for url in CARD_LINK.findall(card_text)}
    return ",".join(f"WebFetch(domain:{h})" for h in sorted(hosts) if HOST.match(h))


def leakage_verdict(raw: str | None) -> tuple[str | None, str]:
    """(risk, comment section) from the leakage pass's structured output; risk None when it gave none."""
    try:
        out = json.loads(raw or "")
    except ValueError:
        out = None
    if not isinstance(out, dict) or out.get("risk") not in ("none", "possible", "unknown"):
        return None, "**Pre-training overlap** (never blocks the merge): no verdict (the pass failed)"
    lines = [f"**Pre-training overlap** (never blocks the merge): **{out['risk']}**"]
    for item in list(out.get("evidence") or [])[:5]:
        if isinstance(item, dict):
            url = str(item.get("url", ""))[:300]
            url = url if re.match(r"^https?://[^\s<>()]+$", url) else "no valid URL"
            quote = " ".join(str(item.get("quote", "")).split())[:400]
            lines.append(f"- \"{quote}\" ({url})".replace("@", "@\u200b"))
    return out["risk"], "\n".join(lines)


def parse_review(raw: str | None) -> dict | None:
    """One pass's structured output, or None when the pass failed or returned something else."""
    try:
        review = json.loads(raw or "")
    except ValueError:
        return None
    ok = isinstance(review, dict) and review.get("recommendation") in ("MERGE", "FIX", "REJECT")
    return review if ok else None


def review_verdict(outputs: list[str | None]) -> tuple[str, str, str]:
    """("success" | "failure", status description, comment lines) from the passes' outputs."""
    reviews = [parse_review(raw) for raw in outputs]
    lines = []
    for i, review in enumerate(reviews, 1):
        if review is None:
            lines.append(f"Pass {i}/{len(reviews)}: no verdict (the review job failed)")
            continue
        lines.append(f"Pass {i}/{len(reviews)}: **{review['recommendation']}**")
        # The text comes from a model that read untrusted content: short, and no mentions.
        lines += [f"- {str(p)[:300]}".replace("@", "@​") for p in list(review.get("problems") or [])[:10]]
    if all(r and r["recommendation"] == "MERGE" for r in reviews):
        return "success", "both Claude reviews say MERGE", "\n".join(lines)
    return "failure", "the two Claude reviews of this commit do not both say MERGE", "\n".join(lines)


def decide(pr: dict, paths: list[str], status: dict | None, checks_run: dict | None,
           existing_models: set[str]) -> tuple[str, str]:
    """("merge" | "wait" | "needs" | "skip", reason). ``status``: the "Claude review" commit status
    of the current head, or None; ``checks_run``: the latest Checks run of the current head."""
    if pr["state"] != "open":
        return "skip", "closed"
    if pr.get("draft"):
        return "wait", "draft"
    scope = scope_problems(paths, pr["user"]["login"], existing_models)
    status_pending = not status or status["state"] == "pending"
    if scope:
        if ENV_INPUT in paths and status_pending:
            return "wait", "the review round reports on the environment change"
        return "needs", "; ".join(scope)
    sha = pr["head"]["sha"]
    if status_pending:
        return "wait", "the Claude review of this commit has not finished"
    if status["state"] != "success":
        return "needs", status.get("description") or "the Claude review did not pass"
    if not checks_run or checks_run["head_sha"] != sha or checks_run["status"] != "completed":
        return "wait", "Checks on this commit have not finished"
    if checks_run["conclusion"] != "success":
        return "needs", f"Checks ended with {checks_run['conclusion']} on {sha[:7]}"
    if NEEDS in {label["name"] for label in pr["labels"]}:
        return "needs", f"the label {NEEDS} is set"
    return "merge", sha


# ---- GitHub through the gh CLI ---------------------------------------------------------------------
def gh(*args, check=True) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check).stdout


def api(path: str, paginate: bool = False):
    out = gh("api", *(["--paginate", "--slurp"] if paginate else []), f"repos/{REPO}/{path}")
    data = json.loads(out)
    return [x for page in data for x in page] if paginate else data


def latest_run(workflow: str, sha: str) -> dict | None:
    runs = api(f"actions/workflows/{workflow}/runs?head_sha={sha}&per_page=20")["workflow_runs"]
    return max(runs, key=lambda r: r["run_started_at"], default=None)


def commit_status(sha: str) -> dict | None:
    """The latest "Claude review" status of a commit (the API lists newest first)."""
    return next((s for s in api(f"commits/{sha}/statuses?per_page=100") if s["context"] == REVIEW_CONTEXT), None)


def set_status(sha: str, state: str, description: str) -> None:
    gh("api", "--method", "POST", f"repos/{REPO}/statuses/{sha}", "-f", f"state={state}",
       "-f", f"context={REVIEW_CONTEXT}", "-f", f"description={description[:140]}")


def existing_models() -> set[str]:
    return {item["name"] for item in api("contents/models?ref=main") if item["type"] == "dir"}


def changed_paths(number: int) -> list[str]:
    files = api(f"pulls/{number}/files?per_page=100", paginate=True)
    return [f["filename"] for f in files] + [f["previous_filename"] for f in files if f.get("previous_filename")]


def comment_once(number: int, body: str, marker: str) -> None:
    """Comment unless a comment with ``marker`` already exists (a re-run says nothing new)."""
    if not any(marker in c["body"] for c in api(f"issues/{number}/comments?per_page=100", paginate=True)):
        gh("pr", "comment", str(number), "--repo", REPO, "--body", f"{body}\n\n{marker}")


def needs_maintainer(pr: dict, reason: str, details: str = "") -> None:
    gh("pr", "edit", str(pr["number"]), "--repo", REPO, "--add-label", NEEDS)
    body = f"@{MAINTAINER} your review is needed: {reason}" + (f"\n\n{details}" if details else "")
    comment_once(pr["number"], body[:60000], f"<!-- automerge-gate {pr['head']['sha']} -->")


def env_details(paths: list[str], raw: str | None) -> str:
    """The env job's report for a pull request that changes env/requirements.in, without mentions."""
    if ENV_INPUT not in paths:
        return ""
    return (raw or "").strip().replace("@", "@\u200b")[:20000] or "**Shared environment**: no report (the env job failed)."


def gate(pr: dict, details: str = "") -> str:
    sha = pr["head"]["sha"]
    verdict, reason = decide(pr, changed_paths(pr["number"]), commit_status(sha), latest_run(CHECKS_WORKFLOW, sha),
                             existing_models())
    print(f"#{pr['number']} {sha[:7]}: {verdict} ({reason})")
    if verdict == "needs":
        needs_maintainer(pr, reason, details)
    elif verdict == "merge":
        merged = subprocess.run(["gh", "pr", "merge", str(pr["number"]), "--repo", REPO, "--merge",
                                 "--match-head-commit", sha], capture_output=True, text=True)
        if merged.returncode:
            if api(f"pulls/{pr['number']}").get("merged"):
                print(f"#{pr['number']}: already merged")  # the other gate run got there first
                return "merged"
            needs_maintainer(pr, f"the merge failed: {merged.stderr.strip()[-300:]}")
            return "needs"
        if any(p.startswith("site/") for p in changed_paths(pr["number"])):
            gh("workflow", "run", "pages.yml", "--repo", REPO, "--ref", "main")
    return verdict


def reset(number: int, sha: str) -> None:
    """A review round starts on ``sha``: status pending, and claude-approved, needs-maintainer and
    pretraining-overlap removed when a bot set them for an older commit. A label the maintainer set
    by hand stays."""
    set_status(sha, "pending", "the two Claude reviews are running")
    events = api(f"issues/{number}/events?per_page=100", paginate=True)
    last = {}
    for e in events:
        if e["event"] == "labeled":
            last[e["label"]["name"]] = e["actor"]["login"]
    labels = {label["name"] for label in api(f"issues/{number}")["labels"]}
    for name in (APPROVED, NEEDS, OVERLAP):
        if name in labels and last.get(name) in BOTS:
            gh("pr", "edit", str(number), "--repo", REPO, "--remove-label", name)
            print(f"#{number}: removed {name}")


def verdict(number: int, sha: str) -> None:
    """The review round of ``sha`` ended: one comment, the labels, the commit status, then the gate.
    The leakage pass ($LEAKAGE, model pull requests only) adds a section and maybe its label, never a block."""
    state, description, lines = review_verdict([os.environ.get("REVIEW_ONE"), os.environ.get("REVIEW_TWO")])
    risk, leakage = leakage_verdict(os.environ.get("LEAKAGE")) if os.environ.get("MODEL") else (None, "")
    body = f"**Claude review of {sha[:7]}**\n{lines}"[:4000] + (f"\n\n{leakage}" if leakage else "")
    comment_once(number, body, f"<!-- claude-review {sha} -->")
    if state == "success":
        gh("pr", "edit", str(number), "--repo", REPO, "--add-label", APPROVED)
    if risk == "possible":
        gh("pr", "edit", str(number), "--repo", REPO, "--add-label", OVERLAP)
    set_status(sha, state, description)
    print(f"#{number} {sha[:7]}: review {state} ({description})")
    pr = api(f"pulls/{number}")
    if pr["head"]["sha"] == sha:  # otherwise a newer commit has its own round
        gate(pr, env_details(changed_paths(number), os.environ.get("ENV_REPORT")))


def inputs(number: int, sha: str) -> dict[str, str]:
    """What the review round's jobs need, written to $GITHUB_OUTPUT by the reset job: whether
    env/requirements.in changes; the one model folder whose model.yaml the pull request adds, if
    any; the web sites its leakage pass may fetch."""
    files = api(f"pulls/{number}/files?per_page=100", paginate=True)
    paths = [f["filename"] for f in files]
    cards = [p for p in paths if p in {f["filename"] for f in files if f["status"] != "removed"} and re.fullmatch(r"models/[^/]+/model\.yaml", p) and SLUG.match(p.split("/")[1])
             and p.split("/")[1] != "_template"]
    out = {"env": str(ENV_INPUT in paths).lower(), "model": "", "web": ""}
    if len(cards) == 1:
        text = gh("api", "-H", "Accept: application/vnd.github.raw", f"repos/{REPO}/contents/{cards[0]}?ref={sha}",
                  check=False)  # unreadable: the fixed sites only
        out.update(model=cards[0].split("/")[1], web=web_tools(text))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["gate", "triage", "reset", "verdict", "inputs"])
    parser.add_argument("--pr", type=int)
    parser.add_argument("--sha")
    args = parser.parse_args(argv)
    if args.mode == "reset":
        reset(args.pr, args.sha)
    elif args.mode == "inputs":
        print("\n".join(f"{k}={v}" for k, v in inputs(args.pr, args.sha).items()))
    elif args.mode == "verdict":
        verdict(args.pr, args.sha)
    elif args.mode == "triage":
        pr = api(f"pulls/{args.pr}")
        if pr["user"]["login"] == MAINTAINER:
            print(f"#{pr['number']}: opened by the maintainer, no triage")
            return 0
        paths = changed_paths(pr["number"])
        scope = scope_problems(paths, pr["user"]["login"], existing_models())
        if pr["state"] == "open" and scope and ENV_INPUT not in paths:  # else the verdict job asks, with the env report
            needs_maintainer(pr, "; ".join(scope))
        print(f"#{pr['number']}: scope " + ("; ".join(scope) if scope else "ok"))
    else:
        prs = [p for p in api("pulls?state=open&per_page=100", paginate=True) if p["head"]["sha"] == args.sha]
        if not prs:
            print(f"no open pull request has head {args.sha}")
        for pr in prs:
            gate(api(f"pulls/{pr['number']}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
