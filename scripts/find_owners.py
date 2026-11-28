#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Find the CODEOWNERS reviewers for a set of changed files.

Typical use, from anywhere inside the repo:

    # Who should review my current branch?
    scripts/find_owners.py

    # Who should review PR #123?
    scripts/find_owners.py --pr 123

    # Who owns these paths?
    scripts/find_owners.py src/vllm_torchtpu/kernels/ docs/index.md

    # Just the handles, e.g. to pipe into `gh pr edit --add-reviewer`
    scripts/find_owners.py --handles

    # Request their review (on the current branch's PR, or --pr 123)
    scripts/find_owners.py --request

    # One least-loaded owner per area instead of the smallest cover
    scripts/find_owners.py --pr 123 --balance

    # What the Reviewer Balancer workflow runs on a new PR
    scripts/find_owners.py --pr 123 --balance --request --prune --comment
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import nullcontext
from typing import NoReturn

CODEOWNERS_CANDIDATES = (
    ".github/CODEOWNERS",
    "CODEOWNERS",
    "docs/CODEOWNERS",
)


# --------------------------------------------------------------------------
# shell helpers
# --------------------------------------------------------------------------
def run(cmd: list[str], cwd: str | None = None) -> str | None:
    """Run a command, returning stdout, or None if it failed."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def run_checked(cmd: list[str], stdin: str | None = None) -> tuple[bool, str]:
    """Run a command, returning (succeeded, output-or-error-message)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, input=stdin)
    except FileNotFoundError:
        return False, f"{cmd[0]} not found"
    if proc.returncode != 0:
        return False, (proc.stderr.strip() or proc.stdout.strip())
    return True, proc.stdout.strip()


def repo_root() -> str:
    root = run(["git", "rev-parse", "--show-toplevel"])
    if not root:
        die("not inside a git repository (and no paths were given)")
    return root


def die(msg: str) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------
# CODEOWNERS parsing and matching
# --------------------------------------------------------------------------
class Rule:
    """One CODEOWNERS line: a path pattern plus the owners of that path."""

    def __init__(self, pattern: str, owners: list[str], lineno: int):
        self.pattern = pattern
        self.owners = owners
        self.lineno = lineno
        self.regex = compile_pattern(pattern)

    def matches(self, path: str) -> bool:
        return self.regex.match(path) is not None


def compile_pattern(pattern: str) -> re.Pattern:
    """Translate a gitignore-style CODEOWNERS pattern into a regex.

    Follows the rules GitHub documents: `/` anchors to the repo root, a
    pattern containing a slash is anchored, `*` stops at a path separator,
    `**` spans separators, and matching a directory matches everything
    beneath it -- except for a rule ending in `*`; see below.
    """
    p = pattern
    directory_rule = p.endswith("/")
    if directory_rule:
        p = p[:-1]
    anchored = p.startswith("/") or "/" in p
    p = p.lstrip("/")

    out: list[str] = []
    i, n = 0, len(p)
    while i < n:
        c = p[i]
        if c == "*":
            if p[i : i + 3] == "**/":
                out.append("(?:.*/)?")
                i += 3
            elif p[i : i + 2] == "**":
                out.append(".*")
                i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1

    prefix = "" if anchored else "(?:.*/)?"
    # A directory rule owns everything under it; a rule whose last segment
    # ends in `*` does not. This is the one place CODEOWNERS departs from
    # gitignore: `docs/*` matches `docs/getting-started.md` but not
    # `docs/build-app/troubleshooting.md`. Bare `*` and `*.py` still reach any
    # depth, via the unanchored prefix.
    globbed_leaf = not directory_rule and p.rsplit("/", 1)[-1].endswith("*")
    subtree = "" if globbed_leaf else "(?:/.*)?"
    return re.compile("^" + prefix + "".join(out) + subtree + "$")


def load_rules(root: str, override: str | None = None) -> tuple[list[Rule], str]:
    if override:
        with open(override, encoding="utf-8") as f:
            return parse_codeowners(f.read()), override
    for rel in CODEOWNERS_CANDIDATES:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return parse_codeowners(f.read()), rel
    die(f"no CODEOWNERS file found (looked in {', '.join(CODEOWNERS_CANDIDATES)})")


def parse_codeowners(content: str) -> list[Rule]:
    rules = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        owners = [o for o in parts[1:] if o.startswith("@") or "@" in o]
        # A pattern with no owners is kept, not skipped: CODEOWNERS uses it to
        # un-own a subdirectory that an earlier rule claimed. Since the last
        # match wins, such a rule leaves the path with no owner at all.
        rules.append(Rule(parts[0], owners, lineno))
    return rules


def owners_for(path: str, rules: list[Rule]) -> Rule | None:
    """Return the last matching rule, which is the one GitHub applies."""
    match = None
    for rule in rules:
        if rule.matches(path):
            match = rule
    return match


# --------------------------------------------------------------------------
# figuring out which files changed
# --------------------------------------------------------------------------
def resolve_base(root: str, base: str | None) -> str:
    if base:
        return base
    for candidate in ("origin/main", "upstream/main", "main"):
        if run(["git", "rev-parse", "--verify", "--quiet", candidate], cwd=root):
            return candidate
    die("could not find a base branch; pass --base <ref>")


def changed_files(root: str, base: str) -> tuple[list[str], str]:
    """Files changed on this branch: committed, staged, unstaged, untracked."""
    merge_base = run(["git", "merge-base", "HEAD", base], cwd=root) or base
    files = set()

    # Deleted files are kept: GitHub still applies ownership to a path you
    # delete, and `gh pr diff --name-only` lists them, so dropping them here
    # would make local runs disagree with --pr runs.
    diff = run(["git", "diff", "--name-only", merge_base], cwd=root)
    if diff:
        files.update(diff.splitlines())

    untracked = run(["git", "ls-files", "--others", "--exclude-standard"], cwd=root)
    if untracked:
        files.update(untracked.splitlines())

    return sorted(files), f"vs {base}"


def pr_files(pr: str, repo: str | None) -> tuple[list[str], str]:
    """The files a PR changes, by their path in the PR.

    The files endpoint pages up to 3000 files. `gh pr diff --name-only`
    fetches the diff media type instead, which GitHub refuses above 300
    files, so a large sweep or rename would never get an answer.
    """
    ok, out = run_checked(
        [
            "gh",
            "api",
            f"repos/{resolve_repo(repo)}/pulls/{pr}/files",
            "--paginate",
            "--jq",
            ".[].filename",
        ]
    )
    if not ok:
        die(f"could not read the files of PR #{pr}: {out}")
    return sorted({line for line in out.splitlines() if line}), f"PR #{pr}"


def expand_paths(root: str, paths: list[str]) -> list[str]:
    """Turn user-supplied paths into repo-relative file paths."""
    files = set()
    for raw in paths:
        abs_path = os.path.abspath(raw)
        rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
        if os.path.isdir(abs_path):
            tracked = run(["git", "ls-files", "--", rel], cwd=root)
            if tracked:
                files.update(tracked.splitlines())
            else:
                # Not a tracked directory; still resolve ownership for it.
                files.add(rel.rstrip("/") + "/")
        else:
            files.add(rel)
    return sorted(files)


# --------------------------------------------------------------------------
# requesting reviewers on a PR
# --------------------------------------------------------------------------
def resolve_author(pr: str | None, repo: str | None) -> str | None:
    """Who will own the PR: its author, or you for not-yet-pushed changes.

    Returns None when `gh` cannot answer, in which case the author simply is
    not excluded from the suggestions.
    """
    if pr:
        cmd = ["gh", "pr", "view", str(pr), "--json", "author", "--jq", ".author.login"]
        if repo:
            cmd += ["--repo", repo]
    else:
        # `gh api` takes no --repo; the login is account-wide anyway.
        cmd = ["gh", "api", "user", "--jq", ".login"]
    return run(cmd) or None


def resolve_pr(pr: str | None, repo: str | None) -> str:
    """The PR to act on: the one given, else the current branch's PR."""
    if pr:
        return str(pr)
    cmd = ["gh", "pr", "view", "--json", "number", "--jq", ".number"]
    if repo:
        cmd += ["--repo", repo]
    ok, out = run_checked(cmd)
    if not ok or not out:
        die(
            "no PR found for the current branch; open one first or pass "
            "--pr <PR_NUMBER>"
        )
    return out


def request_reviewers(
    owners: list[str],
    pr: str | None,
    repo: str | None,
    style: Style,
) -> None:
    """Request review from the given owners on a PR.

    The REST endpoint, not `gh pr edit --add-reviewer`: the latter resolves
    handles against the organization's teams, which the workflow's token
    cannot read.
    """
    handles = [o.lstrip("@") for o in owners]
    if not handles:
        print("No owners to request; nothing to do.")
        return

    pr = resolve_pr(pr, repo)
    ok, out = run_checked(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{resolve_repo(repo)}/pulls/{pr}/requested_reviewers",
            "--input",
            "-",
        ],
        stdin=json.dumps({"reviewers": handles}),
    )
    if not ok:
        die(f"could not request review on PR #{pr}: {out}")

    joined = ", ".join(f"@{h}" for h in handles)
    print(f"\nRequested review on PR #{pr} from {style.green(joined)}")


# --------------------------------------------------------------------------
# grouping and reporting
# --------------------------------------------------------------------------
def group_by_rule(files: list[str], rules: list[Rule]) -> tuple[dict, list[str]]:
    groups: dict[str, dict] = {}
    unowned: list[str] = []
    for path in files:
        rule = owners_for(path, rules)
        if rule is None or not rule.owners:
            unowned.append(path)
            continue
        group = groups.setdefault(rule.pattern, {"owners": rule.owners, "files": []})
        group["files"].append(path)
    return groups, unowned


def eligible_owners(owners: list[str], author: str | None) -> list[str]:
    """Owners who can actually review, i.e. everyone but the author."""
    if not author:
        return list(owners)
    return [o for o in owners if o.lstrip("@").lower() != author.lower()]


def self_owned_areas(groups: dict, author: str | None) -> list[str]:
    """Patterns whose only owner is the author, so nobody else can be asked."""
    return sorted(
        pat
        for pat, g in groups.items()
        if g["owners"] and not eligible_owners(g["owners"], author)
    )


def minimal_cover(groups: dict, author: str | None = None) -> list[str]:
    """Smallest set of owners with at least one owner per matched rule.

    Every module needs one approval from its owners, so this is the shortest
    reviewer list that can unblock the whole PR. Greedy set cover: repeatedly
    take the owner who covers the most still-uncovered rules.

    The author is never chosen. GitHub refuses to register a self-review
    request, and asking for one drops the rest of the request along with it.
    """
    eligible = {pat: eligible_owners(g["owners"], author) for pat, g in groups.items()}
    # Areas the author owns outright have nobody left to ask; they are
    # reported separately rather than silently covered.
    remaining = {pat for pat, owners in eligible.items() if owners}
    chosen: list[str] = []
    while remaining:
        best, best_cover = None, set()
        for owner in sorted({o for owners in eligible.values() for o in owners}):
            covered = {pat for pat in remaining if owner in eligible[pat]}
            if len(covered) > len(best_cover):
                best, best_cover = owner, covered
        if not best:
            break
        chosen.append(best)
        remaining -= best_cover
    return chosen


# --------------------------------------------------------------------------
# balancing: one least-loaded owner per area
# --------------------------------------------------------------------------
# GitHub asks every owner of every area a PR touches, so the people on broad
# rules sit on dozens of pending requests. The balancer keeps CODEOWNERS as
# the record of who may approve and only trims who is asked: one owner per
# area, the one with the fewest pending review requests across the repo.
#
# Label that switches the balancer off for one PR.
OPT_OUT_LABEL = "no-reviewer-balance"
# Hidden prefix of the comment the balancer keeps on a PR; the JSON after it
# records its picks so later runs do not re-roll them.
_MARKER = "<!-- reviewer-balancer "
_MARKER_END = " -->"


def handle_key(handle: str) -> str:
    """Case-folded login without the `@`, for comparing handles."""
    return handle.lstrip("@").lower()


def resolve_repo(repo: str | None) -> str:
    if repo:
        return repo
    out = run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]
    )
    if not out:
        die("could not determine the repository; pass --repo owner/name")
    return out


def loads_query(keys: list[str], repo: str) -> str:
    """One GraphQL query counting each login's pending review requests."""
    fields = []
    for i, key in enumerate(keys):
        search = f"repo:{repo} is:pr is:open review-requested:{key}"
        fields.append(
            f"  u{i}: search(query: {json.dumps(search)}, type: ISSUE) {{ issueCount }}"
        )
    return "query {\n" + "\n".join(fields) + "\n}"


def pending_review_loads(candidates: list[str], repo: str) -> dict[str, int]:
    """Open PRs in `repo` on which each candidate is still asked to review.

    That count is a reviewer's current depth: a request clears as soon as
    they review, so it tracks outstanding work rather than history. One
    GraphQL call serves every candidate; a failure is fatal rather than a
    guess, because a pick made on missing loads is not a balanced one.
    """
    keys = sorted({handle_key(c) for c in candidates})
    if not keys:
        return {}
    ok, out = run_checked(
        ["gh", "api", "graphql", "-f", f"query={loads_query(keys, repo)}"]
    )
    if not ok:
        die(f"could not read reviewer loads: {out}")
    data = json.loads(out).get("data") or {}
    loads: dict[str, int] = {}
    for i, key in enumerate(keys):
        node = data.get(f"u{i}")
        if not node or "issueCount" not in node:
            die(f"reviewer load missing for {key} in: {out}")
        loads[key] = int(node["issueCount"])
    return loads


def _tiebreak(handle: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{handle_key(handle)}".encode()).hexdigest()


def uncovered_areas(groups: dict, author: str | None, covered: set[str]) -> list[str]:
    """Patterns with no eligible owner in `covered`, most constrained first.

    `covered` holds logins already asked or heard from. Areas the author owns
    outright are left out; `self_owned_areas` reports those.
    """
    keys = {handle_key(c) for c in covered}
    open_areas = []
    for pattern, group in groups.items():
        eligible = eligible_owners(group["owners"], author)
        if eligible and not any(handle_key(o) in keys for o in eligible):
            open_areas.append((len(eligible), pattern))
    return [pattern for _, pattern in sorted(open_areas)]


def balanced_picks(
    groups: dict,
    loads: dict[str, int],
    author: str | None = None,
    seed: str = "",
) -> list[str]:
    """One least-loaded eligible owner per matched rule.

    Areas are visited most constrained first (fewest eligible owners), so a
    narrow area picks before a broad one that its pick may also own; an area
    already covered by an earlier pick adds nobody. Ties on load break on a
    hash of the seed and the handle, so reruns for one PR agree without
    favouring the top of the alphabet. The author is never chosen.
    """
    load = {handle_key(k): int(v) for k, v in loads.items()}
    picks: list[str] = []
    for pattern in uncovered_areas(groups, author, set()):
        eligible = eligible_owners(groups[pattern]["owners"], author)
        if any(handle_key(p) in {handle_key(o) for o in eligible} for p in picks):
            continue
        picks.append(
            min(
                eligible, key=lambda o: (load.get(handle_key(o), 0), _tiebreak(o, seed))
            )
        )
    return picks


def prune_candidates(
    groups: dict,
    areas: list[str],
    pending: list[str],
    keep: list[str],
    author: str | None,
) -> list[str]:
    """Pending requests to withdraw: owners of `areas` not in `keep`.

    Only code owners of the areas balanced in this run are touched, spelled
    as GitHub returned them. A pending reviewer who owns none of them was
    asked by a person, and stays.
    """
    owners = {
        handle_key(o)
        for pattern in areas
        for o in eligible_owners(groups[pattern]["owners"], author)
    }
    kept = {handle_key(k) for k in keep}
    return sorted(
        login
        for login in pending
        if handle_key(login) in owners and handle_key(login) not in kept
    )


def format_marker(state: dict) -> str:
    return f"{_MARKER}{json.dumps(state, sort_keys=True)}{_MARKER_END}"


def parse_marker(body: str) -> dict | None:
    """The balancer state in a comment body, or None if it carries none."""
    start = body.find(_MARKER)
    if start < 0:
        return None
    start += len(_MARKER)
    end = body.find(_MARKER_END, start)
    if end < 0:
        return None
    try:
        state = json.loads(body[start:end])
    except ValueError:
        return None
    return state if isinstance(state, dict) else None


def balancer_comment(
    picks: dict[str, tuple[str, int]],
    pruned: list[str],
    state: dict,
    kept: list[str] | None = None,
) -> str:
    """The PR comment explaining a run; handles are not @-mentioned, because
    the review request already notified the picks."""
    intro = (
        "**Reviewer balancer** asked one code owner per area, the one with "
        "the fewest open review requests:"
    )
    outro = (
        "Any owner of an area may still approve it. Request anyone else by "
        f"hand, or add the `{OPT_OUT_LABEL}` label to opt this PR out."
    )
    lines = [intro, ""]
    for pattern, (handle, load) in picks.items():
        lines.append(f"- `{pattern}` → `{handle.lstrip('@')}` ({load} pending)")
    if pruned:
        quoted = ", ".join(f"`{p}`" for p in pruned)
        lines += ["", f"Withdrew the other code-owner requests: {quoted}."]
    if kept:
        quoted = ", ".join(f"`{p}`" for p in kept)
        pinned_note = (
            f"GitHub kept the requests of {quoted}: the ruleset still requires "
            "review from code owners, which pins automatically requested "
            "owners. No action is needed from them."
        )
        lines += ["", pinned_note]
    lines += ["", outro, "", format_marker(state)]
    return "\n".join(lines)


def pr_review_state(pr: str, repo: str) -> tuple[list[str], list[str], list[str]]:
    """(pending requested reviewers, logins who reviewed, labels) of a PR."""
    ok, out = run_checked(
        [
            "gh",
            "api",
            f"repos/{repo}/pulls/{pr}",
            "--jq",
            "{pending: [.requested_reviewers[].login], labels: [.labels[].name]}",
        ]
    )
    if not ok:
        die(f"could not read PR #{pr}: {out}")
    head = json.loads(out)
    ok, out = run_checked(
        [
            "gh",
            "api",
            f"repos/{repo}/pulls/{pr}/reviews",
            "--paginate",
            "--jq",
            ".[] | .user.login // empty",
        ]
    )
    if not ok:
        die(f"could not read the reviews of PR #{pr}: {out}")
    reviewed = sorted({line for line in out.splitlines() if line})
    return head["pending"], reviewed, head["labels"]


def read_marker(pr: str, repo: str) -> tuple[int | None, dict]:
    """The balancer's comment on a PR: (comment id, recorded state)."""
    ok, out = run_checked(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/{pr}/comments",
            "--paginate",
            "--jq",
            ".[] | {id: .id, body: .body}",
        ]
    )
    if not ok:
        die(f"could not read the comments of PR #{pr}: {out}")
    for line in out.splitlines():
        if not line.strip():
            continue
        comment = json.loads(line)
        state = parse_marker(comment.get("body") or "")
        if state is not None:
            return int(comment["id"]), state
    return None, {}


def write_marker(pr: str, repo: str, comment_id: int | None, body: str) -> None:
    if comment_id is None:
        cmd = ["gh", "api", "--method", "POST", f"repos/{repo}/issues/{pr}/comments"]
    else:
        cmd = [
            "gh",
            "api",
            "--method",
            "PATCH",
            f"repos/{repo}/issues/comments/{comment_id}",
        ]
    ok, out = run_checked(cmd + ["-f", f"body={body}"])
    if not ok:
        die(f"could not write the balancer comment on PR #{pr}: {out}")


def remove_reviewers(
    logins: list[str], pr: str, repo: str, style: Style
) -> tuple[list[str], list[str]]:
    """Withdraw pending requests; returns (withdrawn, kept).

    GitHub answers 200 either way. While a ruleset requires review from code
    owners it keeps automatically requested owners whose files are still in
    the diff, so the request list is read back and anyone still on it is
    reported as kept rather than claimed as withdrawn.
    """
    if not logins:
        return [], []
    ok, out = run_checked(
        [
            "gh",
            "api",
            "--method",
            "DELETE",
            f"repos/{repo}/pulls/{pr}/requested_reviewers",
            "--input",
            "-",
        ],
        stdin=json.dumps({"reviewers": logins}),
    )
    if not ok:
        die(f"could not withdraw review requests on PR #{pr}: {out}")
    try:
        still = {u["login"].lower() for u in json.loads(out)["requested_reviewers"]}
    except (ValueError, KeyError, TypeError):
        still = {login.lower() for login in pr_review_state(pr, repo)[0]}
    withdrawn = [login for login in logins if login.lower() not in still]
    kept = [login for login in logins if login.lower() in still]
    if withdrawn:
        joined = ", ".join(f"@{login}" for login in withdrawn)
        print(f"Withdrew review requests on PR #{pr} from {style.yellow(joined)}")
    if kept:
        joined = ", ".join(f"@{login}" for login in kept)
        print(
            style.yellow(
                f"GitHub kept the requests of {joined}: a ruleset that requires "
                "review from code owners pins automatically requested owners."
            )
        )
    return withdrawn, kept


def balance_pr(
    groups: dict,
    author: str | None,
    pr: str,
    repo: str,
    style: Style,
    *,
    request: bool,
    prune: bool,
    comment: bool,
) -> dict:
    """Balance one PR: pick for the areas nobody covers yet, then act.

    An area counts as covered by an earlier pick of this balancer or by
    anyone who has reviewed, whether or not their request is still pending:
    a person who withdrew a pick meant it. Only the areas balanced in this
    run have their other pending owners withdrawn, so a reviewer someone
    added by hand to an already covered area is left alone.
    """
    pending, reviewed, labels = pr_review_state(pr, repo)
    comment_id, state = read_marker(pr, repo)
    prev_picks = [str(p) for p in state.get("picks", [])]
    result: dict = {
        "areas": [],
        "picks": [],
        "loads": {},
        "requested": [],
        "pruned": [],
        "pinned": [],
        "skipped": None,
    }
    if OPT_OUT_LABEL in labels:
        result["skipped"] = f"label {OPT_OUT_LABEL}"
        print(f"PR #{pr} carries the {OPT_OUT_LABEL} label; leaving it alone.")
        return result

    areas = uncovered_areas(groups, author, set(prev_picks) | set(reviewed))
    result["areas"] = areas
    if not areas:
        print(f"Every area of PR #{pr} already has a reviewer; nothing to do.")
        return result

    open_groups = {pattern: groups[pattern] for pattern in areas}
    candidates = sorted(
        {o for g in open_groups.values() for o in eligible_owners(g["owners"], author)}
    )
    loads = pending_review_loads(candidates, repo)
    picks = balanced_picks(open_groups, loads, author, seed=str(pr))
    result["loads"] = loads
    result["picks"] = picks

    pending_keys = {handle_key(p) for p in pending}
    to_request = [p.lstrip("@") for p in picks if handle_key(p) not in pending_keys]
    keep = prev_picks + reviewed + picks
    to_prune = prune_candidates(groups, areas, pending, keep, author) if prune else []
    result["requested"] = to_request
    result["pruned"] = to_prune

    print(style.bold("Least-loaded owner per area"))
    width = max(len(p) for p in areas)
    picked_by_area: dict[str, tuple[str, int]] = {}
    for pattern in areas:
        eligible = {
            handle_key(o) for o in eligible_owners(groups[pattern]["owners"], author)
        }
        pick = next(p for p in picks if handle_key(p) in eligible)
        load = loads.get(handle_key(pick), 0)
        picked_by_area[pattern] = (pick, load)
        depth = style.dim(f"({load} pending)")
        print(f"  {pattern.ljust(width)}  {style.green(pick)}  {depth}")

    if request and to_request:
        request_reviewers(to_request, pr, repo, style)
    withdrawn, kept = remove_reviewers(to_prune, pr, repo, style) if prune else ([], [])
    result["pruned"] = withdrawn
    result["pinned"] = kept
    if comment:
        new_state = {
            "areas": sorted(set(state.get("areas", [])) | set(areas)),
            "picks": sorted({handle_key(p) for p in prev_picks + picks}),
        }
        write_marker(
            pr,
            repo,
            comment_id,
            balancer_comment(picked_by_area, withdrawn, new_state, kept),
        )
    return result


class Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t):
        return self(t, "1")

    def dim(self, t):
        return self(t, "2")

    def green(self, t):
        return self(t, "32")

    def yellow(self, t):
        return self(t, "33")


def report(
    source: str,
    files: list[str],
    groups: dict,
    unowned: list[str],
    codeowners_path: str,
    show_files: bool,
    style: Style,
    author: str | None = None,
    picks: list[str] | None = None,
    loads: dict[str, int] | None = None,
) -> None:
    if not files:
        print(f"No files found ({source}). Nothing to review.")
        return

    print(f"{style.bold(str(len(files)))} file(s) — {source}")
    print(style.dim(f"ownership from {codeowners_path}\n"))

    if not groups:
        print("No CODEOWNERS rule matches these files.")
    else:
        print(style.bold("Owners by area"))
        width = max(len(p) for p in groups)
        for pattern in sorted(groups):
            group = groups[pattern]
            owners = " ".join(group["owners"])
            count = len(group["files"])
            plural = "" if count == 1 else "s"
            print(
                f"  {pattern.ljust(width)}  {style.green(owners)}"
                f"  {style.dim(f'({count} file{plural})')}"
            )
            if show_files:
                for path in group["files"]:
                    print(style.dim(f"      {path}"))

        balanced = picks is not None
        cover = picks if balanced else minimal_cover(groups, author)
        mine = self_owned_areas(groups, author)
        print()
        if cover:
            print(style.bold("Ping these reviewers"))
            if balanced:
                shown = " ".join(
                    f"{o} ({(loads or {}).get(handle_key(o), 0)} pending)"
                    for o in cover
                )
                print(f"  {style.green(shown)}")
                print(
                    style.dim(
                        "  (least-loaded owner per area; "
                        "one approval per area is required)"
                    )
                )
            else:
                print(f"  {style.green(' '.join(cover))}")
                print(
                    style.dim(
                        "  (smallest set covering every area above; "
                        "one approval per area is required)"
                    )
                )
            handles = ",".join(o.lstrip("@") for o in cover)
            print()
            print(style.dim("  gh pr edit --add-reviewer ") + handles)
            print(style.dim("  (or re-run with --request to do that for you)"))
        else:
            print(style.bold("No one to ping"))
            print(style.dim("  you own every area this change touches"))

        if mine:
            print()
            print(
                style.yellow(
                    f"You are the only owner of {len(mine)} area(s); "
                    "ask anyone for a second pair of eyes:"
                )
            )
            for pattern in mine:
                print(f"  {pattern}")

    if unowned:
        print()
        print(style.yellow(f"No owner matched ({len(unowned)} file(s)):"))
        for path in unowned:
            print(f"  {path}")


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the CODEOWNERS reviewers for your changes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Typical use, from anywhere inside the repo:")[-1],
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="files or directories to look up (default: files changed on this branch)",
    )
    parser.add_argument("--pr", help="look up the files of an existing PR number")
    parser.add_argument(
        "--repo", help="owner/name to use with --pr (default: current repo)"
    )
    parser.add_argument(
        "--base", help="base ref to diff against (default: origin/main)"
    )
    parser.add_argument(
        "--codeowners",
        help="read ownership from this file instead of the checked-out one",
    )
    parser.add_argument(
        "--files",
        action="store_true",
        help="list the matched files under each owner group",
    )
    parser.add_argument(
        "--handles",
        action="store_true",
        help="print only a comma-separated reviewer list, nothing else",
    )
    parser.add_argument(
        "--request",
        action="store_true",
        help="request review from the owners on the PR "
        "(--pr, else the current branch's PR)",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="pick one owner per area, the one with the fewest open review "
        "requests in the repo, instead of the smallest covering set",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="with --pr --balance: withdraw the pending requests of the other "
        "owners of the areas balanced in this run",
    )
    parser.add_argument(
        "--comment",
        action="store_true",
        help="with --pr --balance: keep one comment on the PR explaining the "
        "picks (the Reviewer Balancer workflow sets this)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON output")
    parser.add_argument("--no-color", action="store_true", help="disable color")
    args = parser.parse_args()

    if (args.prune or args.comment) and not (args.pr and args.balance):
        parser.error("--prune and --comment need --pr and --balance")

    color = sys.stdout.isatty() and not args.no_color and not os.environ.get("NO_COLOR")
    style = Style(color)

    # With --json the human report goes to stderr, so stdout stays parseable.
    report_to = contextlib.redirect_stdout(sys.stderr) if args.json else nullcontext()

    root = repo_root()
    rules, codeowners_path = load_rules(root, args.codeowners)

    if args.pr:
        files, source = pr_files(args.pr, args.repo)
    elif args.paths:
        files, source = expand_paths(root, args.paths), "explicit paths"
    else:
        base = resolve_base(root, args.base)
        files, source = changed_files(root, base)

    groups, unowned = group_by_rule(files, rules)
    author = resolve_author(args.pr, args.repo)

    if args.balance and args.pr and (args.request or args.prune or args.comment):
        # The workflow path: act on the PR and report what was done.
        with report_to:
            result = balance_pr(
                groups,
                author,
                str(args.pr),
                resolve_repo(args.repo),
                style,
                request=args.request,
                prune=args.prune,
                comment=args.comment,
            )
        if args.json:
            print(json.dumps(result, indent=2))
        return

    picks: list[str] | None = None
    loads: dict[str, int] | None = None
    if args.balance:
        repo = resolve_repo(args.repo)
        candidates = sorted(
            {o for g in groups.values() for o in eligible_owners(g["owners"], author)}
        )
        loads = pending_review_loads(candidates, repo)
        picks = balanced_picks(groups, loads, author, seed=str(args.pr or ""))
    reviewers = picks if picks is not None else minimal_cover(groups, author)

    if args.handles:
        print(",".join(o.lstrip("@") for o in reviewers))
        return

    if args.json:
        print(
            json.dumps(
                {
                    "source": source,
                    "codeowners": codeowners_path,
                    "files": files,
                    "groups": [
                        {
                            "pattern": pattern,
                            "owners": groups[pattern]["owners"],
                            "files": groups[pattern]["files"],
                        }
                        for pattern in sorted(groups)
                    ],
                    "reviewers": reviewers,
                    "balanced": picks is not None,
                    "loads": loads or {},
                    "author": author,
                    "self_owned": self_owned_areas(groups, author),
                    "unowned": unowned,
                },
                indent=2,
            )
        )
        return

    report(
        source,
        files,
        groups,
        unowned,
        codeowners_path,
        args.files,
        style,
        author,
        picks,
        loads,
    )

    if args.request:
        request_reviewers(reviewers, args.pr, args.repo, style)


if __name__ == "__main__":
    main()
