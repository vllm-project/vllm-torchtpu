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
"""`/ci` (`precheck`, `run`, `merge`, `cancel`, `help`) and PR CI info handler."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

VALID_SUBCOMMANDS = ("precheck", "run", "merge", "cancel", "help")
EXTRAS_SUBCOMMANDS = {"run", "merge"}
RUN_EXTRAS = ("perf-eval", "nightly-suite")
MERGE_EXTRAS = ("perf-eval", "nightly-suite", "precheck-only")
WRITE_ROLES = {"write", "maintain", "admin"}
INFO_MARKER = "<!-- torchtpu-ci-info -->"
EXPECTED_PR_WORKFLOWS = (
    (".github/workflows/pre-commit.yml", "pre-commit"),
    (".github/workflows/pr-text.yml", "PR Text"),
)
HELP_TEXT = (
    "- `/ci precheck`: re-run all non-TPU checks "
    "(pre-commit, PR checks, CPU tests).\n"
    "- `/ci run [perf-eval] [nightly-suite]`: TPU tests now.\n"
    "- `/ci merge [perf-eval] [nightly-suite]`: queue for merge once approved "
    "and green (with no extras, same as the **Merge when ready** button at the "
    "bottom of the GitHub PR page).\n"
    "- `/ci merge precheck-only`: queue for merge with prechecks only "
    "(no TPU; not on `releases/*`).\n"
    "- `/ci cancel`: stop this PR's running builds.\n"
    "- `/ci help`.\n\n"
    "`/ci` commands except help need write access."
)
BUILDKITE_BUILDS_URL = (
    "https://api.buildkite.com/v2/organizations/vllm/pipelines/vllm-torchtpu-ci/builds"
)
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
_AUTO_MERGE_MUTATION = (
    "mutation($id: ID!) { "
    "enablePullRequestAutoMerge(input: {pullRequestId: $id, mergeMethod: SQUASH}) "
    "{ pullRequest { number } } }"
)
_CI_LINE_RE = re.compile(r"^/ci(?:\s|$)")


def find_ci_line(body: str) -> str | None:
    """Return the first non-quoted, non-fenced line starting with `/ci`."""
    in_fence = False
    for raw in body.splitlines():
        s = raw.strip()
        if s.startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence and not s.startswith(">") and _CI_LINE_RE.match(s):
            return s
    return None


def parse_ci_line(line: str) -> tuple[str | None, list[str] | None, str | None]:
    """Parse `/ci precheck|run|merge|cancel|help`."""
    tokens = line.split()
    if len(tokens) < 2:
        return (
            None,
            None,
            "Missing `/ci` subcommand. Comment `/ci help` to list valid commands.",
        )
    subcmd = tokens[1]
    if subcmd not in VALID_SUBCOMMANDS:
        return (
            None,
            None,
            f"Unknown `/ci` subcommand `{subcmd}`. "
            "Comment `/ci help` to list valid commands.",
        )
    if subcmd not in EXTRAS_SUBCOMMANDS:
        if len(tokens) > 2:
            return (
                None,
                None,
                f"`/ci {subcmd}` takes no extra arguments (got `{tokens[2]}`). "
                "Comment `/ci help` to list valid commands.",
            )
        return subcmd, [], None
    allowed = MERGE_EXTRAS if subcmd == "merge" else RUN_EXTRAS
    for extra in tokens[2:]:
        if extra not in allowed:
            valid_list = ", ".join(f"`{e}`" for e in allowed)
            return (
                None,
                None,
                f"Unknown `/ci {subcmd}` extra `{extra}`. "
                f"Valid extras: {valid_list} (see `/ci help`).",
            )
    extras = [e for e in allowed if e in tokens[2:]]
    if "precheck-only" in extras and len(extras) > 1:
        return (
            None,
            None,
            "`precheck-only` can't be combined with other extras.",
        )
    return subcmd, extras, None


def _api_json(
    url: str, token: str, *, method: str = "GET", payload: dict[str, Any] | None = None
) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "vllm-torchtpu-ci-command",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def _err_summary(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    return str(exc) or exc.__class__.__name__


def has_write_access(repo: str, author: str, gh_token: str) -> bool:
    quoted = urllib.parse.quote(author, safe="")
    url = f"https://api.github.com/repos/{repo}/collaborators/{quoted}/permission"
    try:
        data = _api_json(url, gh_token)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    return bool({data.get("permission"), data.get("role_name")} & WRITE_ROLES)


def _trigger_build(
    pr: dict[str, Any],
    repo: str,
    pr_number: int,
    message: str,
    env: dict[str, str],
    bk_token: str,
) -> dict[str, Any]:
    if not bk_token:
        raise RuntimeError("BUILDKITE_API_TOKEN is required to trigger builds")
    head = pr["head"]
    head_repo = head.get("repo") or {}
    is_fork = bool(head_repo.get("full_name") and head_repo["full_name"] != repo)
    branch = head.get("label") or head["ref"] if is_fork else head["ref"]
    payload: dict[str, Any] = {
        "commit": head["sha"],
        "branch": branch,
        "message": message,
        "ignore_pipeline_branch_filters": True,
        "pull_request_id": pr_number,
        "pull_request_base_branch": (pr.get("base") or {}).get("ref", "main"),
        "env": env,
    }
    if is_fork and head_repo.get("clone_url"):
        payload["pull_request_repository"] = head_repo["clone_url"]
    return _api_json(BUILDKITE_BUILDS_URL, bk_token, method="POST", payload=payload)


def _run_pr_actions_prechecks(
    repo: str, head_sha: str, gh_token: str
) -> tuple[list[str], bool]:
    """Approve or re-run `pull_request` Actions runs on `head_sha`."""
    sha7 = head_sha[:7]
    runs_base = f"https://api.github.com/repos/{repo}/actions/runs"
    try:
        data = _api_json(f"{runs_base}?head_sha={head_sha}&per_page=100", gh_token)
    except Exception as exc:
        return [f"GitHub Actions lookup failed ({_err_summary(exc)})"], False

    latest: dict[str, dict[str, Any]] = {}
    for run in data.get("workflow_runs") or []:
        if run.get("event") != "pull_request":
            continue
        key = run.get("path") or run.get("name") or str(run["id"])
        if key not in latest or int(run["id"]) > int(latest[key]["id"]):
            latest[key] = run

    targets: list[tuple[str, dict[str, Any] | None]] = []
    seen_ids: set[int] = set()
    for wf_path, wf_name in EXPECTED_PR_WORKFLOWS:
        run = latest.get(wf_path) or latest.get(wf_name)
        if run is not None:
            seen_ids.add(int(run["id"]))
        targets.append((wf_name, run))
    for run in latest.values():
        if int(run["id"]) not in seen_ids:
            targets.append((run.get("name") or run.get("path") or "workflow", run))

    notes: list[str] = []
    all_ok = True
    for wf_name, run in targets:
        if run is None:
            all_ok = False
            notes.append(
                f"{wf_name} has no run for `{sha7}`; push or reopen to trigger it"
            )
            continue
        name = run.get("name") or wf_name
        run_id, url = run["id"], run.get("html_url", "")
        status, conclusion = run.get("status"), run.get("conclusion")
        if status == "action_required" or conclusion == "action_required":
            try:
                _api_json(f"{runs_base}/{run_id}/approve", gh_token, method="POST")
                notes.append(f"{name} approved ({url})")
            except Exception as exc:
                all_ok = False
                notes.append(
                    f"{name} could not be approved ({_err_summary(exc)}; "
                    f"click **Approve and run** on the GitHub PR page: {url})"
                )
        elif status == "completed":
            try:
                _api_json(f"{runs_base}/{run_id}/rerun", gh_token, method="POST")
                notes.append(f"{name} re-run ({url})")
            except Exception as exc:
                all_ok = False
                notes.append(f"{name} re-run failed ({_err_summary(exc)}: {url})")
        else:
            notes.append(f"{name} already {status} ({url})")
    return notes, all_ok


def _cancel_pr_builds(
    pr: dict[str, Any], repo: str, pr_number: int, bk_token: str
) -> list[int]:
    if not bk_token:
        raise RuntimeError("BUILDKITE_API_TOKEN is required to cancel builds")
    head = pr["head"]
    head_repo = head.get("repo") or {}
    is_fork = bool(head_repo.get("full_name") and head_repo["full_name"] != repo)
    target_branch = head.get("label") if is_fork else head.get("ref")
    base_ref = (pr.get("base") or {}).get("ref", "main")
    query = "state[]=scheduled&state[]=running&state[]=failing&per_page=100"
    builds = _api_json(f"{BUILDKITE_BUILDS_URL}?{query}", bk_token)
    cancelled: list[int] = []
    for b in builds:
        b_branch = b.get("branch") or ""
        if b_branch.startswith("gh-readonly-queue/") or b.get("merge_queue"):
            continue
        pr_id = str((b.get("pull_request") or {}).get("id") or "")
        same_branch = (
            bool(target_branch)
            and target_branch != base_ref
            and not target_branch.startswith("releases/")
            and b_branch == target_branch
        )
        if pr_id == str(pr_number) or b.get("commit") == head["sha"] or same_branch:
            num = int(b["number"])
            try:
                _api_json(
                    f"{BUILDKITE_BUILDS_URL}/{num}/cancel", bk_token, method="PUT"
                )
                cancelled.append(num)
            except urllib.error.HTTPError as exc:
                if exc.code != 422:
                    raise
    return sorted(cancelled)


def handle_pr_info(
    repo: str,
    pr_number: int,
    author: str,
    gh_token: str,
    *,
    draft: bool = False,
    is_bot: bool = False,
) -> str | None:
    """Post the one-time CI info comment on non-draft, non-bot PRs."""
    if draft or is_bot or author.endswith("[bot]"):
        return None
    comments_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    page = 1
    while True:
        items = _api_json(f"{comments_url}?per_page=100&page={page}", gh_token)
        if any(INFO_MARKER in (c.get("body") or "") for c in items):
            return None
        if len(items) < 100:
            break
        page += 1
    if has_write_access(repo, author, gh_token):
        guidance = (
            "- Every push runs non-TPU checks; comment `/ci precheck` to re-run "
            "all non-TPU checks (pre-commit, PR checks, CPU tests).\n"
            "- TPU tests don't run on push: comment "
            "`/ci run [perf-eval] [nightly-suite]` (`nightly-suite` is ~3–4h of TPU).\n"
            "- To merge, click the **Merge when ready** button at the bottom of the "
            "GitHub PR page, or comment `/ci merge [perf-eval] [nightly-suite]` "
            "(same as the button when no extras are given).\n"
            "- `/ci merge precheck-only`: queue for merge with prechecks only "
            "(no TPU; not on `releases/*`).\n"
            "- `/ci cancel` stops running builds; `/ci help` lists commands."
        )
    else:
        guidance = (
            "- You don't have write access, so a reviewer can run `/ci` commands "
            "for you (`/ci help` is open to anyone):\n"
            "  - `/ci precheck`: re-run all non-TPU checks "
            "(pre-commit, PR checks, CPU tests).\n"
            "  - `/ci run [perf-eval] [nightly-suite]`: TPU tests now.\n"
            "  - `/ci merge [perf-eval] [nightly-suite]`: queue for merge once "
            "approved and green (or a reviewer can click the **Merge when ready** "
            "button at the bottom of the GitHub PR page).\n"
            "  - `/ci merge precheck-only`: queue for merge with prechecks only "
            "(no TPU; not on `releases/*`).\n"
            "  - `/ci cancel`: stop this PR's running builds."
        )
    body = (
        f"✅ @{author}, thanks for the PR. How CI works here:\n\n"
        f"{guidance}\n\n"
        f"{INFO_MARKER}"
    )
    _api_json(comments_url, gh_token, method="POST", payload={"body": body})
    return body


def handle_comment(
    repo: str,
    pr_number: int,
    author: str,
    comment_body: str,
    gh_token: str,
    bk_token: str,
) -> str | None:
    """Process a PR comment and return the posted reply (or None if ignored)."""
    line = find_ci_line(comment_body)
    if line is None:
        return None
    comments_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"

    def reply(text: str) -> str:
        _api_json(comments_url, gh_token, method="POST", payload={"body": text})
        return text

    parts = line.split()
    if len(parts) >= 2 and parts[1] == "help":
        _, _, err = parse_ci_line(line)
        return reply(err if err is not None else HELP_TEXT)

    cmd_name = (
        f"/ci {parts[1]}" if len(parts) > 1 and parts[1] in VALID_SUBCOMMANDS else "/ci"
    )
    if not has_write_access(repo, author, gh_token):
        return reply(
            f"@{author}, `{cmd_name}` needs write access to this repo. "
            "A reviewer can run it for you (comment `/ci help` for commands)."
        )
    subcmd, extras, err = parse_ci_line(line)
    if err is not None or subcmd is None or extras is None:
        return reply(err or "")
    pr = _api_json(f"https://api.github.com/repos/{repo}/pulls/{pr_number}", gh_token)
    if subcmd == "cancel":
        cancelled = _cancel_pr_builds(pr, repo, pr_number, bk_token)
        if not cancelled:
            return reply("No running CI builds for this PR.")
        nums = ", ".join(f"#{n}" for n in cancelled)
        noun = "build" if len(cancelled) == 1 else "builds"
        return reply(
            f"Cancelled {noun} {nums}. To leave the merge queue, click "
            "**Disable auto-merge** or **Remove from queue** on the GitHub PR page."
        )
    if pr.get("state") != "open":
        return reply(
            f"PR #{pr_number} is closed; reopen it before running `/ci {subcmd}`."
        )
    if subcmd == "merge":
        base_ref = (pr.get("base") or {}).get("ref") or ""
        if "precheck-only" in extras and base_ref.startswith("releases/"):
            return reply("`releases/*` merges always run the full gate.")
        if not pr.get("auto_merge"):
            gql = _api_json(
                GITHUB_GRAPHQL_URL,
                gh_token,
                method="POST",
                payload={
                    "query": _AUTO_MERGE_MUTATION,
                    "variables": {"id": pr["node_id"]},
                },
            )
            if gql.get("errors"):
                reason = gql["errors"][0].get("message", "GraphQL error")
                return reply(
                    f"@{author}, could not enable auto-merge on PR #{pr_number}: "
                    f"{reason}"
                )
        extras_csv = ",".join(extras)
        marker = (
            f"<!-- ci-merge-extras: {extras_csv} -->\n<!-- ci-merge-actor: {author} -->"
            if extras_csv
            else "<!-- ci-merge-extras: -->"
        )
        if extras == ["precheck-only"]:
            return reply(
                "Queued for merge with prechecks only (no TPU) once approved and "
                "green. To leave the queue, click **Disable auto-merge** or "
                "**Remove from queue** on the GitHub PR page.\n\n"
                f"{marker}"
            )
        extras_note = ", ".join(extras) if extras else "none"
        if "nightly-suite" in extras:
            extras_note += ", ~3–4h of TPU; the queue waits for it"
        return reply(
            f"Queued for merge once approved and green (extras: {extras_note}). "
            "To cancel, click **Disable auto-merge** or **Remove from queue** "
            "on the GitHub PR page.\n\n"
            f"{marker}"
        )
    if subcmd == "precheck":
        head_sha = pr["head"]["sha"]
        sha7 = head_sha[:7]
        notes, actions_ok = _run_pr_actions_prechecks(repo, head_sha, gh_token)
        bk_ok = True
        try:
            build = _trigger_build(
                pr,
                repo,
                pr_number,
                f"/ci precheck (PR #{pr_number} by @{author})",
                {"CI_PRECHECK": "1"},
                bk_token,
            )
            notes.append(f"Buildkite CPU checks ({build['web_url']})")
        except Exception as exc:
            bk_ok = False
            notes.append(f"Buildkite CPU checks failed ({_err_summary(exc)})")
        if actions_ok and bk_ok:
            return reply(f"Started prechecks on `{sha7}`: " + ", ".join(notes))
        return reply(f"Prechecks on `{sha7}` (partial): " + "; ".join(notes))
    cmd_text = "/ci run" + (f" {' '.join(extras)}" if extras else "")
    build = _trigger_build(
        pr,
        repo,
        pr_number,
        f"{cmd_text} (PR #{pr_number} by @{author})",
        {"CI_RUN_TPU": "1", "CI_EXTRAS": ",".join(extras)},
        bk_token,
    )
    extras_label = ", ".join(extras) if extras else "none"
    return reply(
        f"Started `{cmd_text}` on `{pr['head']['sha'][:7]}` "
        f"(extras: {extras_label}): {build['web_url']}"
    )


if __name__ == "__main__":
    repo, pr_num = os.environ["GITHUB_REPOSITORY"], int(os.environ["PR_NUMBER"])
    actor, gh_tok = os.environ["ACTOR_LOGIN"], os.environ["GH_TOKEN"]
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request_target":
        handle_pr_info(
            repo,
            pr_num,
            actor,
            gh_tok,
            draft=os.environ.get("PR_DRAFT", "").lower() == "true",
            is_bot=os.environ.get("ACTOR_TYPE") == "Bot",
        )
    else:
        body, bk_tok = (
            os.environ.get("COMMENT_BODY", ""),
            os.environ.get("BUILDKITE_API_TOKEN", ""),
        )
        handle_comment(repo, pr_num, actor, body, gh_tok, bk_tok)
    sys.exit(0)
