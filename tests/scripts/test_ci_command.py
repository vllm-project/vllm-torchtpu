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
"""Unit tests for .github/scripts/ci_command.py."""

from __future__ import annotations

import http.client
import importlib.util
import io
import pathlib
import unittest
import urllib.error
from typing import Any
from unittest import mock

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "ci_command", _ROOT / ".github/scripts/ci_command.py"
)
assert _SPEC and _SPEC.loader
ci_command = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci_command)
BK_URL = ci_command.BUILDKITE_BUILDS_URL
CANARY_URL = ci_command.CANARY_BUILDS_URL
GQL_URL = ci_command.GITHUB_GRAPHQL_URL
BK_WEB = "https://buildkite.com/vllm/vllm-torchtpu-ci/builds/9999"
REPO = "vllm-project/vllm-torchtpu"
RUNS_URL = f"https://api.github.com/repos/{REPO}/actions/runs"


class TestCiCommand(unittest.TestCase):
    def _mock_api(
        self,
        *,
        role: str = "write",
        pr_state: str = "open",
        base_ref: str = "main",
        pr_author: str = "bob",
        commit_author_email: str | None = "bob@corp.example.com",
        commit_author_name: str | None = "Bob Example",
        pr_labels: list[str] | None = None,
        auto_merge: dict[str, Any] | None = None,
        head: dict[str, Any] | None = None,
        active_builds: list[dict[str, Any]] | None = None,
        workflow_runs: list[dict[str, Any]] | None = None,
        fail_approve_ids: set[int] | None = None,
        gql_error: str | None = None,
        bk_error: int | None = None,
        in_queue: bool = False,
    ) -> tuple[Any, list[tuple[str, str, dict[str, Any] | None]]]:
        calls: list[tuple[str, str, dict[str, Any] | None]] = []
        comments: list[dict[str, str]] = []
        state: dict[str, Any] = {"auto_merge": auto_merge, "in_queue": in_queue}
        pr_head = head or {
            "sha": "abcdef1234567890",
            "ref": "feat/example",
            "label": "vllm-project:feat/example",
            "repo": {"full_name": REPO},
        }

        def fake_api(
            url: str,
            token: str,
            *,
            method: str = "GET",
            payload: dict[str, Any] | None = None,
        ) -> Any:
            calls.append((method, url, payload))
            if url.endswith("/permission"):
                if role == "404":
                    raise urllib.error.HTTPError(
                        url, 404, "Not Found", http.client.HTTPMessage(), io.BytesIO()
                    )
                return {"permission": role, "role_name": role}
            if "/issues/" in url:
                if method == "POST" and payload is not None:
                    comments.append({"body": payload["body"]})
                return list(comments)
            if "/pulls/" in url:
                return {
                    "state": pr_state,
                    "node_id": "PR_kwDO1234",
                    "user": {"login": pr_author} if pr_author else {},
                    "labels": [{"name": name} for name in (pr_labels or [])],
                    "auto_merge": state["auto_merge"],
                    "head": pr_head,
                    "base": {"ref": base_ref},
                }
            if "/commits/" in url:
                return {
                    "commit": {
                        "author": {
                            "name": commit_author_name or "",
                            "email": commit_author_email or "",
                        }
                    }
                }
            if url.startswith(f"{RUNS_URL}?"):
                return {"workflow_runs": list(workflow_runs or [])}
            if url.startswith(f"{RUNS_URL}/") and method == "POST":
                for bad_id in fail_approve_ids or set():
                    if url.endswith(f"/{bad_id}/approve"):
                        raise urllib.error.HTTPError(
                            url,
                            403,
                            "Forbidden",
                            http.client.HTTPMessage(),
                            io.BytesIO(),
                        )
                return {}
            if url == GQL_URL:
                query = (payload or {}).get("query", "")
                if "isInMergeQueue" in query:
                    auto = {"enabledAt": "t"} if state["auto_merge"] else None
                    node = {
                        "isInMergeQueue": state["in_queue"],
                        "autoMergeRequest": auto,
                    }
                    return {"data": {"node": node}}
                if "dequeuePullRequest" in query:
                    state["in_queue"] = False
                    return {"data": {"node": {}}}
                if "disablePullRequestAutoMerge" in query:
                    state["auto_merge"] = None
                    return {"data": {"node": {}}}
                if gql_error is not None:
                    return {"errors": [{"message": gql_error}]}
                state["auto_merge"] = {"merge_method": "squash"}
                return {"data": {"enablePullRequestAutoMerge": {"pullRequest": {}}}}
            bk = next((b for b in (BK_URL, CANARY_URL) if url.startswith(b)), None)
            if bk is not None and bk_error is not None:
                raise urllib.error.HTTPError(
                    url, bk_error, "Forbidden", http.client.HTTPMessage(), io.BytesIO()
                )
            if bk is not None and url.startswith(f"{bk}?"):
                return list(active_builds or [])
            if bk is not None and url.startswith(f"{bk}/") and url.endswith("/cancel"):
                if url.endswith("/905/cancel"):
                    raise urllib.error.HTTPError(
                        url,
                        422,
                        "Unprocessable",
                        http.client.HTTPMessage(),
                        io.BytesIO(),
                    )
                return {"state": "canceled"}
            return {"web_url": BK_WEB} if url in (BK_URL, CANARY_URL) else {"id": 1}

        return fake_api, calls

    def _run(
        self,
        body: str,
        *,
        role: str = "write",
        pr_state: str = "open",
        base_ref: str = "main",
        pr_author: str = "bob",
        commit_author_email: str | None = "bob@corp.example.com",
        commit_author_name: str | None = "Bob Example",
        pr_labels: list[str] | None = None,
        head: dict[str, Any] | None = None,
        active_builds: list[dict[str, Any]] | None = None,
        workflow_runs: list[dict[str, Any]] | None = None,
        fail_approve_ids: set[int] | None = None,
        bk_error: int | None = None,
    ) -> tuple[str | None, list[tuple[str, str, dict[str, Any] | None]]]:
        fake_api, calls = self._mock_api(
            role=role,
            pr_state=pr_state,
            base_ref=base_ref,
            pr_author=pr_author,
            commit_author_email=commit_author_email,
            commit_author_name=commit_author_name,
            pr_labels=pr_labels,
            head=head,
            active_builds=active_builds,
            workflow_runs=workflow_runs,
            fail_approve_ids=fail_approve_ids,
            bk_error=bk_error,
        )
        with mock.patch.object(ci_command, "_api_json", side_effect=fake_api):
            reply = ci_command.handle_comment(
                REPO, 42, "alice", body, "gh-tok", "bk-tok"
            )
        return reply, calls

    def test_ignores_non_command_quoted_and_fenced_lines(self) -> None:
        for body in (
            "Looks good, /ci workflow is clean.",
            "> /ci run\nAgreed!",
            "```bash\n/ci merge perf-eval\n```",
            "/circa 2026",
        ):
            reply, calls = self._run(body)
            self.assertIsNone(reply)
            self.assertEqual(calls, [])

    def test_help_open_to_anyone_and_other_commands_require_write_access(self) -> None:
        for role in ("404", "none", "read", "triage", "write"):
            reply, calls = self._run("/ci help", role=role)
            assert reply is not None
            for expected in (
                "`/ci precheck`: re-run all non-TPU checks "
                "(pre-commit, PR checks, CPU tests).",
                "`/ci run`: run the TPU tests now. Optionally add `perf-eval`",
                "e.g. `/ci run perf-eval`",
                "`/ci merge`: queue for merge once approved and green",
                "e.g. `/ci merge perf-eval`",
                "`/ci merge precheck-only`",
                "**Merge when ready** button at the bottom of the GitHub PR page",
                "`/ci cancel`: stop this PR's running builds, take it out of the "
                "merge queue, and turn off auto-merge.",
                "`/ci help`.",
                "`/ci` commands except help need write access.",
            ):
                self.assertIn(expected, reply)
            self.assertNotIn("[perf-eval]", reply)
            self.assertFalse(any(u.endswith("/permission") for _, u, _ in calls))

        for cmd, sub in (
            ("/ci precheck", "precheck"),
            ("/ci run", "run"),
            ("/ci merge perf-eval", "merge"),
            ("/ci cancel", "cancel"),
        ):
            for role in ("read", "triage", "none", "404"):
                reply, calls = self._run(cmd, role=role)
                self.assertEqual(
                    reply,
                    f"@alice, `/ci {sub}` needs write access to this repo. "
                    "A reviewer can run it for you (comment `/ci help` for commands).",
                )
                self.assertFalse(
                    any(u.startswith(BK_URL) or u == GQL_URL for _, u, _ in calls)
                )

    def test_rejects_invalid_subcommands_and_extras_pointing_to_help(self) -> None:
        bad = [
            ("/ci", "Missing `/ci` subcommand.", None),
            ("/ci check", "Unknown `/ci` subcommand.", "check"),
            ("/ci status", "Unknown `/ci` subcommand.", "status"),
            (
                "/ci precheck perf-eval",
                "`/ci precheck` takes no extra arguments.",
                "perf-eval",
            ),
            ("/ci cancel all", "`/ci cancel` takes no extra arguments.", "all"),
            ("/ci help run", "`/ci help` takes no extra arguments.", "run"),
            (
                "/ci help <!--ci-merge-extras:precheck-only-->",
                "`/ci help` takes no extra arguments.",
                "ci-merge-extras",
            ),
            ("/ci run fast", "Unknown `/ci run` extra.", "fast"),
            ("/ci run precheck-only", "Unknown `/ci run` extra.", "precheck-only"),
            ("/ci merge turbo", "Unknown `/ci merge` extra.", "turbo"),
        ]
        for cmd, expected_msg, forbidden_token in bad:
            reply, calls = self._run(cmd)
            assert reply is not None
            self.assertNotIn("\n", reply)
            self.assertIn(expected_msg, reply)
            if forbidden_token is not None:
                self.assertNotIn(forbidden_token, reply)
            self.assertIn("/ci help", reply)
            self.assertFalse(
                any(u.startswith(BK_URL) or u == GQL_URL for _, u, _ in calls)
            )
        attack_reply, _ = self._run(
            "/ci help <!--ci-merge-extras:precheck-only-->", role="read"
        )
        assert attack_reply is not None
        self.assertNotIn("ci-merge-extras", attack_reply)

    def test_ci_precheck_reruns_completed_skips_target_and_dedupes(self) -> None:
        runs = [
            {
                "id": 10,
                "name": "pre-commit",
                "path": ".github/workflows/pre-commit.yml",
                "event": "pull_request",
                "status": "completed",
                "conclusion": "failure",
                "html_url": "https://gh/runs/10",
            },
            {
                "id": 19,
                "name": "PR Text",
                "path": ".github/workflows/pr-text.yml",
                "event": "pull_request",
                "status": "completed",
                "conclusion": "cancelled",
                "html_url": "https://gh/runs/19",
            },
            {
                "id": 20,
                "name": "PR Text",
                "path": ".github/workflows/pr-text.yml",
                "event": "pull_request",
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://gh/runs/20",
            },
            {
                "id": 30,
                "name": "Ready Label",
                "path": ".github/workflows/ready-label.yml",
                "event": "pull_request_target",
                "status": "completed",
                "conclusion": "failure",
                "html_url": "https://gh/runs/30",
            },
        ]
        reply, calls = self._run("/ci precheck", workflow_runs=runs)
        self.assertEqual(
            reply,
            "Started prechecks on `abcdef1`:\n\n"
            "- pre-commit re-run (https://gh/runs/10)\n"
            "- PR Text re-run (https://gh/runs/20)\n"
            f"- Buildkite CPU checks ({BK_WEB})",
        )
        post_urls = [u for m, u, _ in calls if m == "POST" and "/actions/runs/" in u]
        self.assertEqual(post_urls, [f"{RUNS_URL}/10/rerun", f"{RUNS_URL}/20/rerun"])
        bk_calls = [p for m, u, p in calls if m == "POST" and u == BK_URL]
        self.assertEqual(len(bk_calls), 1)
        assert bk_calls[0] is not None
        self.assertEqual(bk_calls[0]["env"], {"CI_PRECHECK": "1"})

    def test_ci_precheck_approves_fork_leaves_running_and_handles_missing_or_partial(
        self,
    ) -> None:
        fork_head = {
            "sha": "1234567890abcdef",
            "ref": "main",
            "label": "ext-dev:main",
            "repo": {
                "full_name": "ext-dev/vllm-torchtpu",
                "clone_url": "https://github.com/ext-dev/vllm-torchtpu.git",
            },
        }
        fork_runs = [
            {
                "id": 51,
                "name": "pre-commit",
                "path": ".github/workflows/pre-commit.yml",
                "event": "pull_request",
                "status": "completed",
                "conclusion": "action_required",
                "html_url": "https://gh/runs/51",
            },
            {
                "id": 52,
                "name": "PR Text",
                "path": ".github/workflows/pr-text.yml",
                "event": "pull_request",
                "status": "in_progress",
                "conclusion": None,
                "html_url": "https://gh/runs/52",
            },
        ]
        reply, calls = self._run(
            "/ci precheck", head=fork_head, workflow_runs=fork_runs
        )
        self.assertEqual(
            reply,
            "Started prechecks on `1234567`:\n\n"
            "- pre-commit approved (https://gh/runs/51)\n"
            "- PR Text already in_progress (https://gh/runs/52)\n"
            f"- Buildkite CPU checks ({BK_WEB})",
        )
        action_posts = [u for m, u, _ in calls if m == "POST" and "/actions/runs/" in u]
        self.assertEqual(action_posts, [f"{RUNS_URL}/51/approve"])
        fork_bk = [p for m, u, p in calls if m == "POST" and u == BK_URL][0]
        assert fork_bk is not None
        self.assertEqual(fork_bk["branch"], "ext-dev:main")
        self.assertEqual(
            fork_bk["pull_request_repository"],
            "https://github.com/ext-dev/vllm-torchtpu.git",
        )

        # Partial failure when fork approval returns 403 and PR Text has no run yet.
        partial_reply, _ = self._run(
            "/ci precheck",
            head=fork_head,
            workflow_runs=fork_runs[:1],
            fail_approve_ids={51},
        )
        assert partial_reply is not None
        self.assertIn("Prechecks on `1234567` (partial):", partial_reply)
        self.assertIn("pre-commit could not be approved (HTTP 403;", partial_reply)
        self.assertIn(
            "PR Text has no run for `1234567`; push or reopen to trigger it",
            partial_reply,
        )
        self.assertIn(f"Buildkite CPU checks ({BK_WEB})", partial_reply)

    def test_triggers_buildkite_build_with_extras_and_replies_link(self) -> None:
        cases = (
            ("write", "LGTM\n/ci run", "", "extras: none"),
            ("maintain", "/ci run perf-eval", "perf-eval", "extras: perf-eval"),
            (
                "admin",
                "/ci run nightly-suite perf-eval",
                "perf-eval,nightly-suite",
                "extras: perf-eval, nightly-suite",
            ),
        )
        for role, cmd, expected_extras, label in cases:
            reply, calls = self._run(
                cmd,
                role=role,
                pr_author="ExtAuthor",
                commit_author_email="ext.author@company.com",
                commit_author_name="Ext Author",
                pr_labels=["ready", "oncall-fix"],
            )
            assert reply is not None
            self.assertIn(BK_WEB, reply)
            self.assertIn(label, reply)
            bk_calls = [p for m, u, p in calls if m == "POST" and u == BK_URL]
            self.assertEqual(len(bk_calls), 1)
            assert bk_calls[0] is not None
            self.assertEqual(bk_calls[0]["pull_request_id"], 42)
            self.assertEqual(
                bk_calls[0]["author"],
                {
                    "name": "Ext Author",
                    "email": "ext.author@company.com",
                },
            )
            # /ci builds ignore PR labels, so none are sent.
            self.assertNotIn("pull_request_labels", bk_calls[0])
            self.assertEqual(
                bk_calls[0]["env"], {"CI_RUN_TPU": "1", "CI_EXTRAS": expected_extras}
            )

        # Fallback to <login>@users.noreply.github.com if commit author email is empty.
        _, fb_calls = self._run(
            "/ci run",
            pr_author="ExtAuthor",
            commit_author_email="",
            commit_author_name="",
        )
        fb_bk = [p for m, u, p in fb_calls if m == "POST" and u == BK_URL][0]
        assert fb_bk is not None
        self.assertEqual(
            fb_bk["author"],
            {"name": "ExtAuthor", "email": "extauthor@users.noreply.github.com"},
        )

    def test_ci_merge_enables_auto_merge_writes_and_clears_marker(self) -> None:
        fake_api, calls = self._mock_api(role="write")
        with mock.patch.object(ci_command, "_api_json", side_effect=fake_api):
            r1 = ci_command.handle_comment(
                REPO, 42, "alice", "/ci merge nightly-suite perf-eval", "gh-tok", ""
            )
            r2 = ci_command.handle_comment(REPO, 42, "alice", "/ci merge", "gh-tok", "")
        assert r1 is not None and r2 is not None
        self.assertIn("<!-- ci-merge-extras: perf-eval,nightly-suite -->", r1)
        self.assertIn("<!-- ci-merge-actor: alice -->", r1)
        self.assertIn("<!-- ci-merge-sha: abcdef1234567890 -->", r1)
        self.assertIn("~3–4h of TPU; the queue waits for it", r1)
        self.assertIn(
            "click **Disable auto-merge** or **Remove from queue** "
            "on the GitHub PR page.",
            r1,
        )
        self.assertIn("<!-- ci-merge-extras: -->", r2)
        self.assertIn("extras: none", r2)
        gql_calls = [p for m, u, p in calls if m == "POST" and u == GQL_URL]
        self.assertEqual(len(gql_calls), 1)
        assert gql_calls[0] is not None
        self.assertIn("enablePullRequestAutoMerge", gql_calls[0]["query"])
        self.assertIn("SQUASH", gql_calls[0]["query"])
        self.assertEqual(gql_calls[0]["variables"], {"id": "PR_kwDO1234"})
        self.assertFalse(any(u.startswith(BK_URL) for _, u, _ in calls))
        # The marker is posted before auto-merge is enabled, so a PR that enters
        # the queue at once already has it.
        first_comment = next(
            i for i, (m, u, _) in enumerate(calls) if m == "POST" and "/issues/" in u
        )
        first_gql = next(i for i, (_, u, _) in enumerate(calls) if u == GQL_URL)
        self.assertLess(first_comment, first_gql)

        # If enabling auto-merge fails, a follow-up reply clears the marker.
        fake_err, err_calls = self._mock_api(
            gql_error="Pull request is in clean status"
        )
        with mock.patch.object(ci_command, "_api_json", side_effect=fake_err):
            err = ci_command.handle_comment(
                REPO, 42, "alice", "/ci merge perf-eval", "gh-tok", ""
            )
        self.assertEqual(
            err,
            "@alice, could not enable auto-merge on PR #42: Pull request is in clean "
            "status\n\n<!-- ci-merge-extras: -->",
        )
        bodies = [
            p["body"]
            for m, u, p in err_calls
            if m == "POST" and "/issues/" in u and p is not None
        ]
        self.assertEqual(len(bodies), 2)
        self.assertIn("<!-- ci-merge-extras: perf-eval -->", bodies[0])

        # After a new push updates head.sha, a new `/ci merge` binds to the new SHA.
        new_head = {
            "sha": "9999888877776666",
            "ref": "feat/example",
            "label": "vllm-project:feat/example",
            "repo": {"full_name": REPO},
        }
        r3, _ = self._run("/ci merge perf-eval", head=new_head)
        assert r3 is not None
        self.assertIn("<!-- ci-merge-sha: 9999888877776666 -->", r3)
        self.assertNotIn("abcdef1234567890", r3)

    def test_ci_merge_precheck_only_and_rejections(self) -> None:
        reply, calls = self._run("/ci merge precheck-only", role="maintain")
        self.assertEqual(
            reply,
            "Queued for merge with prechecks only (no TPU) once approved and "
            "green. To leave the queue, click **Disable auto-merge** or "
            "**Remove from queue** on the GitHub PR page.\n\n"
            "<!-- ci-merge-extras: precheck-only -->\n"
            "<!-- ci-merge-actor: alice -->\n"
            "<!-- ci-merge-sha: abcdef1234567890 -->",
        )
        self.assertEqual(sum(1 for m, u, _ in calls if m == "POST" and u == GQL_URL), 1)
        self.assertFalse(any(u.startswith(BK_URL) for _, u, _ in calls))

        for combo in (
            "/ci merge precheck-only perf-eval",
            "/ci merge perf-eval precheck-only",
            "/ci merge nightly-suite precheck-only",
            "/ci merge perf-eval nightly-suite precheck-only",
        ):
            bad_reply, bad_calls = self._run(combo)
            self.assertEqual(
                bad_reply, "`precheck-only` can't be combined with other extras."
            )
            self.assertFalse(
                any(u.startswith(BK_URL) or u == GQL_URL for _, u, _ in bad_calls)
            )

        rel_reply, rel_calls = self._run(
            "/ci merge precheck-only", base_ref="releases/v0.30.0"
        )
        self.assertEqual(rel_reply, "`releases/*` merges always run the full gate.")
        self.assertFalse(
            any(u.startswith(BK_URL) or u == GQL_URL for _, u, _ in rel_calls)
        )

        # Normal `/ci merge` on `releases/*` still succeeds.
        ok_rel, ok_calls = self._run("/ci merge", base_ref="releases/v0.30.0")
        assert ok_rel is not None
        self.assertIn("<!-- ci-merge-extras: -->", ok_rel)
        self.assertEqual(
            sum(1 for m, u, _ in ok_calls if m == "POST" and u == GQL_URL), 1
        )

    def test_ci_cancel_stops_pr_builds_skips_merge_queue_and_handles_empty(
        self,
    ) -> None:
        fork_head = {
            "sha": "abcdef1234567890",
            "ref": "main",
            "label": "ext-dev:main",
            "repo": {"full_name": "ext-dev/vllm-torchtpu"},
        }
        builds = [
            {
                "number": 902,
                "branch": "ext-dev:main",
                "commit": "oldsha",
                "pull_request": {"id": "42"},
            },
            {
                "number": 901,
                "branch": "ext-dev:main",
                "commit": "abcdef1234567890",
                "pull_request": None,
            },
            {
                "number": 905,
                "branch": "ext-dev:main",
                "commit": "abcdef1234567890",
                "pull_request": {"id": "42"},
            },
            {
                "number": 903,
                "branch": "gh-readonly-queue/main/pr-42-deadbeef",
                "commit": "mqsha",
                "pull_request": None,
            },
            {
                "number": 904,
                "branch": "main",
                "commit": "mainsha",
                "pull_request": None,
            },
            {
                "number": 906,
                "branch": "other:feat",
                "commit": "othersha",
                "pull_request": {"id": "99"},
            },
        ]
        reply, calls = self._run("/ci cancel", head=fork_head, active_builds=builds)
        self.assertEqual(
            reply,
            "Cancelled builds #901, #902.\n\n<!-- ci-merge-extras: -->",
        )
        list_urls = [u for m, u, _ in calls if m == "GET" and u.startswith(BK_URL)]
        self.assertEqual(len(list_urls), 1)
        self.assertIn("state[]=scheduled&state[]=running&state[]=failing", list_urls[0])
        put_urls = [u for m, u, _ in calls if m == "PUT"]
        self.assertEqual(
            set(put_urls),
            {f"{BK_URL}/901/cancel", f"{BK_URL}/902/cancel", f"{BK_URL}/905/cancel"},
        )

        empty_reply, _ = self._run("/ci cancel", active_builds=[])
        self.assertEqual(
            empty_reply,
            "No running CI builds for this PR.\n\n<!-- ci-merge-extras: -->",
        )

        # `auto_merge_disabled` event also posts an empty marker to clear extras.
        fake_dis, dis_calls = self._mock_api(role="write")
        with mock.patch.object(ci_command, "_api_json", side_effect=fake_dis):
            dis_reply = ci_command.handle_auto_merge_disabled(REPO, 42, "gh-tok")
        self.assertIn("<!-- ci-merge-extras: -->", dis_reply)
        self.assertEqual(len(dis_calls), 1)
        self.assertEqual(dis_calls[0][0], "POST")

    def test_inert_until_live_except_canary_prs_which_use_canary_pipeline(
        self,
    ) -> None:
        # Before cutover, a PR based on main gets no reply, build, or comment.
        for cmd in ("/ci help", "/ci run", "/ci merge", "/ci typo"):
            fake_api, calls = self._mock_api(base_ref="main")
            with mock.patch.object(ci_command, "_api_json", side_effect=fake_api):
                reply = ci_command.handle_comment(
                    REPO, 42, "alice", cmd, "gh-tok", "bk-tok", live=False
                )
            self.assertIsNone(reply)
            self.assertEqual([m for m, _, _ in calls], ["GET"])
        fake_api, calls = self._mock_api(base_ref="main")
        with mock.patch.object(ci_command, "_api_json", side_effect=fake_api):
            self.assertIsNone(
                ci_command.handle_pr_info(REPO, 42, "alice", "gh-tok", live=False)
            )
            self.assertIsNone(
                ci_command.handle_auto_merge_disabled(REPO, 42, "gh-tok", live=False)
            )
        self.assertFalse(any(m == "POST" for m, _, _ in calls))

        # A PR based on mq-canary works before cutover and builds on the canary
        # pipeline; so does `/ci cancel`. After cutover the routing still follows
        # the base branch.
        for live in (False, True):
            fake_api, calls = self._mock_api(
                base_ref="mq-canary",
                active_builds=[
                    {"number": 7, "branch": "feat/example", "commit": "x"},
                ],
            )
            with mock.patch.object(ci_command, "_api_json", side_effect=fake_api):
                run = ci_command.handle_comment(
                    REPO, 42, "alice", "/ci run", "gh-tok", "bk-tok", live=live
                )
                cancel = ci_command.handle_comment(
                    REPO, 42, "alice", "/ci cancel", "gh-tok", "bk-tok", live=live
                )
            assert run is not None and cancel is not None
            self.assertIn(BK_WEB, run)
            self.assertIn("Cancelled build #7.", cancel)
            bk_urls = [u for _, u, _ in calls if "api.buildkite.com" in u]
            self.assertTrue(bk_urls)
            self.assertTrue(all(u.startswith(CANARY_URL) for u in bk_urls))

    def test_run_and_cancel_reply_when_buildkite_fails(self) -> None:
        # A Buildkite error (say, the token's team can't build the pipeline) gets a
        # reply instead of a crashed workflow and a silent PR.
        for cmd, expected in (
            ("/ci run", "@alice, could not start `/ci run` (HTTP 403)."),
            (
                "/ci cancel",
                "@alice, could not cancel builds (HTTP 403).\n\n"
                "<!-- ci-merge-extras: -->",
            ),
        ):
            reply, calls = self._run(cmd, bk_error=403)
            self.assertEqual(reply, expected)
            self.assertEqual(
                sum(1 for m, u, _ in calls if m == "POST" and "/issues/" in u), 1
            )
        fake_api, _ = self._mock_api()
        with mock.patch.object(ci_command, "_api_json", side_effect=fake_api):
            no_token = ci_command.handle_comment(
                REPO, 42, "alice", "/ci run", "gh-tok", ""
            )
        self.assertEqual(
            no_token,
            "@alice, could not start `/ci run` "
            "(BUILDKITE_API_TOKEN is required to trigger builds).",
        )

    def test_ci_cancel_takes_pr_out_of_queue_and_turns_off_auto_merge(self) -> None:
        def gql_queries(calls: list[tuple[str, str, dict[str, Any] | None]]) -> str:
            return " ".join(p["query"] for _, u, p in calls if u == GQL_URL and p)

        fake_api, calls = self._mock_api(
            in_queue=True, auto_merge={"merge_method": "squash"}, active_builds=[]
        )
        with mock.patch.object(ci_command, "_api_json", side_effect=fake_api):
            reply = ci_command.handle_comment(
                REPO, 42, "alice", "/ci cancel", "gh-tok", "bk-tok"
            )
        self.assertEqual(
            reply,
            "No running CI builds for this PR. Took the PR out of the merge queue. "
            "Turned off auto-merge.\n\n<!-- ci-merge-extras: -->",
        )
        self.assertIn("dequeuePullRequest", gql_queries(calls))
        self.assertIn("disablePullRequestAutoMerge", gql_queries(calls))

        # A Buildkite failure still leaves the queue.
        fake_bk, bk_calls = self._mock_api(in_queue=True, bk_error=403)
        with mock.patch.object(ci_command, "_api_json", side_effect=fake_bk):
            bk_reply = ci_command.handle_comment(
                REPO, 42, "alice", "/ci cancel", "gh-tok", "bk-tok"
            )
        self.assertEqual(
            bk_reply,
            "@alice, could not cancel builds (HTTP 403). Took the PR out of the merge "
            "queue.\n\n<!-- ci-merge-extras: -->",
        )
        self.assertNotIn("disablePullRequestAutoMerge", gql_queries(bk_calls))

    def test_closed_pr_does_not_trigger_build_or_merge(self) -> None:
        for cmd in ("/ci precheck", "/ci run", "/ci merge"):
            reply, calls = self._run(cmd, pr_state="closed")
            assert reply is not None and "closed" in reply
            self.assertFalse(
                any(u.startswith(BK_URL) or u == GQL_URL for _, u, _ in calls)
            )

    def test_pr_info_comment_variants_dedupe_and_skip_drafts_and_bots(self) -> None:
        fake_api, calls = self._mock_api(role="write")
        with mock.patch.object(ci_command, "_api_json", side_effect=fake_api):
            first = ci_command.handle_pr_info(REPO, 42, "alice", "gh-tok")
            second = ci_command.handle_pr_info(REPO, 42, "alice", "gh-tok")
        assert first is not None and second is None
        for expected in (
            "re-run all non-TPU checks (pre-commit, PR checks, CPU tests)",
            "comment `/ci run` to run them. Optionally add `perf-eval`",
            "**Merge when ready** button at the bottom of the GitHub PR page",
            "`/ci merge perf-eval` or `/ci merge nightly-suite`",
            "`/ci merge precheck-only`",
            "`/ci cancel`",
            "`/ci help`",
            ci_command.INFO_MARKER,
        ):
            self.assertIn(expected, first)
        self.assertEqual(sum(1 for m, _, _ in calls if m == "POST"), 1)

        fake_ext, _ = self._mock_api(role="404")
        with mock.patch.object(ci_command, "_api_json", side_effect=fake_ext):
            ext = ci_command.handle_pr_info(REPO, 43, "ext-dev", "gh-tok")
        assert ext is not None
        self.assertIn("You don't have write access", ext)
        self.assertIn("`/ci help` is open to anyone", ext)
        self.assertIn(
            "`/ci precheck`: re-run all non-TPU checks "
            "(pre-commit, PR checks, CPU tests).",
            ext,
        )
        self.assertIn(
            "**Merge when ready** button at the bottom of the GitHub PR page", ext
        )
        self.assertIn("`/ci merge precheck-only`", ext)

        fake_skip, skip_calls = self._mock_api(role="write")
        with mock.patch.object(ci_command, "_api_json", side_effect=fake_skip):
            self.assertIsNone(
                ci_command.handle_pr_info(REPO, 44, "alice", "gh-tok", draft=True)
            )
            self.assertIsNone(
                ci_command.handle_pr_info(REPO, 44, "dependabot[bot]", "gh-tok")
            )
            self.assertIsNone(
                ci_command.handle_pr_info(REPO, 44, "ci-app", "gh-tok", is_bot=True)
            )
        self.assertEqual(skip_calls, [])


if __name__ == "__main__":
    unittest.main()
