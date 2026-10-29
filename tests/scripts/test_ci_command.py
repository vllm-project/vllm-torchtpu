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
        auto_merge: dict[str, Any] | None = None,
        head: dict[str, Any] | None = None,
        active_builds: list[dict[str, Any]] | None = None,
        workflow_runs: list[dict[str, Any]] | None = None,
        fail_approve_ids: set[int] | None = None,
    ) -> tuple[Any, list[tuple[str, str, dict[str, Any] | None]]]:
        calls: list[tuple[str, str, dict[str, Any] | None]] = []
        comments: list[dict[str, str]] = []
        state = {"auto_merge": auto_merge}
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
                        url, 404, "Not Found", {}, io.BytesIO()
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
                    "auto_merge": state["auto_merge"],
                    "head": pr_head,
                    "base": {"ref": base_ref},
                }
            if url.startswith(f"{RUNS_URL}?"):
                return {"workflow_runs": list(workflow_runs or [])}
            if url.startswith(f"{RUNS_URL}/") and method == "POST":
                for bad_id in fail_approve_ids or set():
                    if url.endswith(f"/{bad_id}/approve"):
                        raise urllib.error.HTTPError(
                            url, 403, "Forbidden", {}, io.BytesIO()
                        )
                return {}
            if url == GQL_URL:
                state["auto_merge"] = {"merge_method": "squash"}
                return {"data": {"enablePullRequestAutoMerge": {"pullRequest": {}}}}
            if url.startswith(f"{BK_URL}?"):
                return list(active_builds or [])
            if url.startswith(f"{BK_URL}/") and url.endswith("/cancel"):
                if url.endswith("/905/cancel"):
                    raise urllib.error.HTTPError(
                        url, 422, "Unprocessable", {}, io.BytesIO()
                    )
                return {"state": "canceled"}
            return {"web_url": BK_WEB} if url == BK_URL else {"id": 1}

        return fake_api, calls

    def _run(
        self,
        body: str,
        *,
        role: str = "write",
        pr_state: str = "open",
        base_ref: str = "main",
        head: dict[str, Any] | None = None,
        active_builds: list[dict[str, Any]] | None = None,
        workflow_runs: list[dict[str, Any]] | None = None,
        fail_approve_ids: set[int] | None = None,
    ) -> tuple[str | None, list[tuple[str, str, dict[str, Any] | None]]]:
        fake_api, calls = self._mock_api(
            role=role,
            pr_state=pr_state,
            base_ref=base_ref,
            head=head,
            active_builds=active_builds,
            workflow_runs=workflow_runs,
            fail_approve_ids=fail_approve_ids,
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
                "`/ci run [perf-eval] [nightly-suite]`: TPU tests now.",
                "`/ci merge [perf-eval] [nightly-suite]`",
                "`/ci merge precheck-only`",
                "**Merge when ready** button at the bottom of the GitHub PR page",
                "`/ci cancel`: stop this PR's running builds.",
                "`/ci help`.",
                "`/ci` commands except help need write access.",
            ):
                self.assertIn(expected, reply)
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
            ("/ci", "Missing"),
            ("/ci check", "`check`"),
            ("/ci status", "`status`"),
            ("/ci precheck perf-eval", "`perf-eval`"),
            ("/ci cancel all", "`all`"),
            ("/ci help run", "`run`"),
            ("/ci run fast", "`fast`"),
            ("/ci run precheck-only", "`precheck-only`"),
            ("/ci merge turbo", "`turbo`"),
        ]
        for cmd, token in bad:
            reply, calls = self._run(cmd)
            assert reply is not None
            self.assertNotIn("\n", reply)
            self.assertIn(token, reply)
            self.assertIn("/ci help", reply)
            self.assertFalse(
                any(u.startswith(BK_URL) or u == GQL_URL for _, u, _ in calls)
            )

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
            "Started prechecks on `abcdef1`: "
            "pre-commit re-run (https://gh/runs/10), "
            "PR Text re-run (https://gh/runs/20), "
            f"Buildkite CPU checks ({BK_WEB})",
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
            "Started prechecks on `1234567`: "
            "pre-commit approved (https://gh/runs/51), "
            "PR Text already in_progress (https://gh/runs/52), "
            f"Buildkite CPU checks ({BK_WEB})",
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
            reply, calls = self._run(cmd, role=role)
            assert reply is not None
            self.assertIn(BK_WEB, reply)
            self.assertIn(label, reply)
            bk_calls = [p for m, u, p in calls if m == "POST" and u == BK_URL]
            self.assertEqual(len(bk_calls), 1)
            assert bk_calls[0] is not None
            self.assertEqual(bk_calls[0]["pull_request_id"], 42)
            self.assertEqual(
                bk_calls[0]["env"], {"CI_RUN_TPU": "1", "CI_EXTRAS": expected_extras}
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

    def test_ci_merge_precheck_only_and_rejections(self) -> None:
        reply, calls = self._run("/ci merge precheck-only", role="maintain")
        self.assertEqual(
            reply,
            "Queued for merge with prechecks only (no TPU) once approved and "
            "green. To leave the queue, click **Disable auto-merge** or "
            "**Remove from queue** on the GitHub PR page.\n\n"
            "<!-- ci-merge-extras: precheck-only -->\n"
            "<!-- ci-merge-actor: alice -->",
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
            "Cancelled builds #901, #902. To leave the merge queue, click "
            "**Disable auto-merge** or **Remove from queue** on the GitHub PR page.",
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
        self.assertEqual(empty_reply, "No running CI builds for this PR.")

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
            "`/ci run [perf-eval] [nightly-suite]`",
            "**Merge when ready** button at the bottom of the GitHub PR page",
            "`/ci merge [perf-eval] [nightly-suite]`",
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
