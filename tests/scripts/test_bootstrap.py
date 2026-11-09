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
"""Unit tests for `.buildkite/scripts/bootstrap.py` and its shell wrappers."""

from __future__ import annotations

import contextlib
import http.client
import http.server
import importlib
import io
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from unittest import mock

import yaml  # pyrefly: ignore[missing-import]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / ".buildkite" / "scripts"

_FAKE_AGENT_BODY = """
import json
import os
import pathlib
import sys
import yaml

log_path = pathlib.Path(os.environ["TEST_UPLOAD_LOG"])
args = sys.argv[1:]
if args[:2] == ["meta-data", "set"]:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": [args[2], args[3]]}) + "\\n")
elif args[:2] == ["meta-data", "exists"]:
    entries = (
        [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        if log_path.exists()
        else []
    )
    sys.exit(
        0 if any(e.get("meta", [None])[0] == args[2] for e in entries) else 100
    )
elif args[:2] == ["pipeline", "upload"]:
    content = sys.stdin.read()
    if "--dry-run" in args:
        if os.environ.get("TEST_DRY_RUN_FAIL") == "1":
            sys.exit(1)
        if "TEST_DRY_RUN_RAW" in os.environ:
            sys.stdout.write(os.environ["TEST_DRY_RUN_RAW"])
            sys.exit(0)
        doc = yaml.safe_load(content) or {}
        clean = {k: v for k, v in doc.items() if not k.startswith("x-")}
        json.dump(clean, sys.stdout)
    else:
        print("agent: pipeline upload", flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"args": args[2:], "content": content}) + "\\n")
"""


def _install_upload_stubs(bin_dir: pathlib.Path) -> None:
    agent = bin_dir / "buildkite-agent"
    agent.write_text(
        f"#!{sys.executable}\n{_FAKE_AGENT_BODY.lstrip()}", encoding="utf-8"
    )
    agent.chmod(agent.stat().st_mode | stat.S_IEXEC)
    py3_link = bin_dir / "python3"
    if not py3_link.exists():
        py3_link.symlink_to(sys.executable)


def _expand_env(text: str, env: Mapping[str, str]) -> str:
    def _sub(match: re.Match[str]) -> str:
        key, default = match.group(1), match.group(2)
        val = env.get(key, "")
        return val if val else (default or "")

    return re.sub(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}", _sub, text)


# (extras, actor, error) as returned by `_merge_queue_pr_extras`.
_MqMarker = tuple[tuple[str, ...], str, str | None]


class _RunRecorder:
    """In-memory fake for `_agent`, `_git`, and `_github_labels` in bootstrap runs."""

    def __init__(
        self,
        bp: Any,
        files: list[str] | None = None,
        labels: tuple[str, ...] = (),
        mq_marker: _MqMarker | list[_MqMarker] = ((), "", None),
        fail_git_prefixes: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        self.bp = bp
        self.files = ["src/vllm_torchtpu/envs.py"] if files is None else files
        self.labels = labels
        # A list gives one result per attempt; the last one repeats.
        self.mq_marker = mq_marker
        self.fail_git_prefixes = fail_git_prefixes
        self.meta: list[tuple[str, str]] = []
        self.annotations: list[tuple[str, str, str]] = []
        self.uploads: list[tuple[str, str, dict[str, str]]] = []
        self.git_calls: list[tuple[str, ...]] = []
        self.mq_extras_calls: list[str] = []
        self.github_labels_calls: list[str] = []
        self.sleeps: list[float] = []
        self.stdout_buf = io.StringIO()
        self.stderr_buf = io.StringIO()
        self._current_lane: str | None = None
        self._real_upload = bp.upload_with_priority

    @property
    def stdout(self) -> str:
        return self.stdout_buf.getvalue()

    @property
    def stderr(self) -> str:
        return self.stderr_buf.getvalue()

    def upload_with_priority(self, yaml_file: str, **kwargs: Any) -> None:
        self._current_lane = pathlib.Path(yaml_file).stem.removeprefix("pipeline_")
        try:
            self._real_upload(yaml_file, **kwargs)
        finally:
            self._current_lane = None

    def agent(
        self,
        *args: str,
        input_text: str | None = None,
        check: bool = True,
        env: Mapping[str, str] | None = None,
        cwd: pathlib.Path = _REPO_ROOT,
    ) -> subprocess.CompletedProcess[str]:
        env_map = env or {}
        if args[:2] == ("meta-data", "set"):
            self.meta.append((args[2], args[3]))
        elif args[:2] == ("meta-data", "exists"):
            exists = any(k == args[2] for k, _ in self.meta)
            return subprocess.CompletedProcess(args, 0 if exists else 100, stdout="")
        elif args[:1] == ("annotate",):
            # args: ("annotate", "--style", <style>, "--context", <ctx>, <note>)
            self.annotations.append((args[2], args[4], args[5]))
        elif args[:2] == ("pipeline", "upload") and input_text is not None:
            if "--dry-run" in args:
                doc = yaml.safe_load(_expand_env(input_text, env_map)) or {}
                out = json.dumps(
                    {k: v for k, v in doc.items() if not k.startswith("x-")}
                )
                return subprocess.CompletedProcess(args, 0, stdout=out)
            rendered = (
                input_text
                if "--no-interpolation" in args
                else _expand_env(input_text, env_map)
            )
            tag = self._current_lane or "upload_stages"
            self.uploads.append((tag, rendered, dict(env_map)))
        return subprocess.CompletedProcess(args, 0, stdout="")

    def git(
        self,
        *args: str,
        check: bool = False,
        env: Mapping[str, str] | None = None,
        cwd: pathlib.Path = _REPO_ROOT,
    ) -> subprocess.CompletedProcess[str]:
        self.git_calls.append(args)
        if any(args[: len(p)] == p for p in self.fail_git_prefixes):
            return subprocess.CompletedProcess(args, 1, stdout="")
        if args[:1] == ("ls-remote",):
            return subprocess.CompletedProcess(args, 0, stdout="upstream999\tHEAD\n")
        if args[:1] in (("diff",), ("diff-tree",)):
            return subprocess.CompletedProcess(
                args, 0, stdout="\n".join(self.files) + "\n"
            )
        return subprocess.CompletedProcess(args, 0, stdout="")

    def github_labels(
        self, pr_number: str, *, env: Mapping[str, str] | None = None
    ) -> tuple[str, ...]:
        self.github_labels_calls.append(pr_number)
        return self.labels

    def merge_queue_pr_extras(
        self, pr_number: str, *, env: Mapping[str, str] | None = None
    ) -> _MqMarker:
        self.mq_extras_calls.append(pr_number)
        if isinstance(self.mq_marker, list):
            return self.mq_marker[
                min(len(self.mq_extras_calls), len(self.mq_marker)) - 1
            ]
        return self.mq_marker

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class TestBootstrapUnits(unittest.TestCase):
    """Unit tests for policy tables, chooser functions, helpers, and orchestration."""

    bp: Any
    pinned: str

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        had_bp = "bootstrap" in sys.modules
        had_cap = "cap_pr_tpu_concurrency" in sys.modules
        with mock.patch.object(sys, "path", [str(_SCRIPTS_DIR), *sys.path]):
            cls.bp = importlib.import_module("bootstrap")
        cls.pinned = cls.bp.parse_pinned_vllm_commit(
            (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )

        def _cleanup_modules() -> None:
            if not had_bp:
                sys.modules.pop("bootstrap", None)
            if not had_cap:
                sys.modules.pop("cap_pr_tpu_concurrency", None)

        cls.addClassCleanup(_cleanup_modules)

    def _run_bootstrap(
        self,
        env: dict[str, str],
        *,
        files: list[str] | None = None,
        labels: tuple[str, ...] = (),
        mq_marker: _MqMarker | list[_MqMarker] = ((), "", None),
        fail_git_prefixes: tuple[tuple[str, ...], ...] = (),
        raises: type[Exception] | None = None,
    ) -> _RunRecorder:
        bp = self.bp
        rec = _RunRecorder(
            bp,
            files=files,
            labels=labels,
            mq_marker=mq_marker,
            fail_git_prefixes=fail_git_prefixes,
        )
        with (
            mock.patch.object(bp, "_agent", side_effect=rec.agent),
            mock.patch.object(bp, "_git", side_effect=rec.git),
            mock.patch.object(bp, "_github_labels", side_effect=rec.github_labels),
            mock.patch.object(
                bp, "_merge_queue_pr_extras", side_effect=rec.merge_queue_pr_extras
            ),
            mock.patch.object(
                bp, "upload_with_priority", side_effect=rec.upload_with_priority
            ),
            mock.patch.object(bp.time, "sleep", side_effect=rec.sleep),
            contextlib.redirect_stdout(rec.stdout_buf),
            contextlib.redirect_stderr(rec.stderr_buf),
        ):
            if raises is None:
                self.assertEqual(bp.run_bootstrap(env, repo_root=_REPO_ROOT), 0)
            else:
                with self.assertRaises(raises):
                    bp.run_bootstrap(env, repo_root=_REPO_ROOT)
        return rec

    def _run_next_stage(
        self, prev: _RunRecorder, *, expected_key: str = "upload_stage_1"
    ) -> _RunRecorder:
        bp = self.bp
        self.assertEqual(prev.uploads[0][0], "upload_stages")
        uploader = json.loads(prev.uploads[0][1])
        self.assertEqual(uploader["steps"][0], "wait")
        step = uploader["steps"][1]
        self.assertEqual(step["key"], expected_key)
        self.assertEqual(step["agents"], {"queue": "cpu"})
        self.assertEqual(step["timeout_in_minutes"], 15)
        self.assertEqual(
            step["retry"],
            {
                "manual": {"permit_on_passed": False},
                "automatic": [{"exit_status": -1, "limit": 2}],
            },
        )
        self.assertNotIn("OMNI_IMAGE_SKIP", step["env"])
        prefix = "python3 .buildkite/scripts/bootstrap.py upload-stages "
        self.assertTrue(step["command"].startswith(prefix))
        raw_args = step["command"].removeprefix(prefix).split()
        stages = tuple(bp._split_labels(arg) for arg in raw_args)
        rec = _RunRecorder(bp)
        with (
            mock.patch.object(bp, "_agent", side_effect=rec.agent),
            mock.patch.object(
                bp, "upload_with_priority", side_effect=rec.upload_with_priority
            ),
            contextlib.redirect_stdout(rec.stdout_buf),
            contextlib.redirect_stderr(rec.stderr_buf),
        ):
            self.assertEqual(
                bp.upload_stages(
                    stages,
                    {**step["env"], "BUILDKITE_STEP_KEY": step["key"]},
                    repo_root=_REPO_ROOT,
                ),
                0,
            )
        return rec

    def test_build_context_properties_and_priorities(self) -> None:
        bp = self.bp
        ctx = bp.BuildContext.from_env

        for env, expected_kind, expected_prio in (
            (
                {"NIGHTLY": "1", "BUILDKITE_SOURCE": "schedule"},
                "nightly",
                (1, None),
            ),
            (
                {"TORCHTPU_NIGHTLY": "1", "BUILDKITE_SOURCE": "schedule"},
                "nightly",
                (1, None),
            ),
            (
                {
                    "BUILDKITE_PULL_REQUEST": "123",
                    "BUILDKITE_PULL_REQUEST_LABELS": "ready,oncall-fix",
                },
                "oncall-fix",
                (12, "oncall-fix"),
            ),
            (
                {"BUILDKITE_PULL_REQUEST": "123"},
                "pre-merge",
                (6, "pre-merge"),
            ),
            (
                {"CI_RUN_TPU": "1"},
                "pre-merge",
                (6, "pre-merge"),
            ),
            # /ci builds and post-cutover PR pushes ignore oncall-fix.
            (
                {
                    "BUILDKITE_PULL_REQUEST": "123",
                    "BUILDKITE_PULL_REQUEST_LABELS": "ready,oncall-fix",
                    "CI_RUN_TPU": "1",
                },
                "pre-merge",
                (6, "pre-merge"),
            ),
            (
                {
                    "BUILDKITE_PULL_REQUEST": "123",
                    "BUILDKITE_PULL_REQUEST_LABELS": "ready,oncall-fix",
                    "CI_PR_PUSH_CPU_ONLY": "1",
                },
                "pre-merge",
                (6, "pre-merge"),
            ),
            (
                {
                    "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main",
                    "BUILDKITE_PULL_REQUEST": "123",
                    "BUILDKITE_PULL_REQUEST_LABELS": "oncall-fix",
                },
                "post-merge",
                (11, "post-merge"),
            ),
            (
                {"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"},
                "post-merge",
                (11, "post-merge"),
            ),
            (
                {"BUILDKITE_BRANCH": "release/v1"},
                "other",
                (2, None),
            ),
            (
                {"BUILDKITE_BRANCH": "releases/v0.32.0"},
                "other",
                (2, None),
            ),
            (
                {"BUILDKITE_TAG": "v0.32.0"},
                "other",
                (2, None),
            ),
            (
                {"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration"},
                "other",
                (2, None),
            ),
        ):
            with self.subTest(env=env):
                build_ctx = ctx(env)
                kind = bp.build_kind(build_ctx)
                self.assertEqual(kind, expected_kind)
                self.assertEqual(bp.PRIORITIES[kind], expected_prio)

        # SKIP_TPU_CI="" and "0" are treated as unset; non-zero fails clearly.
        for unset_val in ("", "0"):
            self.assertFalse(ctx({"SKIP_TPU_CI": unset_val}).is_pr)
        with self.assertRaisesRegex(
            bp.BootstrapError,
            r"ERROR: SKIP_TPU_CI was removed; use the skip-tpu-ci PR label "
            r"\(or 'bootstrap\.py upload-stages build'\)",
        ):
            ctx({"SKIP_TPU_CI": "1"})
        with self.assertRaisesRegex(
            bp.BootstrapError,
            r"ERROR: CI_LANES was removed; set CI_SUITE=<name> "
            r"\(suites: default full\)",
        ):
            ctx({"CI_LANES": "perf"})

        # Release-tag / releases/vX.Y.Z branch detection (never true on PRs).
        pr_on_rel = ctx(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_BRANCH": "releases/v0.32.0",
                "BUILDKITE_PULL_REQUEST_LABELS": "ready,oncall-fix",
            }
        )
        self.assertTrue(pr_on_rel.is_pr)
        self.assertFalse(pr_on_rel.is_post_merge)
        self.assertTrue(pr_on_rel.is_oncall_fix)
        self.assertFalse(pr_on_rel.is_release)

        for rel_env in (
            {"BUILDKITE_TAG": "v0.32.0"},
            {"BUILDKITE_TAG": "v0.32.0rc1"},
            {"BUILDKITE_TAG": "0.32.0"},
            {"BUILDKITE_BRANCH": "releases/v0.32.0"},
        ):
            self.assertTrue(ctx(rel_env).is_release)

        for non_rel_env in (
            {"BUILDKITE_BRANCH": "release/v1"},
            {"BUILDKITE_BRANCH": "releases/v01.0.0"},
            {"BUILDKITE_BRANCH": "releases/foo"},
            {"BUILDKITE_TAG": "nightly-20261105"},
        ):
            self.assertFalse(ctx(non_rel_env).is_release)

        # Post-merge webhook on main: only skip-tpu-ci is honored from merged labels.
        pm_base = ctx({"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"})
        self.assertTrue(pm_base.is_post_merge)
        pm_skip = replace(
            pm_base, labels=("skip-tpu-ci", "run-full-ci", "run-perf-eval")
        )
        self.assertFalse(pm_skip.runs_tpu)
        self.assertEqual(pm_skip.extras, frozenset())
        pm_full = replace(pm_base, labels=("run-full-ci", "run-perf-eval"))
        self.assertTrue(pm_full.runs_tpu)
        self.assertEqual(pm_full.extras, frozenset())
        self.assertEqual(bp.choose_suite(pm_full), bp.SUITES["default"])

    def test_skip_docs_vllm_pin_and_merged_pr_parsing(self) -> None:
        bp = self.bp
        self.assertTrue(bp.has_skip_ci_marker("chore: note [skip ci]"))
        self.assertTrue(bp.has_skip_ci_marker("chore: note [ci skip]"))
        self.assertFalse(bp.has_skip_ci_marker("feat: normal change"))

        self.assertFalse(bp.only_docs_changed([]))
        self.assertTrue(
            bp.only_docs_changed(
                ["README.md", "docs/guide.md", ".pyrefly-baseline.json"]
            )
        )
        self.assertFalse(
            bp.only_docs_changed([".pyrefly-baseline-full.json", "README.md"])
        )
        self.assertFalse(bp.only_docs_changed(["src/vllm_torchtpu/envs.py"]))

        pyproject = (
            "[dependency-groups]\nvllm = [\n"
            '  "vllm @ git+https://github.com/vllm-project/vllm.git@abc123def",\n]\n'
        )
        self.assertEqual(bp.parse_pinned_vllm_commit(pyproject), "abc123def")
        # epic/dsv41 pins a vLLM tag in [project] dependencies.
        self.assertEqual(
            bp.parse_pinned_vllm_commit(
                "[project]\ndependencies = [\n"
                '    "vllm @ git+https://github.com/vllm-project/vllm.git@v0.31.1rc0",\n'
                "]\n"
            ),
            "v0.31.1rc0",
        )
        with self.assertRaises(bp.BootstrapError):
            bp.parse_pinned_vllm_commit("[project]\nname = 'vllm-torchtpu'\n")

        self.assertEqual(
            bp.parse_merged_pr_number("feat: add kernel (#789)\nbody (#999)"), "789"
        )
        self.assertIsNone(bp.parse_merged_pr_number("chore: direct commit"))

        ctx = bp.BuildContext.from_env
        self.assertEqual(
            ctx({"BUILDKITE_PULL_REQUEST": "42"}).merge_queue_pr_number, "42"
        )
        self.assertEqual(
            ctx(
                {"BUILDKITE_BRANCH": "gh-readonly-queue/main/pr-55-deadbeef"}
            ).merge_queue_pr_number,
            "55",
        )
        self.assertIsNone(
            ctx({"BUILDKITE_BRANCH": "feature/foo"}).merge_queue_pr_number
        )

        # `/ci merge` bot marker parser (trusts only github-actions[bot], whole-line
        # markers, non-empty actor + matching ci-merge-sha for non-empty extras).
        with self.assertRaises(ValueError):
            bp.parse_merge_queue_comments_page({"not": "a-list"})
        gh_bot = {"type": "Bot", "login": "github-actions[bot]"}
        comments = [
            {"user": {"type": "User"}, "body": "<!-- ci-merge-extras: perf-eval -->"},
            {
                "user": {"type": "Bot", "login": "other-app[bot]"},
                "body": (
                    "<!-- ci-merge-extras: precheck-only -->\n"
                    "<!-- ci-merge-actor: mallory -->\n"
                    "<!-- ci-merge-sha: abcdef1234 -->"
                ),
            },
            {
                "user": gh_bot,
                "body": (
                    ":x: `/ci help` takes no extra arguments "
                    "(got `<!--ci-merge-extras:precheck-only-->`)."
                ),
            },
            {
                "user": gh_bot,
                "body": "<!-- ci-merge-extras: precheck-only -->",
            },
            {
                "user": gh_bot,
                "body": (
                    "<!-- ci-merge-extras: precheck-only -->\n"
                    "<!-- ci-merge-actor: alice -->"
                ),
            },
            {
                "user": gh_bot,
                "body": (
                    "Queued.\n\n"
                    "<!-- ci-merge-extras: perf-eval, nightly-suite -->\n"
                    "<!-- ci-merge-actor: alice -->\n"
                    "<!-- ci-merge-sha: abcdef1234 -->"
                ),
            },
            {
                "user": gh_bot,
                "body": ("<!-- ci-merge-extras:  -->\n<!-- ci-merge-actor: bob -->"),
            },
        ]
        self.assertEqual(
            bp.parse_merge_queue_comments_page(comments[:5], head_sha="abcdef1234"),
            (False, (), ""),
        )
        self.assertEqual(
            bp.parse_merge_queue_comments_page(comments[:6], head_sha="abcdef1234"),
            (True, ("perf-eval", "nightly-suite"), "alice"),
        )
        self.assertEqual(
            bp.parse_merge_queue_comments_page(comments, head_sha="abcdef1234"),
            (True, (), ""),
        )
        # Stale marker before a new push (different head_sha) is ignored.
        self.assertEqual(
            bp.parse_merge_queue_comments_page(
                [
                    {
                        "user": gh_bot,
                        "body": (
                            "<!-- ci-merge-extras: precheck-only -->\n"
                            "<!-- ci-merge-actor: alice -->\n"
                            "<!-- ci-merge-sha: oldsha0000 -->"
                        ),
                    }
                ],
                head_sha="newsha1111",
            ),
            (False, (), ""),
        )
        for bad_marker in (
            "<!-- ci-merge-extras: bogus -->\n<!-- ci-merge-actor: alice -->",
            (
                "<!-- ci-merge-extras: precheck-only, perf-eval -->\n"
                "<!-- ci-merge-actor: alice -->"
            ),
        ):
            with self.assertRaises(ValueError):
                bp.parse_merge_queue_comments_page(
                    [{"user": gh_bot, "body": bad_marker}]
                )

        rendered = bp.prepare_upload_yaml(
            'if: \'build.env("NIGHTLY") == "1"\'\n', job_priority=12, nightly=True
        )
        self.assertEqual(rendered, 'priority: 12\nif: \'"1" == "1"\'\n')

    def test_choose_suite_and_suites_table(self) -> None:
        bp = self.bp
        ctx = bp.BuildContext.from_env
        lanes = (("build", "dsv41"),)

        # 1. PR suite selection (ready and oncall-fix are orthogonal; run-perf-eval
        # adds nothing on epic/dsv41).
        for raw_labels, expected_name, expected_stages, expected_nightly in (
            ("ready", "pr", lanes, False),
            ("ready,oncall-fix", "pr", lanes, False),
            ("ready,oncall-fix,skip-tpu-ci", "pr+build-only", (("build",),), False),
            ("ready,oncall-fix,run-perf-eval", "pr", lanes, False),
            ("ready,oncall-fix,run-full-ci", "pr+nightly-suite", lanes, True),
        ):
            with self.subTest(raw_labels=raw_labels):
                suite = bp.choose_suite(
                    ctx(
                        {
                            "BUILDKITE_PULL_REQUEST": "10",
                            "BUILDKITE_PULL_REQUEST_LABELS": raw_labels,
                        }
                    )
                )
                self.assertEqual(suite.name, expected_name)
                self.assertEqual(suite.stages, expected_stages)
                self.assertEqual(suite.nightly, expected_nightly)

        # 2. Mutually exclusive PR suite labels raise actionable BootstrapError.
        for bad_labels, expected_conflicts in (
            ("skip-tpu-ci,run-perf-eval", "skip-tpu-ci, run-perf-eval"),
            ("run-full-ci,skip-tpu-ci", "skip-tpu-ci, run-full-ci"),
            ("run-perf-eval,run-full-ci", "run-perf-eval, run-full-ci"),
        ):
            with (
                self.subTest(bad_labels=bad_labels),
                self.assertRaisesRegex(
                    bp.BootstrapError,
                    rf"^ERROR: PR labels {re.escape(expected_conflicts)} "
                    r"are mutually exclusive\. Remove all but one of "
                    r"skip-tpu-ci / run-perf-eval / run-full-ci in the "
                    r"\*\*Labels\*\* sidebar on the GitHub PR page, then retry "
                    r"the \*\*Bootstrap Pipeline\*\* step\.$",
                ),
            ):
                bp.choose_suite(
                    ctx(
                        {
                            "BUILDKITE_PULL_REQUEST": "42",
                            "BUILDKITE_PULL_REQUEST_LABELS": bad_labels,
                        }
                    )
                )

        # 3. CI_SUITE override beats skip-tpu-ci and PR labels; main's removed
        # suites (perf, integration) and unknown names raise.
        self.assertEqual(
            bp.choose_suite(
                ctx(
                    {
                        "CI_SUITE": "full",
                        "BUILDKITE_PULL_REQUEST": "10",
                        "BUILDKITE_PULL_REQUEST_LABELS": "skip-tpu-ci",
                    }
                )
            ),
            bp.SUITES["full"],
        )
        for bad_suite in ("bogus", "cpu-only", "perf", "integration"):
            with self.assertRaisesRegex(
                bp.BootstrapError,
                rf"ERROR: unknown CI_SUITE '{bad_suite}'; suites are: default full",
            ):
                bp.choose_suite(ctx({"CI_SUITE": bad_suite}))

        # 4. Merge-queue, /ci, post-merge skip-tpu-ci, full, and default.
        self.assertEqual(
            bp.choose_suite(ctx({"CI_PRECHECK": "1", "CI_EXTRAS": "perf-eval"})).name,
            "pr+build-only",
        )
        with self.assertRaisesRegex(bp.BootstrapError, "CI_EXTRAS names 'bogus'"):
            bp.choose_suite(ctx({"CI_RUN_TPU": "1", "CI_EXTRAS": "bogus"}))
        for mq_env, mq_extras, expected_name in (
            ({"BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main"}, (), "pr"),
            (
                {"BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main"},
                ("precheck-only",),
                "pr+build-only",
            ),
            ({"BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main"}, ("perf-eval",), "pr"),
            (
                {"BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main"},
                ("nightly-suite",),
                "pr+nightly-suite",
            ),
            (
                {"BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "releases/v0.32.0"},
                ("precheck-only",),
                "pr+nightly-suite",
            ),
            (
                {"BUILDKITE_PULL_REQUEST": "10", "CI_PR_PUSH_CPU_ONLY": "1"},
                (),
                "pr+build-only",
            ),
            (
                {
                    "BUILDKITE_PULL_REQUEST": "10",
                    "CI_PR_PUSH_CPU_ONLY": "1",
                    "CI_RUN_TPU": "1",
                },
                (),
                "pr",
            ),
            ({"CI_RUN_TPU": "1", "CI_EXTRAS": "perf-eval"}, (), "pr"),
            ({"CI_RUN_TPU": "1", "CI_EXTRAS": "nightly-suite"}, (), "pr+nightly-suite"),
            (
                {
                    "BUILDKITE_PULL_REQUEST": "10",
                    "BUILDKITE_PULL_REQUEST_LABELS": "skip-tpu-ci,run-perf-eval",
                    "CI_PRECHECK": "1",
                },
                (),
                "pr+build-only",
            ),
        ):
            with self.subTest(mq_env=mq_env, mq_extras=mq_extras):
                self.assertEqual(
                    bp.choose_suite(replace(ctx(mq_env), mq_extras=mq_extras)).name,
                    expected_name,
                )

        pm_skip = replace(
            ctx({"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"}),
            labels=("skip-tpu-ci",),
        )
        self.assertEqual(bp.choose_suite(pm_skip).name, "pr+build-only")

        for full_env in (
            {"NIGHTLY": "1"},
            {"BUILDKITE_TAG": "v0.32.0"},
            {"BUILDKITE_BRANCH": "releases/v0.32.0"},
        ):
            suite = bp.choose_suite(ctx(full_env))
            self.assertEqual(suite, bp.SUITES["full"])
            self.assertTrue(suite.nightly)

        for default_env in (
            {"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"},
            {"BUILDKITE_BRANCH": "epic/dsv41", "BUILDKITE_SOURCE": "webhook"},
            # No integration pipeline on epic: its slug gets the default suite.
            {"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration"},
            {"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci"},
        ):
            suite = bp.choose_suite(ctx(default_env))
            self.assertEqual(suite, bp.SUITES["default"])
            self.assertFalse(suite.nightly)

        # 5. Every suite is one stage of build (+ dsv41), and every lane exists.
        self.assertEqual(set(bp.SUITES), {"default", "full"})
        all_suites = (*bp.SUITES.values(), *bp._PR_SUITES.values())
        for suite in all_suites:
            self.assertTrue(suite.about)
            self.assertEqual(len(suite.stages), 1)
            self.assertIn(suite.stages[0], (("build",), ("build", "dsv41")))
            for lane in suite.stages[0]:
                self.assertTrue(
                    (_REPO_ROOT / f".buildkite/pipeline_{lane}.yml").is_file()
                )

    def test_github_labels_and_merge_queue_comments_resilience(self) -> None:
        bp = self.bp
        seen_auth: list[str | None] = []
        mode = "labels"

        class AuthHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                auth = self.headers.get("Authorization")
                seen_auth.append(auth)
                if mode == "auth_401" and auth is not None:
                    self.send_response(401)
                    self.end_headers()
                    return
                if mode == "both_500":
                    self.send_response(500)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if "comments" in self.path:
                    if mode == "bad_json":
                        self.wfile.write(b"{not-json")
                    elif mode == "cap_50_pages" or (
                        mode == "page_full" and self.path.endswith("page=1")
                    ):
                        items = [{"user": {"type": "User"}, "body": "x"}] * 100
                        self.wfile.write(json.dumps(items).encode("utf-8"))
                    else:
                        sha = "oldsha0000" if mode == "stale_marker" else "deadbeef1234"
                        extra = (
                            "precheck-only" if mode == "stale_marker" else "perf-eval"
                        )
                        payload = [
                            {
                                "user": {
                                    "type": "Bot",
                                    "login": "github-actions[bot]",
                                },
                                "body": (
                                    f"<!-- ci-merge-extras: {extra} -->\n"
                                    "<!-- ci-merge-actor: alice -->\n"
                                    f"<!-- ci-merge-sha: {sha} -->"
                                ),
                            }
                        ]
                        self.wfile.write(json.dumps(payload).encode("utf-8"))
                elif "pulls/" in self.path:
                    self.wfile.write(b'{"head": {"sha": "deadbeef1234"}}')
                else:
                    self.wfile.write(b'[{"name": "skip-tpu-ci"}]')

            def log_message(self, format: str, *args: object) -> None:
                return

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            helper = tmp_path / "git-credential-github-app"
            helper.write_text(
                "#!/usr/bin/env bash\necho password=ghs_tok\n", encoding="utf-8"
            )
            helper.chmod(helper.stat().st_mode | stat.S_IEXEC)

            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), AuthHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                api_url = f"http://127.0.0.1:{server.server_address[1]}"
                with (
                    mock.patch.object(bp, "_GITHUB_API_URL", api_url),
                    mock.patch.object(bp, "_CREDENTIAL_HELPER", helper),
                ):
                    self.assertEqual(bp._github_labels("42", env={}), ("skip-tpu-ci",))
                    self.assertEqual(seen_auth, ["token ghs_tok"])
                    mode = "auth_401"
                    self.assertEqual(
                        bp._merge_queue_pr_extras("42", env={}),
                        (("perf-eval",), "alice", None),
                    )
                    mode = "page_full"
                    self.assertEqual(
                        bp._merge_queue_pr_extras("42", env={}),
                        (("perf-eval",), "alice", None),
                    )
                    mode = "stale_marker"
                    self.assertEqual(
                        bp._merge_queue_pr_extras("42", env={}),
                        ((), "", None),
                    )
                    mode = "cap_50_pages"
                    self.assertEqual(
                        bp._merge_queue_pr_extras("42", env={}),
                        ((), "", "comment history exceeds 5000 comments"),
                    )
                    mode = "bad_json"
                    self.assertIsNotNone(bp._merge_queue_pr_extras("42", env={})[2])
                    mode = "both_500"
                    self.assertEqual(
                        bp._merge_queue_pr_extras("42", env={}),
                        ((), "", "HTTP 500 Internal Server Error"),
                    )
                with (
                    mock.patch.object(bp, "_GITHUB_API_URL", api_url),
                    mock.patch.object(
                        bp, "_CREDENTIAL_HELPER", tmp_path / "missing-helper"
                    ),
                ):
                    mode = "both_500"
                    self.assertEqual(
                        bp._merge_queue_pr_extras("42", env={}),
                        ((), "", "HTTP 500 Internal Server Error"),
                    )
                    mode = "labels"
                    self.assertEqual(bp._github_labels("42", env={}), ("skip-tpu-ci",))
                    self.assertIsNone(seen_auth[-1])
            finally:
                server.shutdown()
                server.server_close()

            with (
                mock.patch.object(bp, "_CREDENTIAL_HELPER", tmp_path / "missing"),
                mock.patch(
                    "urllib.request.urlopen",
                    side_effect=http.client.IncompleteRead(b"partial", 10),
                ),
            ):
                self.assertEqual(bp._github_labels("42", env={}), ())

    def test_dev_mode_vllm_pin_and_shell_wrappers(self) -> None:
        bp = self.bp

        # `bootstrap.py vllm-pin` prints the pin and reports missing/read errors.
        good_cli = subprocess.run(
            [sys.executable, str(_SCRIPTS_DIR / "bootstrap.py"), "vllm-pin"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(good_cli.stdout.strip(), self.pinned)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_repo = pathlib.Path(tmp) / "repo"
            shutil.copytree(_SCRIPTS_DIR, tmp_repo / ".buildkite" / "scripts")
            with self.assertRaisesRegex(bp.BootstrapError, "No such file"):
                bp.read_pinned_vllm_commit(tmp_repo)
            (tmp_repo / "pyproject.toml").write_text(
                "[project]\nname = 'empty'\n", encoding="utf-8"
            )
            bad_cli = subprocess.run(
                [
                    sys.executable,
                    str(tmp_repo / ".buildkite" / "scripts" / "bootstrap.py"),
                    "vllm-pin",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(bad_cli.returncode, 1)
            self.assertIn(
                "ERROR: vLLM commit hash is missing from pyproject.toml.",
                bad_cli.stderr,
            )

            tmp_path = pathlib.Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            _install_upload_stubs(bin_dir)
            log_path = tmp_path / "uploads.jsonl"

            sample_yaml = tmp_path / "sample.yml"
            sample_yaml.write_text(
                "steps:\n"
                '  - label: "TPU step"\n'
                '    if: \'build.env("NIGHTLY") == "1"\'\n'
                "    agents:\n"
                '      queue: "kube"\n'
                "    env:\n"
                '      SHAPE: "tpu7x-standard-1t/1x1x1"\n'
                "    commands:\n"
                "      - echo hi\n",
                encoding="utf-8",
            )
            env = {
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                "HOME": str(tmp_path),
                "TEST_UPLOAD_LOG": str(log_path),
                "DEV_PIPELINE_FILE": str(sample_yaml),
                "NIGHTLY": "1",
                "PR_TPU_CONCURRENCY_GROUP": "tpu-pr/dev@example.com",
            }
            # 1. `bootstrap.sh` wrapper execs `bootstrap.py`.
            ci_proc = subprocess.run(
                ["bash", str(_SCRIPTS_DIR / "bootstrap.sh")],
                cwd=_REPO_ROOT,
                env={**env, "BUILDKITE_MESSAGE": "chore [skip ci]"},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Commit message contains [skip ci].", ci_proc.stdout)

            # 2. `bootstrap_dev.sh` capped upload with unbuffered `---` headers.
            capped = subprocess.run(
                ["bash", str(_SCRIPTS_DIR / "bootstrap_dev.sh")],
                cwd=_REPO_ROOT,
                env={**env, "DEV_JOB_PRIORITY": "3"},
                capture_output=True,
                text=True,
                check=True,
            )
            lines = capped.stdout.splitlines()
            hdr = lines.index(f"--- :pipeline: Uploading {sample_yaml} with priority 3")
            self.assertEqual(lines[hdr + 1], "agent: pipeline upload")
            self.assertEqual(lines[-1], "--- Buildkite Dev Bootstrap Finished")

            # 3. Dry-run exit!=0 and malformed/non-dict JSON warn and upload uncapped.
            for extra_env in (
                {"TEST_DRY_RUN_FAIL": "1"},
                {"TEST_DRY_RUN_RAW": "{not-json"},
                {"TEST_DRY_RUN_RAW": "[1, 2]"},
                {
                    "TEST_DRY_RUN_RAW": (
                        '{"steps": [{"agents": {"queue": "kube"}, "env": ["bad"]}]}'
                    )
                },
            ):
                uncapped = subprocess.run(
                    ["bash", str(_SCRIPTS_DIR / "bootstrap_dev.sh")],
                    cwd=_REPO_ROOT,
                    env={**env, "DEV_JOB_PRIORITY": "9", **extra_env},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("WARNING: could not put", uncapped.stderr)

            events = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            uploads = [e for e in events if "args" in e]
            self.assertEqual(len(uploads), 5)
            self.assertEqual(uploads[0]["args"], ["--no-interpolation"])
            parsed = json.loads(uploads[0]["content"])
            self.assertEqual(parsed["priority"], 3)
            self.assertEqual(parsed["steps"][0]["if"], '"1" == "1"')
            self.assertEqual(
                parsed["steps"][0]["concurrency_group"], "tpu-pr/dev@example.com"
            )
            self.assertEqual(uploads[1]["args"], [])
            self.assertTrue(uploads[1]["content"].startswith("priority: 9\n"))

            # 4. Invalid DEV_JOB_PRIORITY sets metadata first (parity), then fails.
            log_path.unlink()
            bad_prio = subprocess.run(
                ["bash", str(_SCRIPTS_DIR / "bootstrap_dev.sh")],
                cwd=_REPO_ROOT,
                env={**env, "DEV_JOB_PRIORITY": "not-an-int"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(bad_prio.returncode, 1)
            self.assertIn("invalid DEV_JOB_PRIORITY 'not-an-int'", bad_prio.stderr)
            bad_events = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn({"meta": ["JOB_PRIORITY", "not-an-int"]}, bad_events)

            # 5. Missing DEV_PIPELINE_FILE exits 1 with branch/commit message.
            missing = subprocess.run(
                ["bash", str(_SCRIPTS_DIR / "bootstrap_dev.sh")],
                cwd=_REPO_ROOT,
                env={
                    **env,
                    "DEV_PIPELINE_FILE": ".buildkite/missing_dev.yml",
                    "BUILDKITE_BRANCH": "exp/missing",
                    "BUILDKITE_COMMIT": "deadbeef",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(missing.returncode, 1)
            self.assertIn(
                "dev pipeline file '.buildkite/missing_dev.yml' not found on "
                "branch 'exp/missing' at commit deadbeef.",
                missing.stdout,
            )

            # 6. `bootstrap.py upload-stages <stage> [<stage> ...]` CLI uploads a stage.
            log_path.unlink()
            upload_cli = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPTS_DIR / "bootstrap.py"),
                    "upload-stages",
                    "build",
                ],
                cwd=_REPO_ROOT,
                env={**env, "JOB_PRIORITY": "6"},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn(
                "--- :pipeline: Uploading .buildkite/pipeline_build.yml "
                "with priority 6",
                upload_cli.stdout,
            )
            for bad_args, expected_err in (
                (["upload-stages"], bp._USAGE),
                (["upload-stages", ""], bp._USAGE),
                (
                    ["upload-stages", "bogus"],
                    "ERROR: lane file '.buildkite/pipeline_bogus.yml' not found",
                ),
                (["bogus-cmd"], bp._USAGE),
            ):
                bad_proc = subprocess.run(
                    [sys.executable, str(_SCRIPTS_DIR / "bootstrap.py"), *bad_args],
                    cwd=_REPO_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(bad_proc.returncode, 1)
                self.assertIn(expected_err, bad_proc.stderr)

    def _assert_capped(self, rec: _RunRecorder, group: str | None) -> None:
        """`rec` ran with `group`: with a group every chip-holding kube step has a
        concurrency group (the author's, or one the step names itself); without
        one no step is in an author's group."""
        cap = self.bp.cap_pr_tpu_concurrency
        for _, content, env in rec.uploads:
            self.assertEqual(env.get("PR_TPU_CONCURRENCY_GROUP"), group)
            doc = yaml.safe_load(content) or {}

            def _walk(steps: list[Any]) -> None:
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    if isinstance(step.get("steps"), list):
                        _walk(step["steps"])
                    elif group is None:
                        self.assertFalse(
                            str(step.get("concurrency_group", "")).startswith("tpu-pr/")
                        )
                    elif cap.holds_chips(step):
                        self.assertIn("concurrency_group", step)

            _walk(doc.get("steps") or [])

    def test_orchestration_pr_and_labels(self) -> None:
        bp = self.bp

        # Conflicting PR suite labels post a ci-labels error annotation and raise.
        rec = _RunRecorder(bp)
        with (
            mock.patch.object(bp, "_agent", side_effect=rec.agent),
            self.assertRaisesRegex(
                bp.BootstrapError,
                r"^ERROR: PR labels skip-tpu-ci, run-full-ci are mutually exclusive\. "
                r"Remove all but one of skip-tpu-ci / run-perf-eval / run-full-ci",
            ) as ctx_err,
        ):
            bp.run_bootstrap(
                {
                    "BUILDKITE_PULL_REQUEST": "42",
                    "BUILDKITE_PULL_REQUEST_LABELS": "ready,skip-tpu-ci,run-full-ci",
                },
                repo_root=_REPO_ROOT,
            )
        self.assertEqual(
            rec.annotations,
            [("error", "ci-labels", str(ctx_err.exception))],
        )
        self.assertEqual(rec.uploads, [])

        # PR default: priority 6, one stage uploading dsv41 then build (Buildkite
        # inserts each upload after the running step, so build runs first), with
        # the author's concurrency group and no -omni image.
        pr = self._run_bootstrap(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_BASE_BRANCH": "epic/dsv41",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com",
            }
        )
        self.assertEqual(
            pr.meta,
            [
                ("JOB_PRIORITY", "6"),
                ("WORKLOAD_PRIORITY", "pre-merge"),
                ("VLLM_COMMIT_HASH", self.pinned),
                ("upload-stages:upload_stage_0:dsv41", "1"),
                ("upload-stages:upload_stage_0:build", "1"),
            ],
        )
        self.assertEqual(self.pinned, "v0.33.0")
        self.assertEqual(
            pr.annotations,
            [("info", "ci-suite", f"CI suite: pr: {bp._PR_SUITES['pr'].about}")],
        )
        self.assertEqual([u[0] for u in pr.uploads], ["dsv41", "build"])
        self.assertIn(
            ("fetch", "origin", "epic/dsv41", "--depth=20", "--quiet"), pr.git_calls
        )
        for _, _, env in pr.uploads:
            self.assertTrue(env["OMNI_IMAGE_SKIP"])
            self.assertLessEqual(len(env["OMNI_IMAGE_SKIP"]), 70)
        self._assert_capped(pr, "tpu-pr/alice@example.com")
        self.assertEqual(pr.stderr, "")
        for expected_line in (
            "Build type: pre-merge - Priority: 6",
            "--- :pipeline: Suite pr (priority 6)",
            "Stage 1 now: build, dsv41",
            "--- :pipeline: Uploading .buildkite/pipeline_dsv41.yml with priority 6",
            "--- :pipeline: Uploading .buildkite/pipeline_build.yml with priority 6",
        ):
            self.assertIn(expected_line, pr.stdout)
        self.assertNotIn("Stage 2", pr.stdout)

        # run-perf-eval is accepted and runs the same suite as no label.
        perf_eval_pr = self._run_bootstrap(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "ready,run-perf-eval",
            }
        )
        self.assertTrue(perf_eval_pr.annotations[0][2].startswith("CI suite: pr: "))
        self.assertEqual([u[0] for u in perf_eval_pr.uploads], ["dsv41", "build"])
        self.assertNotIn("NIGHTLY", perf_eval_pr.uploads[0][2])

        # run-full-ci + oncall-fix: priority 12, oncall-fix warning, NIGHTLY=1
        # substituted in every uploaded file.
        oncall_full = self._run_bootstrap(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "run-full-ci,oncall-fix",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Oncall@Example.com",
            }
        )
        self.assertIn(("JOB_PRIORITY", "12"), oncall_full.meta)
        self.assertIn(("WORKLOAD_PRIORITY", "oncall-fix"), oncall_full.meta)
        self.assertEqual(oncall_full.annotations[0][:2], ("warning", "oncall-fix"))
        self.assertTrue(
            oncall_full.annotations[1][2].startswith(
                "CI suite: pr+nightly-suite [NIGHTLY=1]: "
            )
        )
        self.assertEqual([u[0] for u in oncall_full.uploads], ["dsv41", "build"])
        self._assert_capped(oncall_full, "tpu-pr/oncall@example.com")
        for _, content, env in oncall_full.uploads:
            self.assertEqual(env["NIGHTLY"], "1")
            self.assertNotIn('build.env("NIGHTLY")', content)

        # skip-tpu-ci on a PR, and the post-merge build of such a PR on main,
        # upload only the image build.
        for fx, prio, wprio in (
            (
                self._run_bootstrap(
                    {
                        "BUILDKITE_PULL_REQUEST": "42",
                        "BUILDKITE_PULL_REQUEST_LABELS": "ready,skip-tpu-ci",
                    }
                ),
                "6",
                "pre-merge",
            ),
            (
                self._run_bootstrap(
                    {
                        "BUILDKITE_BRANCH": "main",
                        "BUILDKITE_SOURCE": "webhook",
                        "BUILDKITE_MESSAGE": "fix: tweak (#99)",
                    },
                    labels=("skip-tpu-ci",),
                ),
                "11",
                "post-merge",
            ),
        ):
            self.assertIn(("JOB_PRIORITY", prio), fx.meta)
            self.assertIn(("WORKLOAD_PRIORITY", wprio), fx.meta)
            self.assertTrue(
                fx.annotations[0][2].startswith("CI suite: pr+build-only: ")
            )
            self.assertEqual([u[0] for u in fx.uploads], ["build"])
            self.assertTrue(fx.uploads[0][2]["OMNI_IMAGE_SKIP"])

    def test_orchestration_branch_schedule_and_skip_flows(self) -> None:
        # A push to epic/dsv41 runs the default suite at "other" priority without
        # looking up merged-PR labels (only main post-merge builds do).
        epic_push = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": "epic/dsv41",
                "BUILDKITE_SOURCE": "webhook",
                "BUILDKITE_MESSAGE": "feat: land change (#100)",
            },
            labels=("skip-tpu-ci",),
        )
        self.assertEqual(epic_push.github_labels_calls, [])
        self.assertEqual(epic_push.meta[0], ("JOB_PRIORITY", "2"))
        self.assertNotIn("WORKLOAD_PRIORITY", dict(epic_push.meta))
        self.assertTrue(epic_push.annotations[0][2].startswith("CI suite: default: "))
        self.assertEqual([u[0] for u in epic_push.uploads], ["dsv41", "build"])
        self._assert_capped(epic_push, None)

        # main post-merge ignores run-full-ci / run-perf-eval from the merged PR.
        main_pm = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": "main",
                "BUILDKITE_SOURCE": "webhook",
                "BUILDKITE_MESSAGE": "feat: land change (#100)",
            },
            labels=("run-full-ci", "run-perf-eval"),
        )
        self.assertEqual(main_pm.github_labels_calls, ["100"])
        self.assertTrue(main_pm.annotations[0][2].startswith("CI suite: default: "))
        self.assertEqual([u[0] for u in main_pm.uploads], ["dsv41", "build"])
        self.assertNotIn("NIGHTLY", main_pm.uploads[0][2])

        # Scheduled nightly, release builds and manual CI_SUITE=full run the full
        # suite with NIGHTLY=1 substituted.
        nightly = self._run_bootstrap(
            {
                "NIGHTLY": "1",
                "BUILDKITE_SOURCE": "schedule",
                "BUILDKITE_BRANCH": "epic/dsv41",
            }
        )
        self.assertEqual(
            nightly.meta[:2], [("JOB_PRIORITY", "1"), ("VLLM_COMMIT_HASH", self.pinned)]
        )
        tag_fx = self._run_bootstrap(
            {"BUILDKITE_TAG": "v0.32.0", "BUILDKITE_BRANCH": "v0.32.0"}
        )
        manual_full = self._run_bootstrap(
            {"CI_SUITE": "full", "BUILDKITE_BRANCH": "exp/manual"}
        )
        for fx in (nightly, tag_fx, manual_full):
            self.assertTrue(
                fx.annotations[0][2].startswith("CI suite: full [NIGHTLY=1]: ")
            )
            self.assertEqual([u[0] for u in fx.uploads], ["dsv41", "build"])
            for _, content, _ in fx.uploads:
                self.assertNotIn('build.env("NIGHTLY")', content)

        # A build on the integration pipeline's slug uses the pin and uploads no
        # promote lane: epic has no LKG promotion.
        integ = self._run_bootstrap(
            {
                "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration",
                "BUILDKITE_SOURCE": "schedule",
            }
        )
        self.assertIn(("VLLM_COMMIT_HASH", self.pinned), integ.meta)
        self.assertFalse(any(c[:1] == ("ls-remote",) for c in integ.git_calls))
        self.assertEqual([u[0] for u in integ.uploads], ["dsv41", "build"])

        # VLLM_COMMIT_HASH_OVERRIDE replaces the pin.
        override = self._run_bootstrap({"VLLM_COMMIT_HASH_OVERRIDE": "abc123"})
        self.assertIn(("VLLM_COMMIT_HASH", "abc123"), override.meta)

        # [skip ci] and docs-only PRs exit 0 before uploading any pipeline steps.
        self.assertEqual(
            self._run_bootstrap({"BUILDKITE_MESSAGE": "doc [skip ci]"}).uploads, []
        )
        self.assertEqual(
            self._run_bootstrap(
                {"BUILDKITE_PULL_REQUEST": "42"}, files=["README.md"]
            ).uploads,
            [],
        )

    def test_upload_stages_chaining_and_retry(self) -> None:
        bp = self.bp

        # No epic suite has two stages, but upload_stages keeps main's chaining:
        # a two-stage suite uploads stage 0 plus a `wait` and the next uploader.
        two_stage = bp.Suite("two-stage", "test", stages=(("build",), ("dsv41",)))
        with mock.patch.dict(bp.SUITES, {"two-stage": two_stage}):
            s0 = self._run_bootstrap(
                {
                    "CI_SUITE": "two-stage",
                    "CI_FLEET": "kube",
                    "PR_TPU_CONCURRENCY_GROUP": "tpu-pr/x@example.com",
                }
            )
        self.assertEqual([u[0] for u in s0.uploads], ["upload_stages", "build"])
        self.assertIn(
            "--- :pipeline: Stage 2 is uploaded by step upload_stage_1 after a wait",
            s0.stdout,
        )
        step_env = json.loads(s0.uploads[0][1])["steps"][1]["env"]
        self.assertEqual(step_env["CI_FLEET"], "kube")
        s1 = self._run_next_stage(s0)
        self.assertEqual([u[0] for u in s1.uploads], ["dsv41"])
        self.assertEqual(s1.meta, [("upload-stages:upload_stage_1:dsv41", "1")])
        self._assert_capped(s1, "tpu-pr/x@example.com")

        # An automatic retry after agent loss skips lanes already uploaded.
        retry_rec = _RunRecorder(bp)
        retry_rec.meta = [("upload-stages:upload_stage_0:dsv41", "1")]
        with (
            mock.patch.object(bp, "_agent", side_effect=retry_rec.agent),
            mock.patch.object(
                bp, "upload_with_priority", side_effect=retry_rec.upload_with_priority
            ),
            contextlib.redirect_stdout(retry_rec.stdout_buf),
            contextlib.redirect_stderr(retry_rec.stderr_buf),
        ):
            self.assertEqual(
                bp.upload_stages(
                    (("build", "dsv41"),), {"JOB_PRIORITY": "6"}, repo_root=_REPO_ROOT
                ),
                0,
            )
        self.assertEqual([u[0] for u in retry_rec.uploads], ["build"])
        self.assertIn(
            "Skipping dsv41 (already uploaded for upload_stage_0)", retry_rec.stdout
        )

    def test_orchestration_merge_queue_and_ci_commands(self) -> None:
        # /ci run with CI_EXTRAS=perf-eval runs the pr suite with the author group.
        ci_run = self._run_bootstrap(
            {
                "CI_RUN_TPU": "1",
                "CI_EXTRAS": "perf-eval",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com",
                "BUILDKITE_MESSAGE": "feat: x [skip ci]",
            }
        )
        self.assertIn(("JOB_PRIORITY", "6"), ci_run.meta)
        self.assertTrue(ci_run.annotations[0][2].startswith("CI suite: pr: "))
        self.assertEqual([u[0] for u in ci_run.uploads], ["dsv41", "build"])
        self._assert_capped(ci_run, "tpu-pr/alice@example.com")

        # CI_PRECHECK=1 runs the image build only, even on [skip ci] / docs-only.
        pre = self._run_bootstrap(
            {
                "CI_PRECHECK": "1",
                "CI_RUN_TPU": "1",
                "BUILDKITE_MESSAGE": "docs [skip ci]",
            },
            files=["README.md"],
        )
        self.assertIn(("JOB_PRIORITY", "6"), pre.meta)
        self.assertTrue(pre.annotations[0][2].startswith("CI suite: pr+build-only: "))
        self.assertEqual([u[0] for u in pre.uploads], ["build"])

        # Merge queue: priority 11 (post-merge), ignores PR labels and
        # CI_PR_PUSH_CPU_ONLY, diffs against the queue base commit, runs uncapped.
        mq = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": "gh-readonly-queue/epic/dsv41/pr-42-deadbeef",
                "BUILDKITE_COMMIT": "deadbeef",
                "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "epic/dsv41",
                "BUILDKITE_MERGE_QUEUE_BASE_COMMIT": "base111",
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "skip-tpu-ci,oncall-fix",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com",
                "BUILDKITE_MESSAGE": "feat: x [skip ci]",
                "CI_PR_PUSH_CPU_ONLY": "1",
            }
        )
        self.assertEqual(
            mq.git_calls[:3],
            [
                ("fetch", "origin", "base111", "--depth=20", "--quiet"),
                ("fetch", "origin", "deadbeef", "--depth=20", "--quiet"),
                ("diff", "--name-only", "base111...deadbeef"),
            ],
        )
        self.assertEqual(mq.mq_extras_calls, ["42"])
        self.assertIn(("JOB_PRIORITY", "11"), mq.meta)
        self.assertIn(("WORKLOAD_PRIORITY", "post-merge"), mq.meta)
        self.assertTrue(mq.annotations[0][2].startswith("CI suite: pr: "))
        self.assertEqual([u[0] for u in mq.uploads], ["dsv41", "build"])
        self._assert_capped(mq, None)

        # precheck-only marker: image build only.
        mq_pre = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": "gh-readonly-queue/epic/dsv41/pr-42-deadbeef",
                "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "epic/dsv41",
            },
            mq_marker=(("precheck-only",), "alice", None),
        )
        self.assertTrue(
            mq_pre.annotations[0][2].startswith("CI suite: pr+build-only: ")
        )
        self.assertEqual([u[0] for u in mq_pre.uploads], ["build"])

        # An unreadable /ci merge marker is retried, then fails the build.
        mq_env = {
            "BUILDKITE_BRANCH": "gh-readonly-queue/epic/dsv41/pr-42-deadbeef",
            "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "epic/dsv41",
        }
        mq_fail = self._run_bootstrap(
            mq_env, mq_marker=((), "", "HTTP 500"), raises=self.bp.BootstrapError
        )
        self.assertEqual(len(mq_fail.mq_extras_calls), 3)
        self.assertEqual(mq_fail.sleeps, [10, 20])
        self.assertEqual(mq_fail.annotations[-1][:2], ("error", "ci-merge-extras"))
        self.assertEqual(mq_fail.uploads, [])

        # Docs-only merge-queue build posts docs-only info annotation and skips.
        mq_docs = self._run_bootstrap(mq_env, files=["README.md"])
        self.assertEqual(
            mq_docs.annotations,
            [
                (
                    "info",
                    "docs-only",
                    (
                        "Docs-only change; skipped the image build and TPU "
                        "checks in the merge queue."
                    ),
                )
            ],
        )
        self.assertEqual(mq_docs.uploads, [])

    def test_scope_equivalence_across_channels(self) -> None:
        """Every channel for a scope uploads the same steps; only priority and
        concurrency differ."""

        def _normalize(rec: _RunRecorder) -> list[tuple[str, Any]]:
            normalized: list[tuple[str, Any]] = []
            for lane, content, _ in rec.uploads:
                raw_doc = yaml.safe_load(content) or {}
                doc = {
                    k: v
                    for k, v in raw_doc.items()
                    if k != "priority" and not k.startswith("x-")
                }
                for step in doc.get("steps") or []:
                    if isinstance(step, dict):
                        for item in step.get("steps") or [step]:
                            if isinstance(item, dict):
                                for key in (
                                    "concurrency",
                                    "concurrency_group",
                                    "concurrency_method",
                                ):
                                    item.pop(key, None)
                normalized.append((lane, doc))
            return normalized

        author = {"BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com"}
        mq_base = {
            "BUILDKITE_BRANCH": "gh-readonly-queue/epic/dsv41/pr-42-deadbeef",
            "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "epic/dsv41",
            **author,
        }
        pr = {"BUILDKITE_PULL_REQUEST": "42", **author}
        for scope, channels in (
            (
                "pr",
                (
                    (pr, (), ()),
                    ({**pr, "BUILDKITE_PULL_REQUEST_LABELS": "run-perf-eval"}, (), ()),
                    (mq_base, (), ()),
                    (mq_base, (), ("perf-eval",)),
                    ({"CI_RUN_TPU": "1", **author}, (), ()),
                    ({"CI_RUN_TPU": "1", "CI_EXTRAS": "perf-eval", **author}, (), ()),
                ),
            ),
            (
                "pr+nightly-suite",
                (
                    ({**pr, "BUILDKITE_PULL_REQUEST_LABELS": "run-full-ci"}, (), ()),
                    (mq_base, (), ("nightly-suite",)),
                    (
                        {"CI_RUN_TPU": "1", "CI_EXTRAS": "nightly-suite", **author},
                        (),
                        (),
                    ),
                ),
            ),
            (
                "pr+build-only",
                (
                    ({**pr, "BUILDKITE_PULL_REQUEST_LABELS": "skip-tpu-ci"}, (), ()),
                    ({**pr, "CI_PR_PUSH_CPU_ONLY": "1"}, (), ()),
                    (mq_base, (), ("precheck-only",)),
                    ({"CI_PRECHECK": "1", **author}, (), ()),
                    (
                        {
                            "BUILDKITE_BRANCH": "main",
                            "BUILDKITE_SOURCE": "webhook",
                            "BUILDKITE_MESSAGE": "fix (#42)",
                        },
                        ("skip-tpu-ci",),
                        (),
                    ),
                ),
            ),
        ):
            baseline: list[tuple[str, Any]] | None = None
            for env, gh_labels, mq_extras in channels:
                with self.subTest(scope=scope, env=env, mq_extras=mq_extras):
                    rec = self._run_bootstrap(
                        env,
                        labels=gh_labels,
                        mq_marker=(mq_extras, "alice" if mq_extras else "", None),
                    )
                    self.assertTrue(
                        rec.annotations[-1][2].startswith(f"CI suite: {scope}")
                    )
                    lanes = _normalize(rec)
                    if baseline is None:
                        baseline = lanes
                    else:
                        self.assertEqual(lanes, baseline)

        # Release-branch merge queue uploads the same steps as `full`.
        mq_rel = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": "gh-readonly-queue/releases/v0.32.0/pr-42-x",
                "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "releases/v0.32.0",
            }
        )
        self.assertEqual(mq_rel.mq_extras_calls, [])
        direct_rel = self._run_bootstrap({"BUILDKITE_BRANCH": "releases/v0.32.0"})
        self.assertEqual(_normalize(mq_rel), _normalize(direct_rel))

    def test_pipeline_yaml_validation(self) -> None:
        bk_bin = shutil.which("bk")
        if bk_bin is None:
            self.skipTest("bk CLI not installed")
        for lane in ("build", "dsv41", "dev", "dynamic"):
            subprocess.run(
                [
                    bk_bin,
                    "pipeline",
                    "validate",
                    "--file",
                    str(_REPO_ROOT / f".buildkite/pipeline_{lane}.yml"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        pr = self._run_bootstrap({"BUILDKITE_PULL_REQUEST": "42", "NIGHTLY": "1"})
        skip_fx = self._run_bootstrap(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "skip-tpu-ci",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            for name, rec in (("pr", pr), ("skip_tpu_ci", skip_fx)):
                steps: list[Any] = []
                for _, content, _ in reversed(rec.uploads):
                    steps.extend((yaml.safe_load(content) or {}).get("steps") or [])
                merged = pathlib.Path(tmp) / f"{name}.yml"
                merged.write_text(
                    yaml.safe_dump({"steps": steps}, sort_keys=False),
                    encoding="utf-8",
                )
                subprocess.run(
                    [bk_bin, "pipeline", "validate", "--file", str(merged)],
                    check=True,
                    capture_output=True,
                    text=True,
                )


if __name__ == "__main__":
    unittest.main()
