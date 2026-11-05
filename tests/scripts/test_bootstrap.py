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

import http.client
import http.server
import importlib
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


class _RunRecorder:
    """In-memory fake for `_agent`, `_git`, and `_github_labels` in bootstrap runs."""

    def __init__(
        self,
        bp: Any,
        files: list[str] | None = None,
        labels: tuple[str, ...] = (),
        mq_marker: tuple[tuple[str, ...], str, str | None] = ((), "", None),
        fail_git_prefixes: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        self.bp = bp
        self.files = ["src/vllm_torchtpu/envs.py"] if files is None else files
        self.labels = labels
        self.mq_marker = mq_marker
        self.fail_git_prefixes = fail_git_prefixes
        self.meta: list[tuple[str, str]] = []
        self.annotations: list[tuple[str, str, str]] = []
        self.uploads: list[tuple[str, str, dict[str, str]]] = []
        self.git_calls: list[tuple[str, ...]] = []
        self.mq_extras_calls: list[str] = []
        self._current_lane: str | None = None
        self._real_upload = bp.upload_with_priority

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
        return self.labels

    def merge_queue_pr_extras(
        self, pr_number: str, *, env: Mapping[str, str] | None = None
    ) -> tuple[tuple[str, ...], str, str | None]:
        self.mq_extras_calls.append(pr_number)
        return self.mq_marker


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
        mq_marker: tuple[tuple[str, ...], str, str | None] = ((), "", None),
        fail_git_prefixes: tuple[tuple[str, ...], ...] = (),
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
        ):
            self.assertEqual(bp.run_bootstrap(env, repo_root=_REPO_ROOT), 0)
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
            r"\(or 'bootstrap\.py upload-stages build,cpu'\)",
        ):
            ctx({"SKIP_TPU_CI": "1"})
        with self.assertRaisesRegex(
            bp.BootstrapError,
            r"ERROR: CI_LANES was removed; set CI_SUITE=<name> "
            r"\(suites: default full perf integration\)",
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
        # markers, non-empty actor for non-empty extras, and comments >= head commit).
        with self.assertRaises(ValueError):
            bp.parse_merge_queue_comments_page({"not": "a-list"})
        gh_bot = {"type": "Bot", "login": "github-actions[bot]"}
        comments = [
            {"user": {"type": "User"}, "body": "<!-- ci-merge-extras: perf-eval -->"},
            {
                "user": {"type": "Bot", "login": "other-app[bot]"},
                "body": (
                    "<!-- ci-merge-extras: precheck-only -->\n"
                    "<!-- ci-merge-actor: mallory -->"
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
                "created_at": "2026-11-05T10:05:00Z",
                "body": (
                    "Queued.\n\n"
                    "<!-- ci-merge-extras: perf-eval, nightly-suite -->\n"
                    "<!-- ci-merge-actor: alice -->"
                ),
            },
            {
                "user": gh_bot,
                "created_at": "2026-11-05T10:06:00Z",
                "body": ("<!-- ci-merge-extras:  -->\n<!-- ci-merge-actor: bob -->"),
            },
        ]
        self.assertEqual(
            bp.parse_merge_queue_comments_page(comments[:4]),
            (False, (), ""),
        )
        self.assertEqual(
            bp.parse_merge_queue_comments_page(comments[:5]),
            (True, ("perf-eval", "nightly-suite"), "alice"),
        )
        self.assertEqual(bp.parse_merge_queue_comments_page(comments), (True, (), ""))
        # Stale marker before a new push is ignored when head_committed_at is newer.
        self.assertEqual(
            bp.parse_merge_queue_comments_page(
                [
                    {
                        "user": gh_bot,
                        "created_at": "2026-11-05T09:55:00Z",
                        "body": (
                            "<!-- ci-merge-extras: precheck-only -->\n"
                            "<!-- ci-merge-actor: alice -->"
                        ),
                    }
                ],
                head_committed_at="2026-11-05T10:00:00Z",
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

        # 1. PR suite selection (ready and oncall-fix are orthogonal).
        for raw_labels, expected_name, expected_stages, expected_nightly in (
            (
                "ready",
                "pr",
                (("build", "cpu"), ("tests", "integration", "verify")),
                False,
            ),
            (
                "ready,oncall-fix",
                "pr",
                (("build", "cpu"), ("tests", "integration", "verify")),
                False,
            ),
            (
                "ready,oncall-fix,skip-tpu-ci",
                "pr+cpu-only",
                (("build", "cpu"),),
                False,
            ),
            (
                "ready,oncall-fix,run-perf-eval",
                "pr+perf-eval",
                (("build", "cpu"), ("tests", "integration", "perf", "verify")),
                False,
            ),
            (
                "ready,oncall-fix,run-full-ci",
                "pr+nightly-suite",
                (("build", "cpu"), ("tests", "integration", "perf", "verify")),
                True,
            ),
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
            (
                "ready,run-full-ci,run-perf-eval,skip-tpu-ci,oncall-fix",
                "skip-tpu-ci, run-perf-eval, run-full-ci",
            ),
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

        # 3. CI_SUITE override beats integration, skip-tpu-ci, release, and PR labels;
        # unknown CI_SUITE (including removed cpu-only) raises.
        self.assertEqual(
            bp.choose_suite(
                ctx(
                    {
                        "CI_SUITE": "perf",
                        "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration",
                        "BUILDKITE_PULL_REQUEST": "10",
                        "BUILDKITE_PULL_REQUEST_LABELS": "skip-tpu-ci",
                    }
                )
            ),
            bp.SUITES["perf"],
        )
        for bad_suite in ("bogus", "cpu-only"):
            with self.assertRaisesRegex(
                bp.BootstrapError,
                rf"ERROR: unknown CI_SUITE '{bad_suite}'; "
                r"suites are: default full perf integration",
            ):
                bp.choose_suite(ctx({"CI_SUITE": bad_suite}))

        # 4. Integration, post-merge skip-tpu-ci, merge-queue, /ci, full, and default.
        self.assertEqual(
            bp.choose_suite(ctx({"CI_PRECHECK": "1", "CI_EXTRAS": "perf-eval"})).name,
            "pr+cpu-only",
        )
        with self.assertRaisesRegex(bp.BootstrapError, "CI_EXTRAS names 'bogus'"):
            bp.choose_suite(ctx({"CI_RUN_TPU": "1", "CI_EXTRAS": "bogus"}))
        for mq_env, mq_extras, expected_name in (
            ({"BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main"}, (), "pr"),
            (
                {"BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main"},
                ("precheck-only",),
                "pr+cpu-only",
            ),
            (
                {"BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main"},
                ("perf-eval",),
                "pr+perf-eval",
            ),
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
                {"BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "releases/foo"},
                ("perf-eval",),
                "pr+perf-eval",
            ),
            (
                {"BUILDKITE_PULL_REQUEST": "10", "CI_PR_TPU_OPT_IN": "1"},
                (),
                "pr+cpu-only",
            ),
            (
                {
                    "BUILDKITE_PULL_REQUEST": "10",
                    "CI_PR_TPU_OPT_IN": "1",
                    "CI_RUN_TPU": "1",
                },
                (),
                "pr",
            ),
            ({"CI_RUN_TPU": "1", "CI_EXTRAS": "perf-eval"}, (), "pr+perf-eval"),
            ({"CI_RUN_TPU": "1", "CI_EXTRAS": "nightly-suite"}, (), "pr+nightly-suite"),
        ):
            with self.subTest(mq_env=mq_env, mq_extras=mq_extras):
                self.assertEqual(
                    bp.choose_suite(replace(ctx(mq_env), mq_extras=mq_extras)).name,
                    expected_name,
                )

        self.assertEqual(
            bp.choose_suite(
                ctx({"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration"})
            ),
            bp.SUITES["integration"],
        )
        self.assertEqual(
            bp.SUITES["integration"].stages,
            ((*bp.SUITES["full"].stages[0], "promote"),),
        )
        pm_skip = replace(
            ctx({"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"}),
            labels=("skip-tpu-ci",),
        )
        self.assertEqual(bp.choose_suite(pm_skip).name, "pr+cpu-only")

        for full_env in (
            {"NIGHTLY": "1"},
            {"BUILDKITE_TAG": "v0.32.0"},
            {"BUILDKITE_TAG": "v0.32.0rc1"},
            {"BUILDKITE_BRANCH": "releases/v0.32.0"},
        ):
            suite = bp.choose_suite(ctx(full_env))
            self.assertEqual(suite, bp.SUITES["full"])
            self.assertTrue(suite.nightly)

        for default_env in (
            {"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"},
            {"BUILDKITE_BRANCH": "release/v1"},
            {
                "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-torchtpu-nightly",
                "TORCHTPU_NIGHTLY": "1",
            },
            {"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci"},
        ):
            suite = bp.choose_suite(ctx(default_env))
            self.assertEqual(suite, bp.SUITES["default"])
            self.assertFalse(suite.nightly)

        # 5. Every suite in SUITES maps to existing .buildkite/pipeline_<lane>.yml.
        self.assertEqual(
            set(bp.SUITES),
            {"default", "full", "perf", "integration"},
        )
        for suite in bp.SUITES.values():
            self.assertTrue(suite.about)
            self.assertEqual(len(suite.stages), 1)
            for stage in suite.stages:
                for lane in stage:
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
                    elif mode == "page_full" and self.path.endswith("page=1"):
                        items = [{"user": {"type": "User"}, "body": "x"}] * 100
                        self.wfile.write(json.dumps(items).encode("utf-8"))
                    else:
                        created = (
                            "2026-11-05T09:55:00Z"
                            if mode == "stale_marker"
                            else "2026-11-05T10:05:00Z"
                        )
                        extra = (
                            "precheck-only" if mode == "stale_marker" else "perf-eval"
                        )
                        payload = [
                            {
                                "user": {
                                    "type": "Bot",
                                    "login": "github-actions[bot]",
                                },
                                "created_at": created,
                                "body": (
                                    f"<!-- ci-merge-extras: {extra} -->\n"
                                    "<!-- ci-merge-actor: alice -->"
                                ),
                            }
                        ]
                        self.wfile.write(json.dumps(payload).encode("utf-8"))
                elif "commits/" in self.path:
                    self.wfile.write(
                        b'{"commit": {"committer": {"date": "2026-11-05T10:00:00Z"}}}'
                    )
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
                    "build,cpu",
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

    def test_orchestration_pr_staging_and_labels(self) -> None:
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

        # PR default: priority 6, stage 0 uploads upload_stages + cpu + build;
        # stage 1 uploads verify + integration + tests with concurrency group.
        pr = self._run_bootstrap(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com",
                "CI_FLEET": "kube",
            }
        )
        self.assertEqual(
            pr.meta,
            [
                ("JOB_PRIORITY", "6"),
                ("WORKLOAD_PRIORITY", "pre-merge"),
                ("VLLM_COMMIT_HASH", self.pinned),
                ("upload-stages:upload_stage_0:cpu", "1"),
                ("upload-stages:upload_stage_0:build", "1"),
            ],
        )
        self.assertEqual(
            pr.annotations,
            [("info", "ci-suite", f"CI suite: pr: {bp._PR_SUITES['pr'].about}")],
        )
        self.assertEqual(
            [u[0] for u in pr.uploads],
            ["upload_stages", "cpu", "build"],
        )
        self.assertIn(
            ":pipeline: Upload tests, integration, verify",
            pr.uploads[0][1],
        )
        self.assertEqual(
            json.loads(pr.uploads[0][1])["steps"][1]["env"]["CI_FLEET"], "kube"
        )
        self.assertEqual(pr.uploads[2][2]["OMNI_IMAGE_SKIP"], "")
        self.assertNotIn("tpu-pr/alice@example.com", pr.uploads[1][1])
        pr_stage1 = self._run_next_stage(pr)
        self.assertEqual(
            [u[0] for u in pr_stage1.uploads],
            ["verify", "integration", "tests"],
        )
        self.assertEqual(
            pr_stage1.meta,
            [
                ("upload-stages:upload_stage_1:verify", "1"),
                ("upload-stages:upload_stage_1:integration", "1"),
                ("upload-stages:upload_stage_1:tests", "1"),
            ],
        )
        self.assertIn("tpu-pr/alice@example.com", pr_stage1.uploads[2][1])

        # Automatic retry after agent loss skips lanes already recorded in meta-data.
        retry_rec = _RunRecorder(bp)
        retry_rec.meta = [
            ("upload-stages:upload_stage_1:verify", "1"),
            ("upload-stages:upload_stage_1:integration", "1"),
        ]
        with (
            mock.patch.object(bp, "_agent", side_effect=retry_rec.agent),
            mock.patch.object(
                bp, "upload_with_priority", side_effect=retry_rec.upload_with_priority
            ),
        ):
            self.assertEqual(
                bp.upload_stages(
                    (("tests", "integration", "verify"),),
                    {"JOB_PRIORITY": "6", "BUILDKITE_STEP_KEY": "upload_stage_1"},
                    repo_root=_REPO_ROOT,
                ),
                0,
            )
        self.assertEqual([u[0] for u in retry_rec.uploads], ["tests"])
        self.assertIn(("upload-stages:upload_stage_1:tests", "1"), retry_rec.meta)

        # 3-stage test-only suite chains stages 0 -> 1 -> 2 with upload_stage_1, _2.
        three_stage_suite = bp.Suite(
            "three-stage",
            "3-stage test suite",
            stages=(("build", "cpu"), ("tests", "integration"), ("verify",)),
        )
        with mock.patch.dict(bp.SUITES, {"three-stage": three_stage_suite}):
            chain_s0 = self._run_bootstrap({"CI_SUITE": "three-stage"})
            self.assertEqual(
                [u[0] for u in chain_s0.uploads],
                ["upload_stages", "cpu", "build"],
            )
            chain_s1 = self._run_next_stage(chain_s0, expected_key="upload_stage_1")
            self.assertEqual(
                [u[0] for u in chain_s1.uploads],
                ["upload_stages", "integration", "tests"],
            )
            chain_s2 = self._run_next_stage(chain_s1, expected_key="upload_stage_2")
            self.assertEqual([u[0] for u in chain_s2.uploads], ["verify"])

        # PR run-perf-eval: pr+perf-eval suite, NIGHTLY unset.
        perf_eval_pr = self._run_bootstrap(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "ready,run-perf-eval",
            }
        )
        self.assertTrue(
            perf_eval_pr.annotations[0][2].startswith("CI suite: pr+perf-eval: ")
        )
        perf_eval_stage1 = self._run_next_stage(perf_eval_pr)
        self.assertEqual(
            [u[0] for u in perf_eval_stage1.uploads],
            ["verify", "perf", "integration", "tests"],
        )
        self.assertNotIn("NIGHTLY", perf_eval_stage1.uploads[0][2])

        # PR run-full-ci + oncall-fix: priority 12, oncall-fix warning, 2-stage
        # pr+nightly-suite with NIGHTLY=1 substituted in both stages.
        oncall_full = self._run_bootstrap(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "run-full-ci,oncall-fix",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Oncall@Example.com",
            }
        )
        self.assertIn(("JOB_PRIORITY", "12"), oncall_full.meta)
        self.assertIn(("WORKLOAD_PRIORITY", "oncall-fix"), oncall_full.meta)
        self.assertEqual(oncall_full.annotations[0][0], "warning")
        self.assertEqual(oncall_full.annotations[0][1], "oncall-fix")
        self.assertTrue(
            oncall_full.annotations[1][2].startswith(
                "CI suite: pr+nightly-suite [NIGHTLY=1]: "
            )
        )
        oncall_stage1 = self._run_next_stage(oncall_full)
        self.assertEqual(
            [u[0] for u in oncall_stage1.uploads],
            ["verify", "perf", "integration", "tests"],
        )
        self.assertEqual(json.loads(oncall_stage1.uploads[0][1])["priority"], 12)
        self.assertIn("tpu-pr/oncall@example.com", oncall_stage1.uploads[3][1])
        for _, content, _ in (*oncall_full.uploads[1:], *oncall_stage1.uploads):
            self.assertNotIn('build.env("NIGHTLY")', content)

    def test_orchestration_single_stage_and_skip_flows(self) -> None:
        # PR skip-tpu-ci & post-merge main webhook upload ONLY ["cpu", "build"].
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
            self.assertTrue(fx.annotations[0][2].startswith("CI suite: pr+cpu-only: "))
            self.assertEqual([u[0] for u in fx.uploads], ["cpu", "build"])
            self.assertTrue(bool(fx.uploads[1][2]["OMNI_IMAGE_SKIP"]))
            self.assertLessEqual(len(fx.uploads[1][2]["OMNI_IMAGE_SKIP"]), 70)

        # main post-merge (even when merged PR had run-full-ci / run-perf-eval)
        # runs default suite in one stage without NIGHTLY.
        main_pm = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": "main",
                "BUILDKITE_SOURCE": "webhook",
                "BUILDKITE_MESSAGE": "feat: land change (#100)",
            },
            labels=("run-full-ci", "run-perf-eval"),
        )
        self.assertIn(("JOB_PRIORITY", "11"), main_pm.meta)
        self.assertIn(("WORKLOAD_PRIORITY", "post-merge"), main_pm.meta)
        self.assertTrue(main_pm.annotations[0][2].startswith("CI suite: default: "))
        self.assertEqual(
            [u[0] for u in main_pm.uploads],
            ["verify", "integration", "tests", "cpu", "build"],
        )
        self.assertNotIn("NIGHTLY", main_pm.uploads[0][2])

        # CI_SUITE takes precedence when CI_LANES is also set on a schedule.
        perf_sched = self._run_bootstrap(
            {"CI_SUITE": "perf", "CI_LANES": "perf", "BUILDKITE_BRANCH": "main"}
        )
        self.assertEqual([u[0] for u in perf_sched.uploads], ["perf", "build"])
        self.assertTrue(bool(perf_sched.uploads[1][2]["OMNI_IMAGE_SKIP"]))

        # Scheduled Nightly, release builds (tag or releases/vX.Y.Z branch), and
        # manual CI_SUITE=full run 1-stage full suite with NIGHTLY=1 substituted.
        nightly = self._run_bootstrap(
            {
                "NIGHTLY": "1",
                "BUILDKITE_SOURCE": "schedule",
                "BUILDKITE_BRANCH": "main",
            }
        )
        self.assertEqual(
            nightly.meta[:2], [("JOB_PRIORITY", "1"), ("VLLM_COMMIT_HASH", self.pinned)]
        )
        tag_fx = self._run_bootstrap(
            {"BUILDKITE_TAG": "v0.32.0", "BUILDKITE_BRANCH": "v0.32.0"}
        )
        rel_branch_fx = self._run_bootstrap({"BUILDKITE_BRANCH": "releases/v0.32.0"})
        manual_full = self._run_bootstrap(
            {"CI_SUITE": "full", "BUILDKITE_BRANCH": "exp/manual"}
        )
        for fx in (nightly, tag_fx, rel_branch_fx, manual_full):
            self.assertTrue(
                fx.annotations[0][2].startswith("CI suite: full [NIGHTLY=1]: ")
            )
            self.assertEqual(
                [u[0] for u in fx.uploads],
                ["verify", "perf", "integration", "tests", "cpu", "build"],
            )
            for _, content, _ in fx.uploads:
                self.assertNotIn('build.env("NIGHTLY")', content)

        # torchtpu-nightly schedule runs default suite at priority 1 (one stage).
        torchtpu_nightly = self._run_bootstrap(
            {
                "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-torchtpu-nightly",
                "TORCHTPU_NIGHTLY": "1",
                "BUILDKITE_SOURCE": "schedule",
                "BUILDKITE_BRANCH": "main",
            }
        )
        self.assertEqual(
            torchtpu_nightly.meta[:2],
            [("JOB_PRIORITY", "1"), ("VLLM_COMMIT_HASH", self.pinned)],
        )
        self.assertEqual(
            [u[0] for u in torchtpu_nightly.uploads],
            ["verify", "integration", "tests", "cpu", "build"],
        )

        # Integration pipeline resolves upstream vLLM HEAD and uploads promote lane.
        integ = self._run_bootstrap(
            {
                "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration",
                "BUILDKITE_SOURCE": "schedule",
            }
        )
        self.assertIn(("VLLM_COMMIT_HASH", "upstream999"), integ.meta)
        self.assertEqual(
            [u[0] for u in integ.uploads],
            ["promote", "verify", "perf", "integration", "tests", "cpu", "build"],
        )

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

    def test_orchestration_merge_queue_and_ci_commands(self) -> None:
        # /ci run with CI_EXTRAS=perf-eval runs pr+perf-eval 2-stage with author group.
        ci_run = self._run_bootstrap(
            {
                "CI_RUN_TPU": "1",
                "CI_EXTRAS": "perf-eval",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com",
                "BUILDKITE_MESSAGE": "feat: x [skip ci]",
            }
        )
        self.assertIn(("JOB_PRIORITY", "6"), ci_run.meta)
        self.assertTrue(ci_run.annotations[0][2].startswith("CI suite: pr+perf-eval: "))
        ci_run_stage1 = self._run_next_stage(ci_run)
        self.assertEqual(
            [u[0] for u in ci_run_stage1.uploads],
            ["verify", "perf", "integration", "tests"],
        )
        self.assertIn("tpu-pr/alice@example.com", ci_run_stage1.uploads[3][1])

        # CI_PRECHECK=1 runs pr+cpu-only even on [skip ci] / docs-only.
        pre = self._run_bootstrap(
            {
                "CI_PRECHECK": "1",
                "CI_RUN_TPU": "1",
                "BUILDKITE_MESSAGE": "docs [skip ci]",
            },
            files=["README.md"],
        )
        self.assertIn(("JOB_PRIORITY", "6"), pre.meta)
        self.assertTrue(pre.annotations[0][2].startswith("CI suite: pr+cpu-only: "))
        self.assertEqual([u[0] for u in pre.uploads], ["cpu", "build"])
        self.assertTrue(bool(pre.uploads[1][2]["OMNI_IMAGE_SKIP"]))

        # Merge queue on main: priority 11 (post-merge), ignores PR labels and
        # CI_PR_TPU_OPT_IN, diffs against BUILDKITE_MERGE_QUEUE_BASE_COMMIT, and
        # runs 2-stage pr suite uncapped.
        mq_main = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": "gh-readonly-queue/main/pr-42-deadbeef",
                "BUILDKITE_COMMIT": "deadbeef",
                "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main",
                "BUILDKITE_MERGE_QUEUE_BASE_COMMIT": "base111",
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "skip-tpu-ci,oncall-fix",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com",
                "BUILDKITE_MESSAGE": "feat: x [skip ci]",
                "CI_PR_TPU_OPT_IN": "1",
            }
        )
        self.assertEqual(
            mq_main.git_calls[:3],
            [
                ("fetch", "origin", "base111", "--depth=20", "--quiet"),
                ("fetch", "origin", "deadbeef", "--depth=20", "--quiet"),
                ("diff", "--name-only", "base111...deadbeef"),
            ],
        )
        self.assertIn(("JOB_PRIORITY", "11"), mq_main.meta)
        self.assertIn(("WORKLOAD_PRIORITY", "post-merge"), mq_main.meta)
        self.assertTrue(mq_main.annotations[0][2].startswith("CI suite: pr: "))
        mq_main_stage1 = self._run_next_stage(mq_main)
        self.assertEqual(
            [u[0] for u in mq_main_stage1.uploads],
            ["verify", "integration", "tests"],
        )
        self.assertNotIn("tpu-pr/", mq_main_stage1.uploads[2][1])

        # Merge queue on releases/vX.Y.Z always runs pr+nightly-suite [NIGHTLY=1]
        # and does not query GitHub comments for /ci merge extras.
        mq_rel = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": "gh-readonly-queue/releases/v0.8.1/pr-42-deadbeef",
                "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "releases/v0.8.1",
            },
            mq_marker=(("precheck-only",), "alice", None),
        )
        self.assertEqual(mq_rel.mq_extras_calls, [])
        self.assertTrue(
            mq_rel.annotations[0][2].startswith(
                "CI suite: pr+nightly-suite [NIGHTLY=1]: "
            )
        )
        mq_rel_stage1 = self._run_next_stage(mq_rel)
        self.assertIn("perf", [u[0] for u in mq_rel_stage1.uploads])

        # Merge queue without BUILDKITE_MERGE_QUEUE_BASE_COMMIT falls back to
        # fetching the base branch ref, and with precheck-only marker runs pr+cpu-only.
        mq_pre = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": "gh-readonly-queue/main/pr-42-deadbeef",
                "BUILDKITE_COMMIT": "deadbeef",
                "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main",
            },
            mq_marker=(("precheck-only",), "alice", None),
        )
        self.assertEqual(
            mq_pre.git_calls[:3],
            [
                (
                    "fetch",
                    "origin",
                    "+refs/heads/main:refs/remotes/origin/main",
                    "--depth=20",
                    "--quiet",
                ),
                ("fetch", "origin", "deadbeef", "--depth=20", "--quiet"),
                ("diff", "--name-only", "origin/main...deadbeef"),
            ],
        )
        self.assertTrue(mq_pre.annotations[0][2].startswith("CI suite: pr+cpu-only: "))
        self.assertEqual([u[0] for u in mq_pre.uploads], ["cpu", "build"])

        # Merge queue fail-open: if base fetch or diff fails even when the queue
        # head commit is docs-only, never fall back to diff-tree on the head commit
        # (which could skip a code PR grouped behind a docs PR); run the full gate.
        for fail_prefixes in (
            (("fetch", "origin", "base111"),),
            (("diff",),),
        ):
            with self.subTest(fail_prefixes=fail_prefixes):
                mq_fail_open = self._run_bootstrap(
                    {
                        "BUILDKITE_BRANCH": "gh-readonly-queue/main/pr-42-deadbeef",
                        "BUILDKITE_COMMIT": "deadbeef",
                        "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main",
                        "BUILDKITE_MERGE_QUEUE_BASE_COMMIT": "base111",
                    },
                    files=["README.md"],
                    fail_git_prefixes=fail_prefixes,
                )
                self.assertFalse(
                    any(c[:1] == ("diff-tree",) for c in mq_fail_open.git_calls)
                )
                self.assertEqual(
                    [u[0] for u in mq_fail_open.uploads],
                    ["upload_stages", "cpu", "build"],
                )

        # Merge queue warning annotation on comment API failure, then default gate.
        mq_warn = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": "gh-readonly-queue/main/pr-42-deadbeef",
                "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main",
            },
            mq_marker=((), "", "HTTP 500"),
        )
        self.assertEqual(
            mq_warn.annotations[0],
            (
                "warning",
                "ci-merge-extras",
                "could not read /ci merge extras for PR #42: HTTP 500; "
                "ran the default gate",
            ),
        )
        self.assertTrue(mq_warn.annotations[1][2].startswith("CI suite: pr: "))

        # Docs-only merge-queue build posts docs-only info annotation and skips upload.
        mq_docs = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": "gh-readonly-queue/main/pr-42-deadbeef",
                "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main",
            },
            files=["README.md"],
        )
        self.assertEqual(
            mq_docs.annotations,
            [
                (
                    "info",
                    "docs-only",
                    "Docs-only change; skipped CPU and TPU checks in the merge queue.",
                )
            ],
        )
        self.assertEqual(mq_docs.uploads, [])

    def test_scope_equivalence_across_channels(self) -> None:
        """All channels for a given scope choose the same Suite and upload the same
        steps; only priority, concurrency_group, and annotation context differ."""
        bp = self.bp

        def _normalize_lane_uploads(
            rec: _RunRecorder, stage1: _RunRecorder | None
        ) -> list[tuple[str, Any]]:
            all_uploads = [
                *(u for u in rec.uploads if u[0] != "upload_stages"),
                *(stage1.uploads if stage1 is not None else ()),
            ]
            normalized: list[tuple[str, Any]] = []
            for lane, content, _ in all_uploads:
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
                                item.pop("concurrency", None)
                                item.pop("concurrency_group", None)
                                item.pop("concurrency_method", None)
                normalized.append((lane, doc))
            return normalized

        author = {"BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com"}
        mq_base = {
            "BUILDKITE_BRANCH": "gh-readonly-queue/main/pr-42-deadbeef",
            "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main",
            **author,
        }
        for scope, expected_suite, expected_nightly, channels in (
            (
                "default",
                "pr",
                False,
                (
                    ({"BUILDKITE_PULL_REQUEST": "42", **author}, (), ()),
                    (mq_base, (), ()),
                    ({"CI_RUN_TPU": "1", **author}, (), ()),
                ),
            ),
            (
                "perf-eval",
                "pr+perf-eval",
                False,
                (
                    (
                        {
                            "BUILDKITE_PULL_REQUEST": "42",
                            "BUILDKITE_PULL_REQUEST_LABELS": "run-perf-eval",
                            **author,
                        },
                        (),
                        (),
                    ),
                    (mq_base, (), ("perf-eval",)),
                    (
                        {"CI_RUN_TPU": "1", "CI_EXTRAS": "perf-eval", **author},
                        (),
                        (),
                    ),
                ),
            ),
            (
                "nightly-suite",
                "pr+nightly-suite",
                True,
                (
                    (
                        {
                            "BUILDKITE_PULL_REQUEST": "42",
                            "BUILDKITE_PULL_REQUEST_LABELS": "run-full-ci",
                            **author,
                        },
                        (),
                        (),
                    ),
                    (mq_base, (), ("nightly-suite",)),
                    (
                        {
                            "CI_RUN_TPU": "1",
                            "CI_EXTRAS": "nightly-suite",
                            **author,
                        },
                        (),
                        (),
                    ),
                ),
            ),
            (
                "cpu-only",
                "pr+cpu-only",
                False,
                (
                    (
                        {
                            "BUILDKITE_PULL_REQUEST": "42",
                            "BUILDKITE_PULL_REQUEST_LABELS": "skip-tpu-ci",
                            **author,
                        },
                        (),
                        (),
                    ),
                    (
                        {
                            "BUILDKITE_PULL_REQUEST": "42",
                            "CI_PR_TPU_OPT_IN": "1",
                            **author,
                        },
                        (),
                        (),
                    ),
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
            baseline_lanes: list[tuple[str, Any]] | None = None
            for env, gh_labels, mq_extras in channels:
                with self.subTest(scope=scope, env=env, mq_extras=mq_extras):
                    ctx = replace(
                        bp.BuildContext.from_env(env),
                        labels=gh_labels or bp.BuildContext.from_env(env).labels,
                        mq_extras=mq_extras,
                    )
                    suite = bp.choose_suite(ctx)
                    self.assertEqual(suite.name, expected_suite)
                    self.assertEqual(suite.nightly, expected_nightly)
                    s0 = self._run_bootstrap(
                        env,
                        labels=gh_labels,
                        mq_marker=(mq_extras, "alice" if mq_extras else "", None),
                    )
                    s1 = self._run_next_stage(s0) if len(suite.stages) > 1 else None
                    lanes = _normalize_lane_uploads(s0, s1)
                    if baseline_lanes is None:
                        baseline_lanes = lanes
                    else:
                        self.assertEqual(lanes, baseline_lanes)

        # Release-branch merge queue uploads the exact same 6 lanes and steps
        # (with NIGHTLY=1) as the `full` suite on a direct release branch.
        mq_rel_s0 = self._run_bootstrap(
            {
                "BUILDKITE_BRANCH": (
                    "gh-readonly-queue/releases/v0.32.0/pr-42-deadbeef"
                ),
                "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "releases/v0.32.0",
            }
        )
        mq_rel_s1 = self._run_next_stage(mq_rel_s0)
        direct_rel = self._run_bootstrap({"BUILDKITE_BRANCH": "releases/v0.32.0"})
        self.assertEqual(
            dict(_normalize_lane_uploads(mq_rel_s0, mq_rel_s1)),
            dict(_normalize_lane_uploads(direct_rel, None)),
        )

    def test_pipeline_yaml_validation_and_nightly_report_keys(self) -> None:
        bp = self.bp
        pr = self._run_bootstrap(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com",
            }
        )
        pr_stage1 = self._run_next_stage(pr)
        main_pm = self._run_bootstrap(
            {"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"}
        )
        skip_fx = self._run_bootstrap(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "skip-tpu-ci",
            }
        )
        perf_sched = self._run_bootstrap({"CI_SUITE": "perf"})
        nightly = self._run_bootstrap({"NIGHTLY": "1", "BUILDKITE_SOURCE": "schedule"})
        integ = self._run_bootstrap(
            {"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration"}
        )

        bk_bin = shutil.which("bk")
        if bk_bin is not None:
            for lane in bp.SUITES["integration"].stages[0]:
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

            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = pathlib.Path(tmp)
                for name, uploads in (
                    ("pr_stage0", list(reversed(pr.uploads))),
                    (
                        "pr_both_stages",
                        [*reversed(pr.uploads), *reversed(pr_stage1.uploads)],
                    ),
                    ("main_post_merge", list(reversed(main_pm.uploads))),
                    ("skip_tpu_ci", list(reversed(skip_fx.uploads))),
                    ("ci_suite_perf", list(reversed(perf_sched.uploads))),
                    ("nightly", list(reversed(nightly.uploads))),
                    ("integration", list(reversed(integ.uploads))),
                ):
                    steps: list[Any] = []
                    for _, content, _ in uploads:
                        doc = yaml.safe_load(content) or {}
                        steps.extend(doc.get("steps") or [])
                    merged_file = tmp_dir / f"{name}.yml"
                    merged_file.write_text(
                        yaml.safe_dump({"steps": steps}, sort_keys=False),
                        encoding="utf-8",
                    )
                    subprocess.run(
                        [bk_bin, "pipeline", "validate", "--file", str(merged_file)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

        verify_doc = yaml.safe_load(
            (_REPO_ROOT / ".buildkite/pipeline_verify.yml").read_text(encoding="utf-8")
        )
        report_step = next(
            s
            for s in verify_doc["steps"]
            if isinstance(s, dict) and s.get("key") == "nightly_report"
        )
        report_deps = set(report_step["depends_on"]) - {"v6e-group", "v7x-group"}
        scanned = subprocess.run(
            [
                "bash",
                "-c",
                "grep -hoE '^[[:space:]]*key:[[:space:]]*\"?[A-Za-z0-9_-]+' "
                ".buildkite/pipeline_cpu.yml .buildkite/pipeline_tests.yml "
                "| sed -E 's/.*key:[[:space:]]*\"?//' "
                "| grep -v '_pr$'",
            ],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(set(scanned.stdout.split()), report_deps)


if __name__ == "__main__":
    unittest.main()
