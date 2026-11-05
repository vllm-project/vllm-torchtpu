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
from typing import Any
from unittest import mock

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


class TestBootstrapUnits(unittest.TestCase):
    """Unit tests for policy tables, chooser functions, helpers, and orchestration."""

    bp: Any

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        had_bp = "bootstrap" in sys.modules
        had_cap = "cap_pr_tpu_concurrency" in sys.modules
        with mock.patch.object(sys, "path", [str(_SCRIPTS_DIR), *sys.path]):
            cls.bp = importlib.import_module("bootstrap")

        def _cleanup_modules() -> None:
            if not had_bp:
                sys.modules.pop("bootstrap", None)
            if not had_cap:
                sys.modules.pop("cap_pr_tpu_concurrency", None)

        cls.addClassCleanup(_cleanup_modules)

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
            build_ctx = ctx(env)
            kind = bp.build_kind(build_ctx)
            self.assertEqual(kind, expected_kind)
            self.assertEqual(bp.PRIORITIES[kind], expected_prio)

        # BuildContext named properties (labels, extras, is_release,
        # runs_nightly_steps).
        pr_ctx = ctx(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_BRANCH": "releases/v0.32.0",
                "BUILDKITE_PULL_REQUEST_LABELS": (
                    "ready,run-full-ci,run-perf-eval,skip-tpu-ci,oncall-fix"
                ),
            }
        )
        self.assertTrue(pr_ctx.is_pr)
        self.assertTrue(pr_ctx.is_oncall_fix)
        self.assertFalse(pr_ctx.is_release)
        self.assertTrue(pr_ctx.skips_tpu)
        self.assertEqual(pr_ctx.extras, frozenset({"nightly-suite", "perf-eval"}))
        self.assertTrue(pr_ctx.runs_nightly_steps)

        for rel_env in (
            {"BUILDKITE_TAG": "v0.32.0"},
            {"BUILDKITE_TAG": "v0.32.0rc1"},
            {"BUILDKITE_TAG": "0.32.0"},
            {"BUILDKITE_BRANCH": "releases/v0.32.0"},
        ):
            rel_ctx = ctx(rel_env)
            self.assertTrue(rel_ctx.is_release)
            self.assertTrue(rel_ctx.runs_nightly_steps)

        for non_rel_env in (
            {"BUILDKITE_BRANCH": "release/v1"},
            {"BUILDKITE_BRANCH": "releases/v01.0.0"},
            {"BUILDKITE_BRANCH": "releases/foo"},
            {"BUILDKITE_TAG": "nightly-20261105"},
        ):
            non_rel_ctx = ctx(non_rel_env)
            self.assertFalse(non_rel_ctx.is_release)
            self.assertFalse(non_rel_ctx.runs_nightly_steps)

        post_merge_ctx = ctx(
            {"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"},
            merged_labels=("skip-tpu-ci", "run-full-ci", "run-perf-eval"),
        )
        self.assertFalse(post_merge_ctx.is_pr)
        self.assertFalse(post_merge_ctx.is_oncall_fix)
        self.assertFalse(post_merge_ctx.is_release)
        self.assertTrue(post_merge_ctx.skips_tpu)
        self.assertEqual(post_merge_ctx.extras, frozenset())
        self.assertFalse(post_merge_ctx.runs_nightly_steps)

        post_merge_full_label_ctx = ctx(
            {"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"},
            merged_labels=("run-full-ci", "run-perf-eval"),
        )
        self.assertEqual(post_merge_full_label_ctx.extras, frozenset())
        self.assertFalse(post_merge_full_label_ctx.runs_nightly_steps)
        self.assertEqual(
            bp.choose_suite(post_merge_full_label_ctx), bp.SUITES["default"]
        )

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

        rendered = bp.prepare_upload_yaml(
            'if: \'build.env("NIGHTLY") == "1"\'\n', job_priority=12, nightly=True
        )
        self.assertEqual(rendered, 'priority: 12\nif: \'"1" == "1"\'\n')

    def test_choose_suite_pr_suite_and_suites_table(self) -> None:
        bp = self.bp
        ctx = bp.BuildContext.from_env

        # 1. pr_suite with each label/extras combination (including oncall-fix
        # and skip-tpu-ci).
        for raw_labels, expected_name, expected_about, expected_stages in (
            (
                "ready",
                "pr",
                "PR: TPU waits for the CPU checks",
                (("build", "cpu"), ("tests", "integration", "verify")),
            ),
            (
                "ready,oncall-fix",
                "pr",
                "PR: TPU waits for the CPU checks",
                (("build", "cpu"), ("tests", "integration", "verify")),
            ),
            (
                "ready,run-perf-eval",
                "pr+perf-eval",
                "PR: TPU waits for the CPU checks",
                (("build", "cpu"), ("tests", "integration", "perf", "verify")),
            ),
            (
                "ready,run-full-ci",
                "pr+nightly-suite",
                "PR: TPU waits for the CPU checks",
                (("build", "cpu"), ("tests", "integration", "perf", "verify")),
            ),
            (
                "ready,run-perf-eval,run-full-ci",
                "pr+nightly-suite+perf-eval",
                "PR: TPU waits for the CPU checks",
                (("build", "cpu"), ("tests", "integration", "perf", "verify")),
            ),
            (
                "ready,skip-tpu-ci",
                "pr+cpu-only",
                "CPU checks only (skip-tpu-ci)",
                (("build", "cpu"),),
            ),
            (
                "ready,skip-tpu-ci,run-perf-eval",
                "pr+cpu-only",
                "CPU checks only (skip-tpu-ci)",
                (("build", "cpu"),),
            ),
            (
                "ready,skip-tpu-ci,run-full-ci,run-perf-eval",
                "pr+cpu-only+nightly-suite",
                "CPU checks only (skip-tpu-ci)",
                (("build", "cpu"),),
            ),
        ):
            suite = bp.pr_suite(
                ctx(
                    {
                        "BUILDKITE_PULL_REQUEST": "10",
                        "BUILDKITE_PULL_REQUEST_LABELS": raw_labels,
                    }
                )
            )
            self.assertEqual(
                suite,
                bp.Suite(expected_name, expected_about, stages=expected_stages),
            )

        # 2. One test per branch of choose_suite (first match wins).
        # Branch 1: CI_SUITE override beats integration, skip-tpu-ci, release,
        # and PR labels; unknown CI_SUITE (including removed cpu-only) raises.
        self.assertEqual(
            bp.choose_suite(
                ctx(
                    {
                        "CI_SUITE": "perf",
                        "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration",
                        "BUILDKITE_PULL_REQUEST": "10",
                        "BUILDKITE_PULL_REQUEST_LABELS": "skip-tpu-ci,run-full-ci",
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

        # Branch 2: vllm-torchtpu-integration pipeline extends full with promote.
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

        # Branch 3: PRs (including oncall-fix and skip-tpu-ci) and post-merge
        # skip-tpu-ci webhook builds all route to pr_suite(ctx).
        for raw_labels, expected_name in (
            ("ready", "pr"),
            ("ready,oncall-fix", "pr"),
            ("ready,oncall-fix,run-perf-eval", "pr+perf-eval"),
            ("ready,oncall-fix,run-full-ci", "pr+nightly-suite"),
            ("ready,run-perf-eval,run-full-ci", "pr+nightly-suite+perf-eval"),
            ("ready,skip-tpu-ci", "pr+cpu-only"),
            ("run-full-ci,run-perf-eval,skip-tpu-ci", "pr+cpu-only+nightly-suite"),
        ):
            pr_env = {
                "BUILDKITE_PULL_REQUEST": "10",
                "BUILDKITE_PULL_REQUEST_LABELS": raw_labels,
            }
            self.assertEqual(bp.choose_suite(ctx(pr_env)), bp.pr_suite(ctx(pr_env)))
            self.assertEqual(bp.choose_suite(ctx(pr_env)).name, expected_name)

        self.assertEqual(
            bp.choose_suite(
                ctx(
                    {"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"},
                    merged_labels=("skip-tpu-ci", "run-full-ci"),
                )
            ),
            bp.Suite(
                "pr+cpu-only",
                "CPU checks only (skip-tpu-ci)",
                stages=(("build", "cpu"),),
            ),
        )

        # Branch 4a: NIGHTLY=1 or release builds (tag or releases/vX.Y.Z branch)
        # -> SUITES["full"].
        for full_env in (
            {"NIGHTLY": "1"},
            {"BUILDKITE_TAG": "v0.32.0"},
            {"BUILDKITE_TAG": "v0.32.0rc1"},
            {"BUILDKITE_BRANCH": "releases/v0.32.0"},
        ):
            self.assertEqual(bp.choose_suite(ctx(full_env)), bp.SUITES["full"])

        # Branch 4b: default for main, torchtpu-nightly, and other non-release builds.
        for default_env in (
            {"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"},
            {"BUILDKITE_BRANCH": "release/v1"},
            {
                "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-torchtpu-nightly",
                "TORCHTPU_NIGHTLY": "1",
            },
            {"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci"},
        ):
            self.assertEqual(bp.choose_suite(ctx(default_env)), bp.SUITES["default"])

        # 3. Every suite in SUITES maps to existing .buildkite/pipeline_<lane>.yml.
        self.assertEqual(
            set(bp.SUITES),
            {"default", "full", "perf", "integration"},
        )
        for suite in bp.SUITES.values():
            self.assertEqual(len(suite.stages), 1)
            for stage in suite.stages:
                for lane in stage:
                    self.assertTrue(
                        (_REPO_ROOT / f".buildkite/pipeline_{lane}.yml").is_file()
                    )

    def test_github_labels_resilience(self) -> None:
        bp = self.bp
        seen_auth: list[str | None] = []

        class AuthHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                seen_auth.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
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
                with (
                    mock.patch.object(bp, "_GITHUB_API_URL", api_url),
                    mock.patch.object(
                        bp, "_CREDENTIAL_HELPER", tmp_path / "missing-helper"
                    ),
                ):
                    self.assertEqual(bp._github_labels("42", env={}), ("skip-tpu-ci",))
            finally:
                server.shutdown()
                server.server_close()

            self.assertEqual(seen_auth, ["token ghs_tok", None])

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
        self.assertEqual(
            good_cli.stdout.strip(), bp.read_pinned_vllm_commit(_REPO_ROOT)
        )

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
                (
                    ["upload-stages"],
                    "Usage: bootstrap.py "
                    "[dev | vllm-pin | upload-stages <stage> [<stage> ...]]",
                ),
                (
                    ["upload-stages", ""],
                    "Usage: bootstrap.py "
                    "[dev | vllm-pin | upload-stages <stage> [<stage> ...]]",
                ),
                (
                    ["upload-stages", "bogus"],
                    "ERROR: lane file '.buildkite/pipeline_bogus.yml' not found",
                ),
                (
                    ["bogus-cmd"],
                    "Usage: bootstrap.py "
                    "[dev | vllm-pin | upload-stages <stage> [<stage> ...]]",
                ),
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

    def test_run_bootstrap_orchestration(self) -> None:
        import yaml  # pyrefly: ignore[missing-import]

        bp = self.bp
        pinned = bp.parse_pinned_vllm_commit(
            (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        tags = (
            ("upload_stages", "upload-stages "),
            ("promote", "promote_lkg"),
            ("verify", "Support matrices and verified commit"),
            ("perf", "Perf and Eval on TPU"),
            ("integration", "integration_tpu"),
            ("tests", "Unit Tests on TPU"),
            ("cpu", "CPU Checks"),
            ("build", "build_docker_group"),
        )

        def expand_env(text: str, env: Mapping[str, str]) -> str:
            def _sub(match: re.Match[str]) -> str:
                key, default = match.group(1), match.group(2)
                val = env.get(key, "")
                return val if val else (default or "")

            return re.sub(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}", _sub, text)

        class RunRecorder:
            def __init__(
                self,
                files: list[str] | None = None,
                labels: tuple[str, ...] = (),
            ) -> None:
                self.files = ["src/vllm_torchtpu/envs.py"] if files is None else files
                self.labels = labels
                self.meta: list[tuple[str, str]] = []
                self.annotations: list[tuple[str, str]] = []
                self.uploads: list[tuple[str, str, dict[str, str]]] = []

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
                elif args[:1] == ("annotate",):
                    self.annotations.append((args[4], args[5]))
                elif args[:2] == ("pipeline", "upload") and input_text is not None:
                    if "--dry-run" in args:
                        doc = yaml.safe_load(expand_env(input_text, env_map)) or {}
                        out = json.dumps(
                            {k: v for k, v in doc.items() if not k.startswith("x-")}
                        )
                        return subprocess.CompletedProcess(args, 0, stdout=out)
                    rendered = (
                        input_text
                        if "--no-interpolation" in args
                        else expand_env(input_text, env_map)
                    )
                    tag = next(
                        (k for k, needle in tags if needle in rendered), "unknown"
                    )
                    self.uploads.append((tag, rendered, dict(env_map)))
                return subprocess.CompletedProcess(args, 0, stdout="")

            def git(
                self,
                *args: str,
                check: bool = False,
                env: Mapping[str, str] | None = None,
                cwd: pathlib.Path = _REPO_ROOT,
            ) -> subprocess.CompletedProcess[str]:
                if args[:1] == ("ls-remote",):
                    return subprocess.CompletedProcess(
                        args, 0, stdout="upstream999\tHEAD\n"
                    )
                if args[:1] in (("diff",), ("diff-tree",)):
                    return subprocess.CompletedProcess(
                        args, 0, stdout="\n".join(self.files) + "\n"
                    )
                return subprocess.CompletedProcess(args, 0, stdout="")

            def github_labels(
                self, pr_number: str, *, env: Mapping[str, str] | None = None
            ) -> tuple[str, ...]:
                return self.labels

        def run(
            env: dict[str, str],
            *,
            files: list[str] | None = None,
            labels: tuple[str, ...] = (),
        ) -> RunRecorder:
            rec = RunRecorder(files=files, labels=labels)
            with (
                mock.patch.object(bp, "_agent", side_effect=rec.agent),
                mock.patch.object(bp, "_git", side_effect=rec.git),
                mock.patch.object(bp, "_github_labels", side_effect=rec.github_labels),
            ):
                self.assertEqual(bp.run_bootstrap(env, repo_root=_REPO_ROOT), 0)
            return rec

        def run_next_stage(prev: RunRecorder) -> RunRecorder:
            self.assertEqual(prev.uploads[0][0], "upload_stages")
            uploader = json.loads(prev.uploads[0][1])
            self.assertEqual(uploader["steps"][0], "wait")
            step = uploader["steps"][1]
            self.assertEqual(step["agents"], {"queue": "cpu"})
            prefix = "python3 .buildkite/scripts/bootstrap.py upload-stages "
            self.assertTrue(step["command"].startswith(prefix))
            raw_args = step["command"].removeprefix(prefix).split()
            stages = tuple(bp._split_labels(arg) for arg in raw_args)
            rec = RunRecorder()
            with mock.patch.object(bp, "_agent", side_effect=rec.agent):
                self.assertEqual(
                    bp.upload_stages(stages, step["env"], repo_root=_REPO_ROOT),
                    0,
                )
            return rec

        # Unknown CI_SUITE and removed CI_LANES / SKIP_TPU_CI knobs fail clearly.
        for bad_suite in ("bogus", "cpu-only"):
            with self.assertRaisesRegex(
                bp.BootstrapError,
                rf"ERROR: unknown CI_SUITE '{bad_suite}'; "
                r"suites are: default full perf integration",
            ):
                run({"CI_SUITE": bad_suite})
        with self.assertRaisesRegex(
            bp.BootstrapError,
            r"ERROR: CI_LANES was removed; set CI_SUITE=<name> "
            r"\(suites: default full perf integration\)",
        ):
            run({"CI_LANES": "perf"})
        with self.assertRaisesRegex(
            bp.BootstrapError,
            r"ERROR: SKIP_TPU_CI was removed; use the skip-tpu-ci PR label "
            r"\(or 'bootstrap\.py upload-stages build,cpu'\)",
        ):
            run({"SKIP_TPU_CI": "1"})

        # PR default: priority 6, stage 0 uploads upload_stages (wait + uploader)
        # + cpu + build; stage 1 uploads verify + integration + tests.
        pr = run(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com",
            }
        )
        self.assertEqual(
            pr.meta,
            [
                ("JOB_PRIORITY", "6"),
                ("WORKLOAD_PRIORITY", "pre-merge"),
                ("VLLM_COMMIT_HASH", pinned),
            ],
        )
        self.assertEqual(
            pr.annotations,
            [("ci-suite", "CI suite: pr: PR: TPU waits for the CPU checks")],
        )
        self.assertEqual(
            [u[0] for u in pr.uploads],
            ["upload_stages", "cpu", "build"],
        )
        self.assertIn(
            ":pipeline: Upload tests, integration, verify",
            pr.uploads[0][1],
        )
        self.assertNotIn("CPU_GATE", pr.uploads[1][2])
        self.assertNotIn("SKIP_TPU_CI", pr.uploads[1][2])
        self.assertEqual(pr.uploads[2][2]["OMNI_IMAGE_SKIP"], "")
        self.assertNotIn("tpu-pr/alice@example.com", pr.uploads[1][1])
        pr_stage1 = run_next_stage(pr)
        self.assertEqual(
            [u[0] for u in pr_stage1.uploads],
            ["verify", "integration", "tests"],
        )
        self.assertIn("tpu-pr/alice@example.com", pr_stage1.uploads[2][1])

        # 3-stage test-only suite chains stages 0 -> 1 -> 2 via upload-stages.
        three_stage_suite = bp.Suite(
            "three-stage",
            "3-stage test suite",
            stages=(("build", "cpu"), ("tests", "integration"), ("verify",)),
        )
        with mock.patch.dict(bp.SUITES, {"three-stage": three_stage_suite}):
            chain_s0 = run({"CI_SUITE": "three-stage"})
            self.assertEqual(
                [u[0] for u in chain_s0.uploads],
                ["upload_stages", "cpu", "build"],
            )
            self.assertIn(
                ":pipeline: Upload tests, integration",
                chain_s0.uploads[0][1],
            )
            chain_s1 = run_next_stage(chain_s0)
            self.assertEqual(
                [u[0] for u in chain_s1.uploads],
                ["upload_stages", "integration", "tests"],
            )
            self.assertIn(
                ":pipeline: Upload verify",
                chain_s1.uploads[0][1],
            )
            chain_s2 = run_next_stage(chain_s1)
            self.assertEqual([u[0] for u in chain_s2.uploads], ["verify"])

        # PR run-perf-eval: pr+perf-eval suite, NIGHTLY unset.
        perf_eval_pr = run(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "ready,run-perf-eval",
            }
        )
        self.assertEqual(
            perf_eval_pr.annotations,
            [("ci-suite", "CI suite: pr+perf-eval: PR: TPU waits for the CPU checks")],
        )
        self.assertEqual(
            [u[0] for u in perf_eval_pr.uploads],
            ["upload_stages", "cpu", "build"],
        )
        perf_eval_stage1 = run_next_stage(perf_eval_pr)
        self.assertEqual(
            [u[0] for u in perf_eval_stage1.uploads],
            ["verify", "perf", "integration", "tests"],
        )
        self.assertNotIn("NIGHTLY", perf_eval_stage1.uploads[0][2])

        # PR skip-tpu-ci & post-merge main webhook upload ONLY ["cpu", "build"].
        for fx, prio, wprio in (
            (
                run(
                    {
                        "BUILDKITE_PULL_REQUEST": "42",
                        "BUILDKITE_PULL_REQUEST_LABELS": "ready,skip-tpu-ci",
                    }
                ),
                "6",
                "pre-merge",
            ),
            (
                run(
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
            self.assertEqual(
                fx.annotations,
                [("ci-suite", "CI suite: pr+cpu-only: CPU checks only (skip-tpu-ci)")],
            )
            self.assertEqual([u[0] for u in fx.uploads], ["cpu", "build"])
            self.assertTrue(bool(fx.uploads[1][2]["OMNI_IMAGE_SKIP"]))
            self.assertLessEqual(len(fx.uploads[1][2]["OMNI_IMAGE_SKIP"]), 70)

        # main post-merge (even when merged PR had run-full-ci / run-perf-eval)
        # runs default suite in one stage without NIGHTLY.
        main_pm = run(
            {
                "BUILDKITE_BRANCH": "main",
                "BUILDKITE_SOURCE": "webhook",
                "BUILDKITE_MESSAGE": "feat: land change (#100)",
            },
            labels=("run-full-ci", "run-perf-eval"),
        )
        self.assertIn(("JOB_PRIORITY", "11"), main_pm.meta)
        self.assertIn(("WORKLOAD_PRIORITY", "post-merge"), main_pm.meta)
        self.assertEqual(
            main_pm.annotations,
            [("ci-suite", "CI suite: default: main, torchtpu-nightly")],
        )
        self.assertEqual(
            [u[0] for u in main_pm.uploads],
            ["verify", "integration", "tests", "cpu", "build"],
        )
        self.assertNotIn("NIGHTLY", main_pm.uploads[0][2])

        # CI_SUITE=perf (with transitional CI_LANES) uploads ONLY ["perf", "build"].
        perf_sched = run(
            {"CI_SUITE": "perf", "CI_LANES": "perf", "BUILDKITE_BRANCH": "main"}
        )
        self.assertEqual([u[0] for u in perf_sched.uploads], ["perf", "build"])
        self.assertTrue(bool(perf_sched.uploads[1][2]["OMNI_IMAGE_SKIP"]))

        # skip-tpu-ci + run-full-ci uploads ["cpu", "build"] with NIGHTLY=1 substituted.
        skip_full = run(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": (
                    "skip-tpu-ci,run-full-ci,run-perf-eval"
                ),
            }
        )
        self.assertEqual(
            skip_full.annotations,
            [
                (
                    "ci-suite",
                    "CI suite: pr+cpu-only+nightly-suite: "
                    "CPU checks only (skip-tpu-ci)",
                )
            ],
        )
        self.assertEqual([u[0] for u in skip_full.uploads], ["cpu", "build"])
        self.assertEqual(skip_full.uploads[0][2]["NIGHTLY"], "1")
        self.assertNotIn('build.env("NIGHTLY")', skip_full.uploads[0][1])

        # PR run-full-ci + oncall-fix: priority 12, oncall-fix warning, still 2-stage
        # pr+nightly-suite (oncall-fix changes priority only, not suite).
        oncall_full = run(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "run-full-ci,oncall-fix",
                "BUILDKITE_BUILD_AUTHOR_EMAIL": "Oncall@Example.com",
            }
        )
        self.assertIn(("JOB_PRIORITY", "12"), oncall_full.meta)
        self.assertIn(("WORKLOAD_PRIORITY", "oncall-fix"), oncall_full.meta)
        self.assertEqual(
            oncall_full.annotations,
            [
                (
                    "oncall-fix",
                    "Runs at on-call priority (`oncall-fix` label): "
                    "ahead of every other build, including main.",
                ),
                (
                    "ci-suite",
                    "CI suite: pr+nightly-suite: PR: TPU waits for the CPU checks",
                ),
            ],
        )
        self.assertEqual(
            [u[0] for u in oncall_full.uploads],
            ["upload_stages", "cpu", "build"],
        )
        oncall_stage1 = run_next_stage(oncall_full)
        self.assertEqual(
            [u[0] for u in oncall_stage1.uploads],
            ["verify", "perf", "integration", "tests"],
        )
        self.assertEqual(json.loads(oncall_stage1.uploads[0][1])["priority"], 12)
        self.assertIn("tpu-pr/oncall@example.com", oncall_stage1.uploads[3][1])
        for _, content, _ in (*oncall_full.uploads[1:], *oncall_stage1.uploads):
            self.assertNotIn('build.env("NIGHTLY")', content)

        # Scheduled Nightly and release builds (tag or releases/vX.Y.Z branch)
        # run 1-stage full suite with NIGHTLY=1 substituted.
        nightly = run(
            {
                "NIGHTLY": "1",
                "BUILDKITE_SOURCE": "schedule",
                "BUILDKITE_BRANCH": "main",
            }
        )
        self.assertEqual(
            nightly.meta, [("JOB_PRIORITY", "1"), ("VLLM_COMMIT_HASH", pinned)]
        )
        tag_fx = run({"BUILDKITE_TAG": "v0.32.0", "BUILDKITE_BRANCH": "v0.32.0"})
        self.assertEqual(
            tag_fx.meta, [("JOB_PRIORITY", "2"), ("VLLM_COMMIT_HASH", pinned)]
        )
        rel_branch_fx = run({"BUILDKITE_BRANCH": "releases/v0.32.0"})
        self.assertEqual(
            rel_branch_fx.meta, [("JOB_PRIORITY", "2"), ("VLLM_COMMIT_HASH", pinned)]
        )
        for fx in (nightly, tag_fx, rel_branch_fx):
            self.assertEqual(
                fx.annotations,
                [("ci-suite", "CI suite: full: Nightly schedule and release builds")],
            )
            self.assertEqual(
                [u[0] for u in fx.uploads],
                ["verify", "perf", "integration", "tests", "cpu", "build"],
            )
            for _, content, _ in fx.uploads:
                self.assertNotIn('build.env("NIGHTLY")', content)

        # torchtpu-nightly schedule runs default suite at priority 1 (one stage).
        torchtpu_nightly = run(
            {
                "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-torchtpu-nightly",
                "TORCHTPU_NIGHTLY": "1",
                "BUILDKITE_SOURCE": "schedule",
                "BUILDKITE_BRANCH": "main",
            }
        )
        self.assertEqual(
            torchtpu_nightly.meta,
            [("JOB_PRIORITY", "1"), ("VLLM_COMMIT_HASH", pinned)],
        )
        self.assertEqual(
            [u[0] for u in torchtpu_nightly.uploads],
            ["verify", "integration", "tests", "cpu", "build"],
        )

        integ_manual = run({"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration"})
        self.assertEqual(
            [u[0] for u in integ_manual.uploads],
            ["promote", "verify", "perf", "integration", "tests", "cpu", "build"],
        )

        integ = run(
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
        self.assertEqual(run({"BUILDKITE_MESSAGE": "doc [skip ci]"}).uploads, [])
        self.assertEqual(
            run({"BUILDKITE_PULL_REQUEST": "42"}, files=["README.md"]).uploads, []
        )

        # Validate raw fragments, combined rendered pipelines (in Buildkite display
        # order: reversed(uploads)), and nightly_report.sh's step-key scan.
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

            skip_fx = run(
                {
                    "BUILDKITE_PULL_REQUEST": "42",
                    "BUILDKITE_PULL_REQUEST_LABELS": "skip-tpu-ci",
                }
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
        report_deps = set(verify_doc["steps"][1]["depends_on"]) - {
            "v6e-group",
            "v7x-group",
        }
        scanned = subprocess.run(
            [
                "bash",
                "-c",
                "grep -hoE '^[[:space:]]*key:[[:space:]]*\"?[A-Za-z0-9_-]+' "
                ".buildkite/pipeline_cpu.yml .buildkite/pipeline_tests.yml "
                "| sed -E 's/.*key:[[:space:]]*\"?//' "
                "| grep -Ev '(_pr|^cpu_checks)$'",
            ],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(set(scanned.stdout.split()), report_deps)


if __name__ == "__main__":
    unittest.main()
