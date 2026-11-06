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

# Every file a lane uploads: one per lane, and the tests lane's CPU steps.
_ALL_LANE_FILES = frozenset(
    f".buildkite/pipeline_{name}.yml"
    for name in ("verify", "perf", "integration", "tests", "tests_cpu")
)

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


def _render_doc(path: str, env: Mapping[str, str]) -> dict[str, Any]:
    """Pipeline file `path` as uploaded with `env`, parsed."""
    import yaml  # pyrefly: ignore[missing-import]

    # Buildkite's upload-time interpolation, as far as these files use
    # it: ${VAR}, ${VAR:-default}, and $$ for a literal $.
    text = (_REPO_ROOT / path).read_text(encoding="utf-8").replace("$$", "\0")
    text = re.sub(
        r"\$\{(\w+)(?::-([^}]*))?\}",
        lambda m: env.get(m.group(1)) or m.group(2) or "",
        text,
    )
    return yaml.safe_load(text.replace("\0", "$"))


def _render_steps(path: str, env: Mapping[str, str]) -> list[dict[str, Any]]:
    """The steps of `path` as uploaded with `env`, with groups flattened."""
    steps: list[dict[str, Any]] = []
    for step in _render_doc(path, env)["steps"]:
        steps.extend(step["steps"] if "group" in step else [step])
    return steps


# How a step's `if` names the generation it runs on: "v6e" == "v7x" and the like.
_GENERATION_CHECK_RE = re.compile(r'"\w+" == "v7x"')


def _same_test(step: dict[str, Any]) -> dict[str, Any]:
    # A step less what a generation changes: its names, chips, soft_fail, and
    # the generation its `if` compares, which is checked on its own.
    rest = {k: v for k, v in step.items() if k not in ("label", "key", "soft_fail")}
    if "env" in step:
        rest["env"] = {k: v for k, v in step["env"].items() if k != "SHAPE"}
    if "if" in step:
        rest["if"] = _GENERATION_CHECK_RE.sub('"<gen>" == "v7x"', step["if"])
    return rest


class TestBootstrapUnits(unittest.TestCase):
    """Focused unit tests for planning helpers, effects resilience, and upload CLI."""

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

    def test_plan_priority_and_concurrency_group(self) -> None:
        bp = self.bp
        nightly = bp.plan_priority(bp.BuildContext.from_env({"NIGHTLY": "1"}))
        self.assertEqual(nightly.job_priority, bp.PRIORITY_NIGHTLY)
        self.assertIsNone(nightly.workload_priority)
        self.assertIsNone(nightly.pr_tpu_concurrency_group)

        pr = bp.plan_priority(
            bp.BuildContext.from_env(
                {
                    "BUILDKITE_PULL_REQUEST": "123",
                    "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@Example.com",
                }
            )
        )
        self.assertEqual(pr.job_priority, bp.PRIORITY_PRE_MERGE)
        self.assertEqual(pr.workload_priority, "pre-merge")
        self.assertEqual(pr.pr_tpu_concurrency_group, "tpu-pr/alice@example.com")

        oncall = bp.plan_priority(
            bp.BuildContext.from_env(
                {
                    "BUILDKITE_PULL_REQUEST": "123",
                    "BUILDKITE_PULL_REQUEST_LABELS": "ready,oncall-fix",
                    "BUILDKITE_BUILD_CREATOR_EMAIL": "Oncall@Example.com",
                }
            )
        )
        self.assertEqual(oncall.job_priority, bp.PRIORITY_ONCALL_FIX)
        self.assertEqual(oncall.workload_priority, "oncall-fix")
        self.assertEqual(oncall.annotation.context, "oncall-fix")
        self.assertEqual(oncall.pr_tpu_concurrency_group, "tpu-pr/oncall@example.com")

        post_merge = bp.plan_priority(
            bp.BuildContext.from_env({"BUILDKITE_BRANCH": "main"})
        )
        self.assertEqual(post_merge.job_priority, bp.PRIORITY_POST_MERGE)
        self.assertEqual(post_merge.workload_priority, "post-merge")

        general = bp.plan_priority(
            bp.BuildContext.from_env({"BUILDKITE_BRANCH": "release/v1"})
        )
        self.assertEqual(general.job_priority, bp.PRIORITY_DEFAULT)
        self.assertIsNone(general.workload_priority)

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
        self.assertEqual(
            bp.resolve_vllm_commit_hash("abc123def", "override99"), "override99"
        )
        with self.assertRaises(bp.BootstrapError):
            bp.parse_pinned_vllm_commit("[project]\nname = 'vllm-torchtpu'\n")

        self.assertEqual(
            bp.parse_merged_pr_number("feat: add kernel (#789)\nbody (#999)"), "789"
        )
        self.assertIsNone(bp.parse_merged_pr_number("chore: direct commit"))

    def test_plan_uploads_lanes_overrides_and_skip_reasons(self) -> None:
        bp = self.bp
        default_plan = bp.plan_uploads(
            bp.BuildContext.from_env({"BUILDKITE_PULL_REQUEST": "10"}),
            merged_labels=(),
            existing_lane_files=_ALL_LANE_FILES,
        )
        self.assertEqual(default_plan.lanes, ("verify", "integration", "tests"))
        # The tests lane is two files, TPU then CPU, so the CPU steps show first.
        self.assertEqual(
            default_plan.lane_files,
            (
                ".buildkite/pipeline_verify.yml",
                ".buildkite/pipeline_integration.yml",
                ".buildkite/pipeline_tests.yml",
                ".buildkite/pipeline_tests_cpu.yml",
            ),
        )
        self.assertEqual(default_plan.omni_image_skip, "")

        # run-full-ci + oncall-fix turns on NIGHTLY and all four lanes.
        full_plan = bp.plan_uploads(
            bp.BuildContext.from_env(
                {
                    "BUILDKITE_PULL_REQUEST": "10",
                    "BUILDKITE_PULL_REQUEST_LABELS": "ready,run-full-ci,oncall-fix",
                }
            ),
            merged_labels=(),
            existing_lane_files=_ALL_LANE_FILES,
        )
        self.assertTrue(full_plan.nightly)
        self.assertEqual(full_plan.lanes, bp.ALL_LANES)
        self.assertEqual(full_plan.pre_lane_annotations[0].context, "full-ci")

        # skip-tpu-ci overrides run-full-ci and run-perf-eval; skip reason <= 70 chars.
        skip_plan = bp.plan_uploads(
            bp.BuildContext.from_env(
                {
                    "BUILDKITE_PULL_REQUEST": "10",
                    "BUILDKITE_PULL_REQUEST_LABELS": (
                        "run-full-ci,run-perf-eval,skip-tpu-ci"
                    ),
                }
            ),
            merged_labels=(),
            existing_lane_files=_ALL_LANE_FILES,
        )
        self.assertTrue(skip_plan.skip_tpu_ci)
        self.assertEqual(skip_plan.lanes, ("tests",))
        self.assertEqual(
            skip_plan.lane_files,
            (".buildkite/pipeline_tests.yml", ".buildkite/pipeline_tests_cpu.yml"),
        )
        self.assertLessEqual(len(skip_plan.omni_image_skip), 70)

        # Post-merge webhook honors merged PR's skip-tpu-ci label; CI_LANES=perf
        # skips the -omni image.
        post_merge_skip = bp.plan_uploads(
            bp.BuildContext.from_env(
                {"BUILDKITE_BRANCH": "main", "BUILDKITE_SOURCE": "webhook"}
            ),
            merged_labels=("skip-tpu-ci",),
            existing_lane_files=_ALL_LANE_FILES,
        )
        self.assertTrue(post_merge_skip.skip_tpu_ci)
        self.assertEqual(post_merge_skip.lanes, ("tests",))

        perf_only = bp.plan_uploads(
            bp.BuildContext.from_env({"CI_LANES": "perf"}),
            merged_labels=(),
            existing_lane_files=_ALL_LANE_FILES,
        )
        self.assertEqual(perf_only.lanes, ("perf",))
        self.assertLessEqual(len(perf_only.omni_image_skip), 70)

        # Invalid CI_LANES and missing lane handling (warn by default, fail when named).
        invalid = bp.plan_uploads(
            bp.BuildContext.from_env({"CI_LANES": "bogus"}),
            merged_labels=(),
            existing_lane_files=_ALL_LANE_FILES,
        )
        self.assertIsNotNone(invalid.invalid_ci_lanes_error)

        gap_warn = bp.plan_uploads(
            bp.BuildContext.from_env({}),
            merged_labels=(),
            existing_lane_files={".buildkite/pipeline_tests.yml"},
        )
        self.assertIsNone(gap_warn.missing_ci_lanes_error)
        self.assertEqual(gap_warn.post_lane_annotations[0].context, "lane-gaps")

        gap_err = bp.plan_uploads(
            bp.BuildContext.from_env({"CI_LANES": "perf"}),
            merged_labels=(),
            existing_lane_files={".buildkite/pipeline_tests.yml"},
        )
        self.assertIsNotNone(gap_err.missing_ci_lanes_error)

        # The tests lane's CPU file has to exist too when the lane is named.
        no_cpu_file = bp.plan_uploads(
            bp.BuildContext.from_env({"CI_LANES": "tests"}),
            merged_labels=(),
            existing_lane_files={".buildkite/pipeline_tests.yml"},
        )
        self.assertIn(
            "pipeline_tests_cpu.yml", no_cpu_file.missing_ci_lanes_error or ""
        )

        rendered = bp.prepare_upload_yaml(
            'if: \'build.env("NIGHTLY") == "1"\'\n', job_priority=12, nightly=True
        )
        self.assertEqual(rendered, 'priority: 12\nif: \'"1" == "1"\'\n')

    def test_effects_github_error_resilience(self) -> None:
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

            slow_helper = tmp_path / "slow-credential-helper"
            slow_helper.write_text("#!/usr/bin/env bash\nsleep 5\n", encoding="utf-8")
            slow_helper.chmod(slow_helper.stat().st_mode | stat.S_IEXEC)

            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), AuthHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                api_url = f"http://127.0.0.1:{server.server_address[1]}"
                ok_effects = bp.BootstrapEffects(
                    repo_root=_REPO_ROOT,
                    credential_helper=helper,
                    github_api_url=api_url,
                )
                timeout_effects = bp.BootstrapEffects(
                    repo_root=_REPO_ROOT,
                    credential_helper=slow_helper,
                    credential_timeout=0.05,
                    github_api_url=api_url,
                )
                self.assertEqual(
                    ok_effects.merged_pr_labels("42", env={}), ("skip-tpu-ci",)
                )
                self.assertEqual(
                    timeout_effects.merged_pr_labels("42", env={}), ("skip-tpu-ci",)
                )
            finally:
                server.shutdown()
                server.server_close()

            self.assertEqual(seen_auth, ["token ghs_tok", None])

            with mock.patch(
                "urllib.request.urlopen",
                side_effect=http.client.IncompleteRead(b"partial", 10),
            ):
                err_effects = bp.BootstrapEffects(
                    repo_root=_REPO_ROOT,
                    credential_helper=tmp_path / "missing",
                )
                self.assertEqual(err_effects.merged_pr_labels("42", env={}), ())

    def test_dev_mode_vllm_pin_and_shell_wrappers(self) -> None:
        bp = self.bp
        dev_default = bp.plan_dev_bootstrap({})
        self.assertEqual(dev_default.raw_job_priority, str(bp.PRIORITY_DEFAULT))
        self.assertEqual(dev_default.pipeline_file, ".buildkite/pipeline_dev.yml")
        self.assertEqual(dev_default.branch, "unknown")
        self.assertEqual(dev_default.commit, "unknown")

        dev_custom = bp.plan_dev_bootstrap(
            {
                "DEV_JOB_PRIORITY": "4",
                "DEV_PIPELINE_FILE": ".buildkite/exp.yml",
                "BUILDKITE_BRANCH": "exp/test",
                "BUILDKITE_COMMIT": "abc1234",
                "NIGHTLY": "1",
            }
        )
        self.assertEqual(bp.parse_dev_job_priority(dev_custom.raw_job_priority), 4)
        self.assertEqual(dev_custom.pipeline_file, ".buildkite/exp.yml")
        self.assertTrue(dev_custom.nightly)
        with self.assertRaises(bp.BootstrapError):
            bp.parse_dev_job_priority("not-an-int")

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

    def test_run_bootstrap_orchestration(self) -> None:
        import yaml  # pyrefly: ignore[missing-import]

        bp = self.bp
        pinned = bp.parse_pinned_vllm_commit(
            (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        tags = (
            ("promote", "promote_lkg"),
            ("verify", "Support matrices and verified commit"),
            ("perf", "Perf and Eval on TPU"),
            ("integration", "integration_tpu"),
            ("tests", "Unit Tests on TPU"),
            ("tests_cpu", "Unit Tests - CPU (PR)"),
            ("build", "build_docker_group"),
        )

        class FakeEffects(bp.BootstrapEffects):
            def __init__(
                self,
                files: list[str] | None = None,
                labels: tuple[str, ...] = (),
            ) -> None:
                super().__init__(repo_root=_REPO_ROOT)
                self.files = ["src/vllm_torchtpu/envs.py"] if files is None else files
                self.labels = labels
                self.meta: list[tuple[str, str]] = []
                self.annotations: list[str] = []
                self.uploads: list[tuple[str, str, dict[str, str]]] = []

            def set_metadata(
                self, key: str, value: str, *, env: Mapping[str, str]
            ) -> None:
                self.meta.append((key, value))

            def annotate(self, annotation: Any, *, env: Mapping[str, str]) -> None:
                self.annotations.append(annotation.context)

            def dry_run_upload(
                self, content: str, *, env: Mapping[str, str]
            ) -> str | None:
                doc = yaml.safe_load(content) or {}
                return json.dumps(
                    {k: v for k, v in doc.items() if not k.startswith("x-")}
                )

            def upload_pipeline(
                self, content: str, *, no_interpolation: bool, env: Mapping[str, str]
            ) -> None:
                tag = next((k for k, needle in tags if needle in content), "unknown")
                if "TPU_GENERATION" in env:
                    tag = f"{tag}:{env['TPU_GENERATION']}"
                self.uploads.append((tag, content, dict(env)))

            def pr_changed_files(
                self, base_branch: str, commit: str, *, env: Mapping[str, str]
            ) -> list[str]:
                return list(self.files)

            def commit_changed_files(
                self, commit: str, *, env: Mapping[str, str]
            ) -> list[str]:
                return list(self.files)

            def merged_pr_labels(
                self, pr_number: str, *, env: Mapping[str, str]
            ) -> tuple[str, ...]:
                return self.labels

            def vllm_upstream_head(self, *, env: Mapping[str, str]) -> str:
                return "upstream999"

        def run(
            env: dict[str, str],
            *,
            files: list[str] | None = None,
            labels: tuple[str, ...] = (),
        ) -> FakeEffects:
            fx = FakeEffects(files=files, labels=labels)
            self.assertEqual(bp.run_bootstrap(env, repo_root=_REPO_ROOT, effects=fx), 0)
            return fx

        # The tests lane: its TPU file once per generation, then its CPU file.
        tests_lane = ["tests:v6e", "tests:v7x", "tests_cpu"]
        # The integration lane: its one file, once per generation.
        integration_lane = ["integration:v6e", "integration:v7x"]

        # PR default: priority 6, pre-merge class, 7 uploads, SKIP_TPU_CI=0.
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
        self.assertEqual(pr.annotations, [])
        self.assertEqual(
            [u[0] for u in pr.uploads],
            ["verify", *integration_lane, *tests_lane, "build"],
        )
        contents = {tag: text for tag, text, _ in pr.uploads}
        envs = {tag: upload_env for tag, _, upload_env in pr.uploads}
        self.assertEqual(envs["tests:v6e"]["SKIP_TPU_CI"], "0")
        self.assertEqual(envs["build"]["OMNI_IMAGE_SKIP"], "")
        for lane in ("tests", "integration"):
            v6e, v7x = f"{lane}:v6e", f"{lane}:v7x"
            self.assertEqual(envs[v6e]["KUBE_SHAPE_SINGLE"], "ct6e-standard-1t/1x1")
            self.assertEqual(envs[v6e]["TPU_SOFT_FAIL"], "true")
            self.assertEqual(envs[v7x]["TPU_KEY_PREFIX"], "")
            self.assertEqual(envs[v7x]["TPU_SOFT_FAIL"], "false")
            self.assertIn("tpu-pr/alice@example.com", contents[v6e])
            self.assertIn("tpu-pr/alice@example.com", contents[v7x])
        # The CPU steps hold no chips, so they stay out of the author's TPU cap.
        self.assertNotIn("tpu-pr/alice@example.com", contents["tests_cpu"])

        # PR skip-tpu-ci & post-merge main webhook honoring merged PR skip-tpu-ci.
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
            self.assertEqual(fx.annotations, ["skip-tpu-ci"])
            self.assertEqual([u[0] for u in fx.uploads], [*tests_lane, "build"])
            self.assertEqual({u[2]["SKIP_TPU_CI"] for u in fx.uploads[:2]}, {"1"})
            self.assertTrue(bool(fx.uploads[-1][2]["OMNI_IMAGE_SKIP"]))

        # PR run-full-ci + oncall-fix & scheduled Nightly (no WORKLOAD_PRIORITY).
        full = run(
            {
                "BUILDKITE_PULL_REQUEST": "42",
                "BUILDKITE_PULL_REQUEST_LABELS": "run-full-ci,oncall-fix",
            }
        )
        self.assertIn(("JOB_PRIORITY", "12"), full.meta)
        self.assertIn(("WORKLOAD_PRIORITY", "oncall-fix"), full.meta)
        self.assertEqual(full.annotations, ["oncall-fix", "full-ci"])
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
        for fx in (full, nightly):
            self.assertEqual(
                [u[0] for u in fx.uploads],
                ["verify", "perf", *integration_lane, *tests_lane, "build"],
            )
            for _, content, _ in fx.uploads:
                self.assertNotIn('build.env("NIGHTLY")', content)

        # Integration pipeline (promote first, upstream head) and early exits.
        integ = run({"BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration"})
        self.assertIn(("VLLM_COMMIT_HASH", "upstream999"), integ.meta)
        self.assertEqual(
            [u[0] for u in integ.uploads],
            ["promote", "verify", "perf", *integration_lane, *tests_lane, "build"],
        )
        self.assertEqual(run({"BUILDKITE_MESSAGE": "doc [skip ci]"}).uploads, [])
        self.assertEqual(
            run({"BUILDKITE_PULL_REQUEST": "42"}, files=["README.md"]).uploads, []
        )

        # TPU_VERSION on a build started by hand: each file that goes up per
        # generation goes up for that generation alone, with its shapes; the
        # rest go up as before.
        for version, gen in bp.TPU_VERSIONS.items():
            one = run({"BUILDKITE_BRANCH": "dev/x", "TPU_VERSION": version})
            self.assertEqual(one.annotations, ["tpu-version"])
            self.assertEqual(
                [u[0] for u in one.uploads],
                [
                    "verify",
                    f"integration:{gen.name}",
                    f"tests:{gen.name}",
                    "tests_cpu",
                    "build",
                ],
            )
            for tag, _, upload_env in one.uploads:
                if ":" in tag:
                    self.assertEqual(upload_env["KUBE_SHAPE_SINGLE"], gen.shape_single)
                    self.assertEqual(upload_env["KUBE_SHAPE_MULTI"], gen.shape_multi)
        # A build that must run every generation is refused before any upload.
        refused = FakeEffects()
        with self.assertRaises(bp.BootstrapError):
            bp.run_bootstrap(
                {"NIGHTLY": "1", "TPU_VERSION": "tpu6e"},
                repo_root=_REPO_ROOT,
                effects=refused,
            )
        self.assertEqual(refused.uploads, [])

    def test_tpu_version_runs_one_generation(self) -> None:
        bp = self.bp
        self.assertEqual(
            {version: gen.name for version, gen in bp.TPU_VERSIONS.items()},
            {"tpu6e": "v6e", "tpu7x": "v7x"},
        )
        # Unset, a per-generation file goes up once per generation; set, once,
        # for the generation it names. Any other file goes up once, as it is.
        for path, generations in bp.TPU_GENERATION_FILES.items():
            self.assertEqual(
                bp.upload_envs(path), tuple(g.upload_env() for g in generations)
            )
            for version, gen in bp.TPU_VERSIONS.items():
                self.assertEqual(bp.upload_envs(path, version), (gen.upload_env(),))
        for version in ("", *bp.TPU_VERSIONS):
            self.assertEqual(
                bp.upload_envs(".buildkite/pipeline_perf.yml", version), ({},)
            )

        def plan(env: dict[str, str]) -> Any:
            return bp.plan_uploads(
                bp.BuildContext.from_env(env),
                merged_labels=(),
                existing_lane_files=_ALL_LANE_FILES,
            )

        manual = plan({"TPU_VERSION": "tpu6e"})
        self.assertIsNone(manual.invalid_tpu_version_error)
        self.assertEqual(
            [a.context for a in manual.pre_lane_annotations], ["tpu-version"]
        )
        self.assertIn("v6e", manual.pre_lane_annotations[0].body)
        # With skip-tpu-ci no TPU step runs, so there is nothing to annotate.
        skip = plan({"TPU_VERSION": "tpu6e", "SKIP_TPU_CI": "1"})
        self.assertIsNone(skip.invalid_tpu_version_error)
        self.assertEqual(
            [a.context for a in skip.pre_lane_annotations], ["skip-tpu-ci"]
        )
        # Unset, there is nothing to refuse, on any build.
        self.assertIsNone(plan({"NIGHTLY": "1"}).invalid_tpu_version_error)
        # Refused: a name it does not know, builds that must run every
        # generation, and a build with no lane that goes up per generation.
        for env in (
            {"TPU_VERSION": "v6e"},
            {"TPU_VERSION": "tpu6e", "NIGHTLY": "1"},
            {
                "TPU_VERSION": "tpu6e",
                "BUILDKITE_PULL_REQUEST": "10",
                "BUILDKITE_PULL_REQUEST_LABELS": "run-full-ci",
            },
            {"TPU_VERSION": "tpu7x", "BUILDKITE_TAG": "v0.33.0"},
            {
                "TPU_VERSION": "tpu7x",
                "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration",
            },
            {"TPU_VERSION": "tpu6e", "CI_LANES": "perf"},
        ):
            self.assertIsNotNone(plan(env).invalid_tpu_version_error, env)

    def test_tests_lane_tpu_generations(self) -> None:
        import yaml  # pyrefly: ignore[missing-import]

        bp = self.bp
        tpu_file = ".buildkite/pipeline_tests.yml"
        generations = bp.TPU_GENERATION_FILES[tpu_file]
        self.assertEqual([g.name for g in generations], ["v6e", "v7x"])
        on = {g.name: _render_steps(tpu_file, g.upload_env()) for g in generations}

        # The same tests, filtered and sharded the same way, on each generation.
        self.assertEqual(
            [_same_test(s) for s in on["v6e"]], [_same_test(s) for s in on["v7x"]]
        )
        # v7x is what an upload of the file on its own gives: the keys it had.
        self.assertEqual(on["v7x"], _render_steps(tpu_file, {}))
        self.assertEqual({s["soft_fail"] for s in on["v7x"]}, {"false"})
        self.assertEqual({s["soft_fail"] for s in on["v6e"]}, {"true"})
        # Each step's label names the generation it runs on.
        for gen, steps in on.items():
            self.assertTrue(all(s["label"].startswith(f"{gen} ") for s in steps))
        self.assertEqual(
            {s["env"]["SHAPE"] for s in on["v6e"]},
            {bp.TPU_V6E.shape_single, bp.TPU_V6E.shape_multi},
        )

        # Buildkite rejects a duplicate key, and GitHub names a commit status
        # after its step's label, so neither may repeat across the lane.
        cpu = _render_steps(".buildkite/pipeline_tests_cpu.yml", {})
        lane = cpu + on["v6e"] + on["v7x"]
        for field in ("key", "label"):
            names = [s[field] for s in lane]
            self.assertEqual(len(names), len(set(names)), field)

        # The tests-lane steps the nightly report waits on are CPU or v7x ones.
        verify = yaml.safe_load(
            (_REPO_ROOT / ".buildkite/pipeline_verify.yml").read_text(encoding="utf-8")
        )
        report = next(s for s in verify["steps"] if s.get("key") == "nightly_report")
        waits_on = {k for k in report["depends_on"] if not k.endswith("-group")}
        self.assertLessEqual(waits_on, {s["key"] for s in cpu + on["v7x"]})

    def test_integration_lane_tpu_generations(self) -> None:
        bp = self.bp
        path = ".buildkite/pipeline_integration.yml"
        generations = bp.TPU_GENERATION_FILES[path]
        self.assertEqual([g.name for g in generations], ["v6e", "v7x"])
        on = {g.name: _render_steps(path, g.upload_env()) for g in generations}

        # The same steps on each generation, and v7x's as the file on its own.
        self.assertEqual(
            [_same_test(s) for s in on["v6e"]], [_same_test(s) for s in on["v7x"]]
        )
        self.assertEqual(on["v7x"], _render_steps(path, {}))
        self.assertEqual({s["soft_fail"] for s in on["v7x"]}, {"false"})
        self.assertEqual({s["soft_fail"] for s in on["v6e"]}, {"true"})
        for gen in (bp.TPU_V6E, bp.TPU_V7X):
            steps = on[gen.name]
            self.assertTrue(all(s["label"].startswith(f"{gen.name} ") for s in steps))
            shapes = {s.get("env", {}).get("SHAPE") for s in steps}
            self.assertEqual(shapes - {None}, {gen.shape_single, gen.shape_multi})

        # A step pinned to v7x - run_multihost.sh's, which name no SHAPE, or one
        # setting TPU_ACCELERATOR_TYPE=tpu7x - is left out of v6e's upload by
        # its `if`, and no other step is. v7x's upload leaves none out.
        def pinned_to_v7x(step: dict[str, Any]) -> bool:
            env = step.get("env", {})
            return "SHAPE" not in env or env.get("TPU_ACCELERATOR_TYPE") == "tpu7x"

        def checks(step: dict[str, Any]) -> list[str]:
            return _GENERATION_CHECK_RE.findall(step.get("if", ""))

        pinned = {s["key"] for s in on["v6e"] if pinned_to_v7x(s)}
        self.assertTrue(pinned)
        self.assertLess(len(pinned), len(on["v6e"]))
        self.assertEqual(
            {s["key"] for s in on["v6e"] if '"v6e" == "v7x"' in checks(s)}, pinned
        )
        self.assertEqual({c for s in on["v7x"] for c in checks(s)}, {'"v7x" == "v7x"'})

        # The group and every step have their own key, and every step its own
        # label, across both uploads.
        groups = [
            s["key"]
            for g in generations
            for s in _render_doc(path, g.upload_env())["steps"]
            if "group" in s
        ]
        keys = groups + [s["key"] for s in on["v6e"] + on["v7x"]]
        self.assertEqual(len(keys), len(set(keys)))
        labels = [s["label"] for s in on["v6e"] + on["v7x"]]
        self.assertEqual(len(labels), len(set(labels)))


if __name__ == "__main__":
    unittest.main()
