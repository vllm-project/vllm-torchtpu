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
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from typing import Any
from unittest import mock

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / ".buildkite" / "scripts"

_ALL_LANE_FILES = frozenset(
    f".buildkite/pipeline_{lane}.yml"
    for lane in ("verify", "perf", "integration", "tests")
)

_FAKE_AGENT_BODY = """
import json
import os
import pathlib
import sys
import yaml

log_path = pathlib.Path(os.environ["TEST_AGENT_LOG"])
args = sys.argv[1:]
if args[:2] == ["meta-data", "set"]:
    print("agent: meta-data set", flush=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": [args[2], args[3]]}) + "\\n")
elif args[:2] == ["pipeline", "upload"]:
    content = sys.stdin.read()
    if "--dry-run" in args:
        if os.environ.get("TEST_DRY_RUN_FAIL") == "1":
            sys.exit(1)
        doc = yaml.safe_load(content) or {}
        clean = {k: v for k, v in doc.items() if not k.startswith("x-")}
        json.dump(clean, sys.stdout)
    else:
        print("agent: pipeline upload", flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"upload_args": args[2:], "content": content}) + "\\n"
            )
"""


def _install_agent_stubs(bin_dir: pathlib.Path) -> None:
    agent = bin_dir / "buildkite-agent"
    agent.write_text(
        f"#!{sys.executable}\n{_FAKE_AGENT_BODY.lstrip()}", encoding="utf-8"
    )
    agent.chmod(agent.stat().st_mode | stat.S_IEXEC)
    py3_link = bin_dir / "python3"
    if not py3_link.exists():
        py3_link.symlink_to(sys.executable)


class TestBootstrapUnits(unittest.TestCase):
    """Focused unit tests for CI/dev planning, `vllm-pin`, resilience, and wrappers."""

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

    def test_plan_priority_and_dev_bootstrap_plan(self) -> None:
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

        dev_default = bp.plan_dev_bootstrap({})
        self.assertEqual(dev_default.job_priority, bp.PRIORITY_DEFAULT)
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
        self.assertEqual(dev_custom.job_priority, 4)
        self.assertEqual(dev_custom.pipeline_file, ".buildkite/exp.yml")
        self.assertTrue(dev_custom.nightly)
        with self.assertRaises(bp.BootstrapError):
            bp.plan_dev_bootstrap({"DEV_JOB_PRIORITY": "not-an-int"})

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

        # tomllib ignores commented-out pins and reads [dependency-groups].vllm.
        pyproject = (
            "[dependency-groups]\nvllm = [\n"
            '  # "vllm @ git+https://github.com/vllm-project/vllm.git@stale",\n'
            '  "vllm @ git+https://github.com/vllm-project/vllm.git@v0.32.0",\n]\n'
        )
        self.assertEqual(bp.parse_pinned_vllm_commit(pyproject), "v0.32.0")
        self.assertEqual(
            bp.resolve_vllm_commit_hash("v0.32.0", "override99"), "override99"
        )
        with self.assertRaises(bp.BootstrapError):
            bp.parse_pinned_vllm_commit("[project]\nname = 'vllm-torchtpu'\n")

        # `bootstrap.py vllm-pin` prints the pin on success and exits 1 when missing.
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
            tmp_repo = pathlib.Path(tmp)
            shutil.copytree(_SCRIPTS_DIR, tmp_repo / ".buildkite" / "scripts")
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
        self.assertEqual(default_plan.omni_image_skip, "")

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
        self.assertLessEqual(len(skip_plan.omni_image_skip), 70)

        perf_only = bp.plan_uploads(
            bp.BuildContext.from_env({"CI_LANES": "perf"}),
            merged_labels=(),
            existing_lane_files=_ALL_LANE_FILES,
        )
        self.assertEqual(perf_only.lanes, ("perf",))
        self.assertLessEqual(len(perf_only.omni_image_skip), 70)

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

    def test_shell_wrappers_exec_main_and_dev_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            _install_agent_stubs(bin_dir)
            log_path = tmp_path / "agent.jsonl"

            dev_yaml = tmp_path / "custom_dev.yml"
            dev_yaml.write_text(
                "steps:\n"
                '  - label: "TPU dev step"\n'
                '    if: \'build.env("NIGHTLY") == "1"\'\n'
                "    agents:\n"
                '      queue: "kube"\n'
                "    env:\n"
                '      SHAPE: "tpu7x-standard-1t/1x1x1"\n'
                "    commands:\n"
                "      - echo dev\n",
                encoding="utf-8",
            )
            base_env = {
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                "HOME": str(tmp_path),
                "TEST_AGENT_LOG": str(log_path),
            }

            # 1. `bootstrap.sh` execs `bootstrap.py` (skip-ci path verifies wrapper).
            ci_proc = subprocess.run(
                ["bash", str(_SCRIPTS_DIR / "bootstrap.sh")],
                cwd=_REPO_ROOT,
                env={
                    **base_env,
                    "BUILDKITE_MESSAGE": "chore: quick check [skip ci]",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Commit message contains [skip ci].", ci_proc.stdout)

            # 2. `bootstrap_dev.sh` execs `bootstrap.py dev` with unbuffered groups.
            log_path.unlink(missing_ok=True)
            dev_proc = subprocess.run(
                ["bash", str(_SCRIPTS_DIR / "bootstrap_dev.sh")],
                cwd=_REPO_ROOT,
                env={
                    **base_env,
                    "DEV_PIPELINE_FILE": str(dev_yaml),
                    "DEV_JOB_PRIORITY": "3",
                    "NIGHTLY": "1",
                    "PR_TPU_CONCURRENCY_GROUP": "tpu-pr/dev@example.com",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            lines = dev_proc.stdout.splitlines()
            upload_hdr = lines.index(
                f"--- :pipeline: Uploading {dev_yaml} with priority 3"
            )
            self.assertEqual(lines[upload_hdr + 1], "agent: pipeline upload")
            self.assertEqual(lines[-1], "--- Buildkite Dev Bootstrap Finished")

            events = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            uploads = [e for e in events if "upload_args" in e]
            self.assertEqual(len(uploads), 1)
            self.assertEqual(uploads[0]["upload_args"], ["--no-interpolation"])
            parsed = json.loads(uploads[0]["content"])
            self.assertEqual(parsed["priority"], 3)
            self.assertEqual(parsed["steps"][0]["if"], '"1" == "1"')
            self.assertEqual(
                parsed["steps"][0]["concurrency_group"], "tpu-pr/dev@example.com"
            )

            # 3. Missing `DEV_PIPELINE_FILE` exits 1 with branch/commit message.
            missing_proc = subprocess.run(
                ["bash", str(_SCRIPTS_DIR / "bootstrap_dev.sh")],
                cwd=_REPO_ROOT,
                env={
                    **base_env,
                    "DEV_PIPELINE_FILE": ".buildkite/missing_dev.yml",
                    "BUILDKITE_BRANCH": "exp/missing",
                    "BUILDKITE_COMMIT": "deadbeef",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(missing_proc.returncode, 1)
            self.assertIn(
                "dev pipeline file '.buildkite/missing_dev.yml' not found on "
                "branch 'exp/missing' at commit deadbeef.",
                missing_proc.stdout,
            )


if __name__ == "__main__":
    unittest.main()
