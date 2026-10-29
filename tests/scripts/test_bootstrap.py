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
"""Golden parity and unit tests for `.buildkite/scripts/bootstrap.py`."""

from __future__ import annotations

import http.server
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

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / ".buildkite" / "scripts"
_GOLDENS_PATH = (
    pathlib.Path(__file__).resolve().parent / "fixtures" / "bootstrap_goldens.json"
)
_BK_BIN = shutil.which("bk")

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import bootstrap as bp  # noqa: E402

_FAKE_AGENT_BODY = """
import json
import os
import pathlib
import shlex
import sys
import yaml

log_dir = pathlib.Path(os.environ["TEST_LOG_DIR"])
args = sys.argv[1:]


def append_jsonl(name: str, obj: object) -> None:
    with (log_dir / name).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\\n")


append_jsonl("commands.jsonl", shlex.join(["buildkite-agent", *args]))

if args[:2] == ["meta-data", "set"]:
    append_jsonl("metadata.jsonl", {"key": args[2], "value": args[3]})
elif args[:1] == ["annotate"]:
    style = ""
    context = ""
    body_parts = []
    i = 1
    while i < len(args):
        if args[i] == "--style":
            style = args[i + 1]
            i += 2
        elif args[i] == "--context":
            context = args[i + 1]
            i += 2
        else:
            body_parts.append(args[i])
            i += 1
    append_jsonl(
        "annotations.jsonl",
        {"style": style, "context": context, "body": " ".join(body_parts)},
    )
elif args[:2] == ["pipeline", "upload"]:
    content = sys.stdin.read()
    if "--dry-run" in args:
        doc = yaml.safe_load(content) or {}
        clean = {k: v for k, v in doc.items() if not k.startswith("x-")}
        json.dump(clean, sys.stdout)
    else:
        tracked_vars = (
            "JOB_PRIORITY",
            "PR_TPU_CONCURRENCY_GROUP",
            "NIGHTLY",
            "SKIP_TPU_CI",
            "OMNI_IMAGE_SKIP",
        )
        exported = {k: os.environ[k] for k in tracked_vars if k in os.environ}
        append_jsonl(
            "uploads.jsonl",
            {"args": args[2:], "env": exported, "content": content},
        )
"""

_FAKE_GIT_BODY = """
import json
import os
import pathlib
import shlex
import sys

log_dir = pathlib.Path(os.environ["TEST_LOG_DIR"])
args = sys.argv[1:]
with (log_dir / "commands.jsonl").open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(shlex.join(["git", *args])) + "\\n")

if args[:1] == ["fetch"]:
    sys.exit(0)
if args[:1] in (["diff"], ["diff-tree"]):
    diff_lines = json.loads(os.environ.get("TEST_GIT_DIFF_JSON", "[]"))
    if diff_lines:
        sys.stdout.write("\\n".join(diff_lines) + "\\n")
    sys.exit(0)
if args[:1] == ["ls-remote"]:
    head = os.environ.get("TEST_VLLM_UPSTREAM_HEAD", "upstreamhead0000")
    sys.stdout.write(f"{head}\\tHEAD\\n")
    sys.exit(0)
sys.exit(f"Unexpected git command: {args!r}")
"""


def _read_jsonl(path: pathlib.Path) -> list[Any]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_bootstrap_case(
    case_input: dict[str, Any],
    *,
    fixture_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run `bootstrap.sh` against stub `buildkite-agent`, `git`, and GitHub API."""
    labels_map: dict[str, list[str]] = case_input.get("github_labels") or {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        log_dir = tmp_path / "logs"
        bin_dir = tmp_path / "bin"
        log_dir.mkdir()
        bin_dir.mkdir()

        if fixture_files is not None:
            repo_dir = tmp_path / "repo"
            repo_dir.mkdir()
            for rel, content in fixture_files.items():
                dest = repo_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            shutil.copytree(_SCRIPTS_DIR, repo_dir / ".buildkite" / "scripts")
            bootstrap_sh = repo_dir / ".buildkite" / "scripts" / "bootstrap.sh"
        else:
            repo_dir = _REPO_ROOT
            bootstrap_sh = _SCRIPTS_DIR / "bootstrap.sh"

        for name, body in (
            ("buildkite-agent", _FAKE_AGENT_BODY),
            ("git", _FAKE_GIT_BODY),
        ):
            path = bin_dir / name
            path.write_text(f"#!{sys.executable}\n{body.lstrip()}", encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IEXEC)

        # Also put sys.executable's directory on PATH so `python3` resolves to an
        # interpreter with PyYAML when the test runs inside a virtualenv.
        py3_link = bin_dir / "python3"
        if not py3_link.exists():
            py3_link.symlink_to(sys.executable)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                auth = self.headers.get("Authorization") or "none"
                with (log_dir / "commands.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(f"github-api GET {self.path} auth={auth}") + "\n"
                    )
                prefix = "/repos/vllm-project/vllm-torchtpu/issues/"
                suffix = "/labels"
                if self.path.startswith(prefix) and self.path.endswith(suffix):
                    pr_num = self.path[len(prefix) : -len(suffix)]
                    if pr_num in labels_map:
                        payload = json.dumps(
                            [{"name": label} for label in labels_map[pr_num]]
                        ).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            run_env = {
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                "HOME": str(tmp_path),
                "TEST_LOG_DIR": str(log_dir),
                "TEST_GIT_DIFF_JSON": json.dumps(case_input.get("git_diff", [])),
                "TEST_VLLM_UPSTREAM_HEAD": case_input.get(
                    "vllm_upstream_head", "upstreamhead0000"
                ),
                "GITHUB_API_URL": f"http://127.0.0.1:{port}",
                "BUILDKITE_COMMIT": "HEAD",
                "BUILDKITE_BRANCH": "feature-branch",
                "BUILDKITE_PULL_REQUEST": "false",
                "BUILDKITE_PULL_REQUEST_LABELS": "",
                "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci",
                **case_input["env"],
            }
            proc = subprocess.run(
                ["bash", str(bootstrap_sh)],
                cwd=repo_dir,
                env=run_env,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            server.shutdown()
            server.server_close()

        return {
            "exit_code": proc.returncode,
            "commands": _read_jsonl(log_dir / "commands.jsonl"),
            "metadata": _read_jsonl(log_dir / "metadata.jsonl"),
            "annotations": _read_jsonl(log_dir / "annotations.jsonl"),
            "uploads": _read_jsonl(log_dir / "uploads.jsonl"),
        }


class TestBootstrapGoldensAndUnits(unittest.TestCase):
    """Verify exact parity with recorded `main` goldens and pure helper edge cases."""

    maxDiff = None

    def _validate_upload(self, content: str, label: str) -> None:
        if not _BK_BIN:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".yml") as tmp:
            tmp.write(content)
            tmp.flush()
            proc = subprocess.run(
                [_BK_BIN, "pipeline", "validate", "-f", tmp.name],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=f"bk pipeline validate failed for {label}:\n"
                f"{proc.stdout}\n{proc.stderr}",
            )

    def test_matches_recorded_bash_goldens_and_validates_uploads(self) -> None:
        golden_bundle = json.loads(_GOLDENS_PATH.read_text(encoding="utf-8"))
        fixture_files: dict[str, str] = golden_bundle["fixture_files"]
        cases: list[dict[str, Any]] = golden_bundle["cases"]
        self.assertGreaterEqual(len(cases), 13)

        for case in cases:
            with self.subTest(case=case["name"]):
                actual = _run_bootstrap_case(case["input"], fixture_files=fixture_files)
                self.assertEqual(actual, case["expected"])
                for idx, upload in enumerate(actual["uploads"]):
                    self._validate_upload(upload["content"], f"{case['name']}[{idx}]")

    def test_real_repo_uploads_validate_with_bk(self) -> None:
        for name, env in (
            (
                "pr",
                {
                    "BUILDKITE_PULL_REQUEST": "123",
                    "BUILDKITE_PULL_REQUEST_LABELS": "ready",
                    "BUILDKITE_BUILD_AUTHOR_EMAIL": "Alice@example.com",
                },
            ),
            (
                "nightly",
                {
                    "BUILDKITE_BRANCH": "main",
                    "BUILDKITE_SOURCE": "schedule",
                    "NIGHTLY": "1",
                },
            ),
            (
                "integration",
                {
                    "BUILDKITE_BRANCH": "main",
                    "BUILDKITE_SOURCE": "schedule",
                    "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-integration",
                },
            ),
        ):
            with self.subTest(scenario=name):
                res = _run_bootstrap_case(
                    {
                        "env": env,
                        "git_diff": ["src/vllm_torchtpu/runner/tpu_runner.py"],
                    }
                )
                self.assertEqual(res["exit_code"], 0)
                self.assertGreater(len(res["uploads"]), 0)
                for idx, upload in enumerate(res["uploads"]):
                    self._validate_upload(upload["content"], f"real_{name}[{idx}]")

    def test_constants_stay_in_sync_with_pipeline_config_sh(self) -> None:
        config_sh = _SCRIPTS_DIR / "configs" / "pipeline_config.sh"
        names = (
            "PRIORITY_ONCALL_FIX",
            "PRIORITY_POST_MERGE",
            "PRIORITY_PRE_MERGE",
            "PRIORITY_INTEGRATION",
            "PRIORITY_BENCHMARK",
            "PRIORITY_DEFAULT",
            "PRIORITY_NIGHTLY",
            "PRIORITY_KERNEL_TUNING",
            "PR_TPU_JOBS_PER_AUTHOR",
        )
        prints = "; ".join(f'printf "%s=%s\\n" {n} "${n}"' for n in names)
        proc = subprocess.run(
            ["bash", "-c", f'source "{config_sh}" && {prints}'],
            capture_output=True,
            text=True,
            check=True,
        )
        shell_vals = {}
        for line in proc.stdout.splitlines():
            key, val = line.split("=", 1)
            shell_vals[key] = int(val)
        py_vals = {name: getattr(bp, name) for name in names}
        self.assertEqual(py_vals, shell_vals)

    def test_pure_planning_and_credential_helper_edge_cases(self) -> None:
        # .pyrefly-baseline.json is skippable; .pyrefly-baseline-full.json is not.
        self.assertTrue(bp.only_docs_changed([".pyrefly-baseline.json", "README.md"]))
        self.assertFalse(
            bp.only_docs_changed([".pyrefly-baseline-full.json", "README.md"])
        )
        self.assertFalse(bp.only_docs_changed([]))
        self.assertTrue(bp.has_skip_ci_marker("chore: note [ci skip]"))
        self.assertEqual(
            bp.parse_merged_pr_number("feat: add kernel (#789)\nbody (#999)"), "789"
        )

        # Missing unrequested lane warns; missing explicitly requested CI_LANES errors.
        ctx_default = bp.BuildContext.from_env({})
        plan_gap = bp.plan_uploads(
            ctx_default,
            merged_labels=(),
            existing_lane_files={".buildkite/pipeline_tests.yml"},
        )
        self.assertIsNone(plan_gap.missing_ci_lanes_error)
        self.assertEqual(plan_gap.post_lane_annotations[0].context, "lane-gaps")

        ctx_explicit = bp.BuildContext.from_env({"CI_LANES": "perf"})
        plan_missing = bp.plan_uploads(
            ctx_explicit,
            merged_labels=(),
            existing_lane_files={".buildkite/pipeline_tests.yml"},
        )
        self.assertIsNotNone(plan_missing.missing_ci_lanes_error)

        # Credential helper supplies GitHub App token to Authorization header.
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
            helper = pathlib.Path(tmp) / "git-credential-github-app"
            helper.write_text(
                "#!/usr/bin/env bash\n"
                "echo username=x-access-token\n"
                "echo password=ghs_tok\n",
                encoding="utf-8",
            )
            helper.chmod(helper.stat().st_mode | stat.S_IEXEC)
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), AuthHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                effects = bp.BootstrapEffects(
                    repo_root=_REPO_ROOT, credential_helper=helper
                )
                labels = effects.merged_pr_labels(
                    "42",
                    github_api_url=f"http://127.0.0.1:{server.server_address[1]}",
                    env={},
                )
            finally:
                server.shutdown()
                server.server_close()

        self.assertEqual(labels, ("skip-tpu-ci",))
        self.assertEqual(seen_auth, ["token ghs_tok"])


if __name__ == "__main__":
    unittest.main()
