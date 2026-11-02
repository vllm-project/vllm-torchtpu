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
"""Unit tests for .buildkite/scripts/bootstrap.sh and pipeline_config.sh."""

import json
import os
import pathlib
import stat
import subprocess
import tempfile
import unittest

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BOOTSTRAP = _REPO_ROOT / ".buildkite" / "scripts" / "bootstrap.sh"

_FAKE_AGENT = """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${TEST_AGENT_LOG}/calls.log"
if [[ "$1" == "pipeline" && "$2" == "upload" ]]; then
  shopt -s nullglob
  if [[ "${3:-}" == "--dry-run" ]]; then
    dfiles=("${TEST_AGENT_LOG}"/dry_run_*.yml)
    cat > "${TEST_AGENT_LOG}/dry_run_${#dfiles[@]}.yml"
    echo '{"steps": [{"label": "stub", "agents": {"queue": "kube"}}]}'
  else
    ufiles=("${TEST_AGENT_LOG}"/upload_*.yml)
    cat > "${TEST_AGENT_LOG}/upload_${#ufiles[@]}.yml"
  fi
fi
"""

_FAKE_GIT = """#!/usr/bin/env bash
if [[ "${1:-}" == "fetch" ]]; then
  exit 0
fi
if [[ "${1:-}" == "diff" || "${1:-}" == "diff-tree" ]]; then
  echo "src/vllm_torchtpu/runner/tpu_runner.py"
  exit 0
fi
exec /usr/bin/git "$@"
"""

# GitHub's labels API for a merged PR: TEST_GITHUB_LABELS, or a failed request.
_FAKE_CURL = """#!/usr/bin/env bash
if [[ -z "${TEST_GITHUB_LABELS:-}" ]]; then
  exit 22
fi
echo "${TEST_GITHUB_LABELS}"
"""


def run_bootstrap(
    **env: str,
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """Run bootstrap.sh against a stub buildkite-agent."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, script in (
            ("buildkite-agent", _FAKE_AGENT),
            ("git", _FAKE_GIT),
            ("curl", _FAKE_CURL),
        ):
            path = bin_dir / name
            path.write_text(script)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)

        # Provide a clean base environment so host Buildkite vars never leak in.
        run_env = {
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(tmp_path),
            "TEST_AGENT_LOG": str(tmp_path),
            "BUILDKITE_COMMIT": "HEAD",
            "BUILDKITE_BRANCH": "feature-branch",
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_PULL_REQUEST_LABELS": "",
            "BUILDKITE_PIPELINE_SLUG": "vllm-torchtpu-ci",
            **env,
        }
        proc = subprocess.run(
            ["bash", str(_BOOTSTRAP)],
            cwd=_REPO_ROOT,
            env=run_env,
            capture_output=True,
            text=True,
            check=True,
        )
        calls_file = tmp_path / "calls.log"
        calls = calls_file.read_text().splitlines() if calls_file.exists() else []
        uploads = [p.read_text() for p in sorted(tmp_path.glob("upload_*.yml"))]
        dry_runs = [p.read_text() for p in sorted(tmp_path.glob("dry_run_*.yml"))]
        return (
            calls,
            uploads,
            dry_runs,
            {"stdout": proc.stdout, "stderr": proc.stderr},
        )


class TestBootstrap(unittest.TestCase):
    """Verify lane selection, priority, and NIGHTLY substitution in bootstrap.sh."""

    def test_plain_pr_skips_perf_and_keeps_nightly_unset(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready",
        )
        # 3 lane files (verify, integration, tests) + pipeline_build.yml
        self.assertEqual(len(uploads), 4)
        self.assertNotIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertIn("meta-data set JOB_PRIORITY 6", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY pre-merge", calls)
        for content in uploads:
            self.assertTrue(content.startswith("priority: 6\n"))
        for content in uploads[:-1]:
            self.assertIn('build.env("NIGHTLY")', content)
        self.assertFalse(any("annotate" in c for c in calls))

    def test_run_perf_eval_adds_perf_lane_without_nightly(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-perf-eval",
        )
        # 4 lane files (verify, perf, integration, tests) + pipeline_build.yml
        self.assertEqual(len(uploads), 5)
        self.assertIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        self.assertTrue(any("--context perf-eval" in c for c in calls))
        self.assertFalse(any("--context full-ci" in c for c in calls))
        for content in uploads[:-1]:
            self.assertIn('build.env("NIGHTLY")', content)

    def test_run_full_ci_enables_all_lanes_and_substitutes_nightly(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci",
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("Uploading .buildkite/pipeline_perf.yml", out["stdout"])
        # Priority remains pre-merge (6), not nightly (1).
        self.assertIn("meta-data set JOB_PRIORITY 6", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY pre-merge", calls)
        self.assertTrue(any("--context full-ci" in c for c in calls))
        self.assertFalse(any("--context perf-eval" in c for c in calls))
        for content in uploads:
            self.assertTrue(content.startswith("priority: 6\n"))
            self.assertNotIn('build.env("NIGHTLY")', content)
            parsed = yaml.safe_load(content)
            self.assertIsInstance(parsed, dict)
            self.assertIn("steps", parsed)

    def test_run_full_ci_with_oncall_fix_keeps_oncall_priority(self) -> None:
        calls, uploads, _, _ = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci,oncall-fix",
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("meta-data set JOB_PRIORITY 12", calls)
        self.assertIn("meta-data set WORKLOAD_PRIORITY oncall-fix", calls)
        self.assertTrue(any("--context full-ci" in c for c in calls))
        for content in uploads:
            self.assertTrue(content.startswith("priority: 12\n"))
            self.assertNotIn('build.env("NIGHTLY")', content)

    def test_run_full_ci_with_author_caps_tpu_concurrency_with_nightly_substituted(
        self,
    ) -> None:
        _, uploads, dry_runs, _ = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci",
            BUILDKITE_BUILD_AUTHOR_EMAIL="Alice@example.com",
        )
        self.assertEqual(len(uploads), 5)
        self.assertEqual(len(dry_runs), 5)
        for raw_yaml in dry_runs:
            self.assertNotIn('build.env("NIGHTLY")', raw_yaml)
        for content in uploads:
            doc = json.loads(content)
            self.assertEqual(
                doc["steps"][0]["concurrency_group"],
                "tpu-pr/alice@example.com",
            )

    def test_skip_tpu_ci_runs_only_the_tests_lane(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_PULL_REQUEST="123",
            BUILDKITE_PULL_REQUEST_LABELS="ready,run-full-ci,skip-tpu-ci",
        )
        # pipeline_tests.yml + pipeline_build.yml
        self.assertEqual(len(uploads), 2)
        self.assertIn("Uploading .buildkite/pipeline_tests.yml", out["stdout"])
        for lane in ("verify", "perf", "integration"):
            self.assertNotIn(f"Uploading .buildkite/pipeline_{lane}.yml", out["stdout"])
        self.assertTrue(any("--context skip-tpu-ci" in c for c in calls))
        # The TPU group's condition, filled in from SKIP_TPU_CI at upload.
        self.assertIn('"${SKIP_TPU_CI:-0}" != "1"', uploads[0])

    def test_post_merge_of_a_skip_tpu_ci_pr_runs_only_the_tests_lane(self) -> None:
        calls, uploads, _, out = run_bootstrap(
            BUILDKITE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            BUILDKITE_SOURCE="webhook",
            BUILDKITE_MESSAGE="[CI/Build] Fix a script (#123)",
            TEST_GITHUB_LABELS='[{"name": "ready"}, {"name": "skip-tpu-ci"}]',
        )
        self.assertEqual(len(uploads), 2)
        self.assertNotIn("Uploading .buildkite/pipeline_integration.yml", out["stdout"])
        self.assertTrue(any("--context skip-tpu-ci" in c for c in calls))

    def test_post_merge_runs_everything_without_the_label_or_the_lookup(
        self,
    ) -> None:
        for labels in ('[{"name": "ready"}]', ""):
            calls, uploads, _, _ = run_bootstrap(
                BUILDKITE_BRANCH="main",
                BUILDKITE_PULL_REQUEST="false",
                BUILDKITE_SOURCE="webhook",
                BUILDKITE_MESSAGE="[Core] Change something (#124)",
                TEST_GITHUB_LABELS=labels,
            )
            self.assertEqual(len(uploads), 4)
            self.assertFalse(any("--context skip-tpu-ci" in c for c in calls))

    def test_scheduled_nightly_keeps_nightly_priority(self) -> None:
        calls, uploads, _, _ = run_bootstrap(
            BUILDKITE_BRANCH="main",
            BUILDKITE_PULL_REQUEST="false",
            NIGHTLY="1",
        )
        self.assertEqual(len(uploads), 5)
        self.assertIn("meta-data set JOB_PRIORITY 1", calls)
        self.assertFalse(any("WORKLOAD_PRIORITY" in c for c in calls))
        self.assertFalse(any("--context full-ci" in c for c in calls))
        for content in uploads:
            self.assertTrue(content.startswith("priority: 1\n"))
            self.assertNotIn('build.env("NIGHTLY")', content)


if __name__ == "__main__":
    unittest.main()
